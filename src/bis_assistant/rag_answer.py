"""Create chat responses only from model-generated text."""
from __future__ import annotations

import logging

from .allowlist import KYS_PORTAL, safe_public_url
from .i18n_privacy import find_pii
from .rag_llm import (
    MAX_EVIDENCE_SOURCES,
    evidence_label,
    generate_grounded_answer,
    is_configured,
    is_runtime_identity_query,
    is_underspecified_standard_query,
    last_failure,
)
from .verifier import (
    evidence_type,
    finalize_citations,
    normalize_markers,
    verify_grounded_response,
)

log = logging.getLogger("bis.api")


def source_label(e: dict) -> str:
    """Short display name: the designation for standards, else the title."""
    return str(e.get("standard_number") or e.get("title") or "BIS document")


def format_rag_citation(e: dict) -> str:
    url = safe_public_url(e.get("source_url"), KYS_PORTAL)
    title = e.get("title") or ""
    if evidence_type(e) == "catalogue_record":
        return f"{source_label(e)}: {title} [catalogue record], Source: {url}"
    if e.get("standard_number"):
        return f"{e['standard_number']}: {title} [{e.get('doc_type') or 'document'}], Source: {url}"
    heading = f" › {e['heading']}" if e.get("heading") else ""
    return f"{title}{heading} [{e.get('doc_type') or 'document'}], Source: {url}"


def build_sources(evidence: list[dict], numbered: bool = True) -> list[dict]:
    sources = []
    for ref, e in enumerate(evidence, 1):
        source = {
            "evidence_type": evidence_type(e),
            "label": evidence_label(e),
            "standard_number": e.get("standard_number", ""),
            "title": e.get("title", ""),
            "url": safe_public_url(e.get("source_url"), KYS_PORTAL),
            "doc_type": e.get("doc_type", e.get("type", "")),
            "score": round(float(e.get("score", 0.0) or 0.0), 3),
        }
        if numbered:
            source["ref"] = ref
        if source["evidence_type"] == "catalogue_record":
            source["metadata_only"] = True
            source["department"] = e.get("department", e.get("committee", ""))
            source["date"] = e.get("published_on", e.get("date", ""))
        else:
            source.update({
                "heading": e.get("heading", ""),
                "chunk_text": (e.get("chunk_text") or "")[:1200],
                "chunk_index": e.get("chunk_index", 0),
                "source_file": e.get("source_file", ""),
            })
        for key in ("rank", "selection_reason"):
            if key in e:
                source[key] = e[key]
        sources.append(source)
    return sources


def grounding_failure_response(query: str, lang: str = "en",
                                evidence: list[dict] | None = None) -> dict:
    """Fail closed when one model repair still contains unsupported claims.

    The retrieved documents are still listed as related reading (unnumbered),
    so the user is not left without a lead.
    """
    related = build_sources((evidence or [])[:3], numbered=False)
    text = (
        "मैं उपलब्ध BIS स्रोतों से इस उत्तर की पुष्टि नहीं कर सका। नीचे दिए गए संबंधित दस्तावेज़ देखें, या अपना प्रश्न उत्पाद के नाम या IS संख्या के साथ दोबारा पूछें।"
        if lang == "hi" else
        "I couldn't verify a safe answer from the BIS sources I found. The related "
        "documents below may help, or try asking again with the product name or IS number."
    )
    return {
        "text": text,
        "refused": True,
        "kind": "grounding_refusal",
        "lang": lang,
        "citations": [],
        "sources": [],
        "related_sources": related,
        "rag_evidence": [],
        "rag_mode": "llm_grounding_guard",
        "rag_used_llm": True,
        "model_available": True,
        "pii": find_pii(query),
        "needs_info": False,
        "questions": [],
        "known": [],
        "assumptions": [],
        "context": {"history": [], "rounds": 0},
    }


def model_busy_response(query: str, lang: str = "en") -> dict:
    """Provider rate limit: a retryable state, distinct from being offline."""
    response = model_unavailable_response(query, lang)
    response.update({
        "kind": "model_busy",
        "text": (
            "अभी बहुत अधिक अनुरोध आ रहे हैं। कृपया कुछ सेकंड बाद फिर से पूछें।"
            if lang == "hi" else
            "Manak Mitra is getting a lot of questions right now. Please try again in a few seconds."),
        "rag_mode": "model busy",
        "retryable": True,
    })
    return response


def model_unavailable_response(query: str, lang: str = "en") -> dict:
    """Return a service-state notice, never a substitute answer."""
    text = (
        "चैटबॉट अभी उपलब्ध नहीं है क्योंकि भाषा मॉडल उपलब्ध नहीं है। कृपया बाद में फिर कोशिश करें।"
        if lang == "hi" else
        "The chatbot is offline because its language model is unavailable. Please try again later."
    )
    return {
        "text": text,
        "refused": False,
        "kind": "model_unavailable",
        "lang": lang,
        "citations": [],
        "sources": [],
        "rag_evidence": [],
        "rag_mode": "model unavailable",
        "rag_used_llm": False,
        "model_available": False,
        "pii": find_pii(query),
        "needs_info": False,
        "questions": [],
        "known": [],
        "assumptions": [],
        "context": {"history": [], "rounds": 0},
    }


def _unavailable(query: str, lang: str) -> dict:
    if last_failure() == "http_429":
        return model_busy_response(query, lang)
    return model_unavailable_response(query, lang)


def build_rag_answer(query: str, lang: str, evidence: list[dict],
                     llm_cfg: dict | None = None,
                     history: list[str] | None = None) -> dict:
    """Generate the model answer from retrieved evidence, verify, cite.

    Empty evidence is still sent to the model so it can answer runtime facts
    or say the BIS sources have no support. Retrieved text is never used as
    the answer. Sources returned are exactly the ones the answer cites,
    renumbered [1]..[n] in order of first citation.
    """
    try:
        if not is_configured(llm_cfg):
            return model_unavailable_response(query, lang)
        if is_runtime_identity_query(query) or is_underspecified_standard_query(query):
            evidence = []
        answer_evidence = evidence[:MAX_EVIDENCE_SOURCES]
        text = generate_grounded_answer(
            query, answer_evidence, lang, llm_cfg, history=history)
    except Exception:
        log.exception("LLM answer generation failed")
        return model_unavailable_response(query, lang)

    if not text or not text.strip():
        return _unavailable(query, lang)

    text = normalize_markers(text)
    violations = verify_grounded_response(text, answer_evidence, query)
    if violations:
        log.warning("LLM answer failed grounding validation",
                    extra={"ctx": {"issue_codes": violations}})
        try:
            repaired = generate_grounded_answer(
                query, answer_evidence, lang, llm_cfg, history=history,
                retry_feedback=violations)
        except Exception:
            log.exception("LLM grounding repair failed")
            repaired = None
        repaired = normalize_markers(repaired) if repaired else repaired
        if repaired and not verify_grounded_response(
                repaired, answer_evidence, query):
            text = repaired
        elif repaired is None and last_failure() == "http_429":
            return model_busy_response(query, lang)
        else:
            return grounding_failure_response(query, lang, answer_evidence)

    text, cited = finalize_citations(text.strip(), answer_evidence)
    return {
        "text": text,
        "refused": False,
        "kind": "llm_answer",
        "lang": lang,
        "citations": [format_rag_citation(e) for e in cited],
        "sources": build_sources(cited),
        "rag_evidence": build_sources(cited),
        "rag_mode": "llm",
        "rag_used_llm": True,
        "model_available": True,
        "pii": find_pii(query),
        "needs_info": False,
        "questions": [],
        "known": [],
        "assumptions": [],
        "context": {"history": [], "rounds": 0},
    }
