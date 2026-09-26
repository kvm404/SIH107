"""Single LLM-backed chat path with retrieved BIS evidence as model context."""
from __future__ import annotations

import logging
import re

from . import labs
from .i18n_privacy import detect_lang, redact
from .rag_answer import build_rag_answer
from .rag_llm import (
    MAX_EVIDENCE_SOURCES,
    is_configured,
    is_runtime_identity_query,
    is_underspecified_standard_query,
    needs_rewrite,
    rewrite_query,
)
from .rag_config import load_llm_config, load_rag_config
from .verifier import strip_markers
from . import threads as threadmod

log = logging.getLogger("bis.api")

_DIAGNOSTIC_REASON_CODES = {
    "below_relevance_threshold": "relevance_threshold",
    "insufficient_query_term_overlap": "missing_query_terms",
    "no_relevance_overlap": "no_lexical_match",
    "top_k_limit": "outside_top_k",
    "combined_top_k_limit": "outside_top_k",
    "catalogue_limit": "outside_top_k",
    "duplicate_chunk": "duplicate_source",
    "duplicate_evidence": "duplicate_source",
    "source_diversity_limit": "source_diversity",
    "retrieval_error": "not_selected",
    "retrieval_disabled": "not_selected",
}


def _safe_diagnostic_branch(*, enabled: bool, evidence: list[dict],
                            document_searched: bool,
                            catalogue_searched: bool) -> str:
    if not enabled:
        return "disabled"
    if not evidence:
        return "no_results"
    if document_searched and catalogue_searched:
        return "hybrid"
    if document_searched:
        return "document"
    if catalogue_searched:
        return "catalogue"
    return "no_results"


def _diagnostic_reason_counts(reasons: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for reason, count in reasons.items():
        code = _DIAGNOSTIC_REASON_CODES.get(reason, "not_selected")
        normalized[code] = normalized.get(code, 0) + int(count)
    return normalized


def _rag_lookup(text: str, top_k: int | None = None,
                _conn=None) -> tuple[list[dict], dict]:
    """Fetch relevant document and catalogue evidence without rendering it.

    Catalogue records are explicitly tagged as metadata-only context. The
    bounded diagnostics contain retrieval metadata, never the query or passage
    text.
    """
    cfg: dict = {}
    try:
        from .rag_retriever import search_rag
        from .catalogue_search import search_catalogue

        cfg = load_rag_config()
        if not cfg.get("enabled"):
            cfg["retrieval_diagnostics"] = {
                "branch": "disabled",
                "branches": {}, "selected_count": 0, "rejected_count": 0,
                "rejection_reasons": {"not_selected": 1},
                "results": [], "selected": [],
            }
            return [], cfg
        final_top_k = max(1, int(top_k or cfg.get("top_k", 5)))
        minimum_relevance = max(0.0, float(cfg.get("minimum_relevance", 0.25)))
        catalogue_top_k = max(1, int(cfg.get("catalogue_top_k", final_top_k)))
        document_diag: dict = {}
        catalogue_diag: dict = {}
        try:
            documents = search_rag(
                text,
                top_k=final_top_k,
                db_path=cfg.get("db_path"),
                lexical_weight=cfg.get("weight_lexical", 1.0),
                semantic_weight=cfg.get("weight_semantic", 0.3),
                exact_boost=cfg.get("exact_boost", 50.0),
                semantic=cfg.get("semantic", True),
                embedding_model=cfg.get("embedding_model", ""),
                minimum_relevance=minimum_relevance,
                diagnostics=document_diag,
                _conn=_conn,
            )
        except Exception:
            log.exception("BIS document retrieval branch failed")
            documents = []
            document_diag = {"branch": "document_chunks", "candidate_count": 0,
                             "selected_count": 0, "rejected_count": 0,
                             "rejection_reasons": {"retrieval_error": 1},
                             "selected": []}
        try:
            catalogue = search_catalogue(
                text, db_path=cfg.get("db_path"), limit=catalogue_top_k,
                minimum_relevance=minimum_relevance, diagnostics=catalogue_diag,
                _conn=_conn,
            )
        except Exception:
            log.exception("BIS catalogue retrieval branch failed")
            catalogue = []
            catalogue_diag = {"branch": "catalogue_records", "candidate_count": 0,
                              "selected_count": 0, "rejected_count": 0,
                              "rejection_reasons": {"retrieval_error": 1},
                              "selected": []}

        # Normalize catalogue rows to the same evidence contract as document
        # chunks. The explicit limits are also consumed by the answer prompt.
        catalogue_evidence = [{
            **item,
            "evidence_type": "catalogue_record",
            "doc_type": item.get("type_name") or "catalogue metadata",
            "heading": "",
            "chunk_text": "",
            "source_file": "",
            "metadata_only": True,
            "metadata_notice": "Catalogue metadata only; full standard text was not retrieved.",
            "metadata_scope": ["designation", "title", "department", "type", "date"],
            "unsupported_claims": [
                "clause-level scope", "technical requirements",
                "current legal applicability", "QCO or certification",
                "compliance", "product suitability",
            ],
            "selection_reason": (
                "exact_designation" if item.get("exact_match") else
                "relevance_threshold_passed"),
        } for item in catalogue]

        candidates = [*documents, *catalogue_evidence]
        candidates.sort(key=lambda item: (
            -float(item.get("rank_score", item.get("relevance", 0.0))),
            not bool(item.get("exact_match")),
            item.get("evidence_type") != "document_chunk",
            -float(item.get("score", 0.0)),
        ))
        evidence: list[dict] = []
        reasons: dict[str, int] = {}
        merge_reasons: dict[str, dict[str, int]] = {}
        seen: set[tuple] = set()
        per_standard: dict[str, int] = {}
        # Count branch-level rejections already recorded by the retrievers.
        for branch_diag in (document_diag, catalogue_diag):
            for reason, count in branch_diag.get("rejection_reasons", {}).items():
                reasons[reason] = reasons.get(reason, 0) + int(count)
        for candidate_index, item in enumerate(candidates):
            branch = item.get("evidence_type", "document_chunk")
            if len(evidence) >= final_top_k:
                remaining = candidates[candidate_index:]
                reasons["combined_top_k_limit"] = len(remaining)
                for skipped in remaining:
                    skipped_branch = skipped.get("evidence_type", "document_chunk")
                    branch_reasons = merge_reasons.setdefault(skipped_branch, {})
                    branch_reasons["combined_top_k_limit"] = (
                        branch_reasons.get("combined_top_k_limit", 0) + 1)
                break
            if item.get("evidence_type") == "catalogue_record":
                key = ("catalogue", item.get("standard_id") or item.get("standard_number"))
            else:
                key = ("document", item.get("chunk_id") or item.get("doc_id"),
                       item.get("chunk_index"))
            if key in seen:
                reasons["duplicate_evidence"] = reasons.get("duplicate_evidence", 0) + 1
                branch_reasons = merge_reasons.setdefault(branch, {})
                branch_reasons["duplicate_evidence"] = (
                    branch_reasons.get("duplicate_evidence", 0) + 1)
                continue
            standard = item.get("standard_number") or str(key[1])
            if per_standard.get(standard, 0) >= 2:
                reasons["source_diversity_limit"] = reasons.get(
                    "source_diversity_limit", 0) + 1
                branch_reasons = merge_reasons.setdefault(branch, {})
                branch_reasons["source_diversity_limit"] = (
                    branch_reasons.get("source_diversity_limit", 0) + 1)
                continue
            seen.add(key)
            per_standard[standard] = per_standard.get(standard, 0) + 1
            evidence.append(item)
        evidence = evidence[:final_top_k]
        for rank, item in enumerate(evidence, 1):
            item["rank"] = rank
            item["relevance_reason"] = item.get(
                "selection_reason", "relevance_threshold_passed")
        branch_counts = {
            "document_chunks": int(document_diag.get("candidate_count", 0)),
            "catalogue_records": int(catalogue_diag.get("candidate_count", 0)),
        }
        branch_selected = {
            branch: sum(1 for item in evidence if item.get("evidence_type") == evidence_type)
            for branch, evidence_type in (
                ("document_chunks", "document_chunk"),
                ("catalogue_records", "catalogue_record"),
            )
        }
        total_candidates = sum(branch_counts.values())
        branch_results = {}
        for branch in ("document_chunks", "catalogue_records"):
            branch_reasons = dict(
                (document_diag if branch == "document_chunks" else catalogue_diag)
                .get("rejection_reasons", {}))
            for reason, count in merge_reasons.get(
                    "document_chunk" if branch == "document_chunks" else
                    "catalogue_record", {}).items():
                branch_reasons[reason] = branch_reasons.get(reason, 0) + count
            branch_results[branch] = {
                "candidate_count": branch_counts[branch],
                "selected_count": branch_selected[branch],
                "rejected_count": max(0, branch_counts[branch] - branch_selected[branch]),
                "rejection_reasons": branch_reasons,
            }
        cfg["retrieval_diagnostics"] = {
            "branch": _safe_diagnostic_branch(
                enabled=True, evidence=evidence,
                document_searched=bool(document_diag),
                catalogue_searched=bool(catalogue_diag)),
            "branches": branch_results,
            "selected_count": len(evidence),
            "rejected_count": max(0, total_candidates - len(evidence)),
            "rejection_reasons": _diagnostic_reason_counts(reasons),
            "results": [
                {"rank": rank, "evidence_type": item.get("evidence_type", "document_chunk"),
                 "score": round(float(item.get("relevance", 0.0)), 6),
                 "source_id": str(item.get("standard_number") or "")[:64],
                 "selected": True}
                for rank, item in enumerate(evidence[:20], 1)
            ],
            "selected": [
                {"branch": item.get("evidence_type", "document_chunk"),
                 "rank": rank,
                 "score": round(float(item.get("relevance", 0.0)), 6),
                 "reason": item.get("selection_reason", "relevance_threshold_passed")}
                for rank, item in enumerate(evidence[:12], 1)
            ],
        }
        return evidence, cfg
    except Exception:
        log.exception("BIS corpus retrieval failed; the model will receive no evidence")
        return [], {"retrieval_diagnostics": {
            "branch": "no_results", "branches": {},
            "selected_count": 0, "rejected_count": 0,
            "rejection_reasons": {"not_selected": 1},
            "results": [], "selected": [],
        }}


# The process page to pair with a compulsory-list hit, so "what do I do next"
# is grounded in the right scheme: (list file, process file, section heading).
_SCHEME_PROCESS = (
    ("knowledge/generated/compulsory-scheme-i.md",
     "knowledge/curated/licence-process.md", "Option 2: simplified procedure with third party test reports"),
    ("knowledge/generated/compulsory-scheme-ii-crs.md",
     "knowledge/generated/crs.md", "CRS registration steps"),
    ("knowledge/generated/upcoming-qcos.md",
     "knowledge/curated/licence-process.md", "Application"),
)


_CRS_PROCESS_RE = re.compile(
    r"\b(?:crs|compulsory registration)\b.*\b(?:how|steps?|apply|register|"
    r"registration|procedure|process|kaise)\b"
    r"|\b(?:how|steps?|apply|register|procedure|process|kaise)\b.*"
    r"\b(?:crs|compulsory registration)\b", re.IGNORECASE)


def _scheme_process_evidence(evidence: list[dict], query: str = "",
                             _conn=None) -> list[dict]:
    """One process chunk for the scheme of the top compulsory-list hit, or
    for CRS when the question asks how to register under it."""
    import sqlite3

    top = next((e for e in evidence if e.get("doc_type") == "compulsory_list"), None)
    source_file = top.get("source_file", "") if top else ""
    if _CRS_PROCESS_RE.search(query or ""):
        source_file = _SCHEME_PROCESS[1][0]
    match = next((m for m in _SCHEME_PROCESS if m[0] == source_file), None)
    if match is None:
        return []
    # Rows past the budget are dropped, so only the kept ones count.
    if any(e.get("source_file") == match[1] and e.get("heading") == match[2]
           for e in evidence[:MAX_EVIDENCE_SOURCES - 1]):
        return []
    cfg = load_rag_config()
    conn = None
    try:
        conn = _conn or sqlite3.connect(str(cfg.get("db_path")))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT c.id, c.chunk_index, c.chunk_text, c.heading, c.doc_type, c.source_url,"
            " d.title, d.source_file, d.id AS doc_id FROM corpus_chunks c"
            " JOIN corpus_documents d ON d.id = c.doc_id"
            " WHERE d.source_file=? AND c.heading=? ORDER BY c.chunk_index LIMIT 1",
            (match[1], match[2])).fetchone()
    except Exception:
        log.exception("scheme process lookup failed")
        return []
    finally:
        if conn is not None and _conn is None:
            conn.close()
    if row is None:
        return []
    return [{
        "evidence_type": "document_chunk", "doc_type": row["doc_type"],
        "standard_number": "", "title": row["title"], "heading": row["heading"],
        "chunk_text": row["chunk_text"], "source_url": row["source_url"],
        "source_file": row["source_file"], "chunk_id": row["id"], "doc_id": row["doc_id"],
        "chunk_index": row["chunk_index"], "relevance": 0.5, "rank_score": 0.5,
        "selection_reason": "scheme_process",
    }]


def _with_lab_evidence(query: str, evidence: list[dict]) -> list[dict]:
    """Prepend LIMS lab rows for lab questions ("where can I test helmets?").

    The product's IS number comes from the query or from a product-only
    search, so words like "lab" and "Pune" do not steer that search.
    """
    if not labs.is_lab_query(query):
        return evidence
    pool = list(evidence)
    if not labs._IS_RE.search(query):
        product_query = labs.product_query(query)
        if product_query:
            extra, _ = _rag_lookup(product_query, top_k=8)
            pool = extra + pool
    lab_rows = labs.lab_evidence(query, pool)
    if not lab_rows:
        return evidence
    return (lab_rows + evidence)[:MAX_EVIDENCE_SOURCES]


def answer(query: str, lang: str | None = None,
           context: dict | None = None) -> dict:
    """Return an LLM answer or an explicit model-unavailable state.

    Every valid chat turn uses one model generation (plus a small query
    rewrite for Hindi or follow-up questions). BIS evidence only enters the
    prompt as context; retrieved text is never rendered as the answer.
    """
    q = (query or "").strip()
    resolved_lang = lang if lang in ("en", "hi") else detect_lang(q)
    turn_context = threadmod.normalize_context(context)
    history = turn_context["history"]

    try:
        llm_cfg = load_llm_config()
        configured = is_configured(llm_cfg)
    except Exception:
        log.exception("Could not load LLM configuration")
        llm_cfg = {}
        configured = False

    evidence: list[dict] = []
    rag_cfg: dict = {}
    search_query = q
    if configured and not is_runtime_identity_query(q) \
            and not is_underspecified_standard_query(q):
        if needs_rewrite(q, resolved_lang, history):
            try:
                search_query = rewrite_query(q, history, llm_cfg) or q
            except Exception:
                log.exception("query rewrite failed; searching the original text")
        evidence, rag_cfg = _rag_lookup(search_query)
        try:
            evidence = _with_lab_evidence(search_query, evidence)
        except Exception:
            log.exception("lab lookup failed; answering without lab rows")
        process = _scheme_process_evidence(evidence, search_query)
        if process:
            # Replace the weakest row so the evidence budget stays fixed.
            evidence = (evidence[:MAX_EVIDENCE_SOURCES - 1] + process)

    response = build_rag_answer(
        q,
        resolved_lang,
        evidence,
        llm_cfg,
        history=history,
    )
    new_history = history
    if q:
        new_history = threadmod.push_turn(
            history, redact(q)[:2000],
            redact(strip_markers(response.get("text", "")))[:600]
            if response.get("kind") == "llm_answer" else "")
    response["context"] = {
        "history": new_history,
        "rounds": turn_context["rounds"],
        "force": turn_context["force"],
    }
    response["search_query"] = search_query if search_query != q else ""
    response["intent"] = "general"
    response["intent_confidence"] = "low"
    response["context_summary"] = ""
    if rag_cfg.get("retrieval_diagnostics") is not None:
        response["retrieval_diagnostics"] = rag_cfg["retrieval_diagnostics"]
    return response
