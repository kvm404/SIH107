"""Hybrid retrieval: exact IS boost, FTS5/BM25, optional dense search and reranking.

Fuse lexical and dense ranks; return source-bearing evidence for generation.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from . import rag_embeddings as emb

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FTS_RESERVED = re.compile(r"[\":*^()]")
_STOP = frozenset("""
what does the cover about which with from that this give summary scope standard
standards indian tell please explain specification requirements requirement
information info is are was were do does did done for on in of to a an and or my
i me we you your yours how when where who whom it its these those by as at be
been being have has had will would can could should suggest recommend suitable
applicable product startup manufacturing manufacture makes make want start
starting company care need needed needs required require necessary get
""".split())


# Everyday words mapped to the vocabulary BIS documents use. Applied to query
# terms only; both sides still go through _normalise_token.
_SYNONYMS = {
    "mandatory": ("compulsory",), "compulsory": ("mandatory",),
    "lab": ("laboratory",), "labs": ("laboratory",), "laboratory": ("lab",),
    "bulb": ("lamp",), "bulbs": ("lamp",), "fridge": ("refrigerating", "refrigerator"),
    "phone": ("mobile",), "mobile": ("phone",), "licence": ("license",),
    "license": ("licence",), "complain": ("complaint",), "fake": ("misuse",),
    "jewelry": ("jewellery",), "jeweler": ("jeweller",), "cert": ("certification",),
}

# Retrieval weight per evidence type. Official guidance pages and the
# compulsory-product lists answer most user questions directly; gazette
# schedules list many unrelated standards per chunk and rank last.
DOC_TYPE_WEIGHTS = {
    "bis_guide": 1.25, "compulsory_list": 1.25, "lab_directory": 1.15,
    "product_manual": 1.0, "catalogue": 0.9, "gazette": 0.55,
}
KNOWLEDGE_DOC_TYPES = ("bis_guide", "compulsory_list", "lab_directory")


def doc_type_weight(doc_type: str) -> float:
    return DOC_TYPE_WEIGHTS.get((doc_type or "").lower(), 0.8)


def extract_is_numbers(query: str) -> list[str]:
    """Raw IS designations found in text, e.g. ['IS 101 (Part 2/Sec 6):2026'].

    Hyphenated forms (`IS-10500`) are accepted; use
    ``retriever.normalize_is_ref`` for canonical comparison.
    """
    out = []
    for m in re.finditer(
            r"IS\s*-?\s*\d+(?:\s*\([^)]*\))?\s*(?::\s*\d{4})?", query, re.IGNORECASE):
        out.append(re.sub(r"\s+", " ", m.group(0).strip()))
    return out


def _parse_is_parts(ref: str) -> tuple[str, str, str]:
    """Split a designation into (base, part, sec), e.g. IS 101 (Part 2/Sec 6).

    Handles `IS 302-1` hyphen parts and `IS/IEC ...` prefixes. Year is
    ignored: retrieval matches designations, not editions.
    """
    n = _normalize_is(ref)
    base, part, sec = "", "", ""
    m = re.search(r"IS(?:/[A-Z]+)?\s*(\d+)", n)
    if m:
        base = m.group(1)
    else:
        return base, part, sec
    mp = re.search(r"PART\s*([A-Z0-9]+)", n)
    if mp:
        part = mp.group(1)
    else:
        mh = re.search(r"IS(?:/[A-Z]+)?\s*\d+\s*-\s*([A-Z0-9]+)", n)
        if mh:
            part = mh.group(1)
    ms = re.search(r"SEC(?:TION|\.)?\s*([A-Z0-9]+)", n)
    if ms:
        sec = ms.group(1)
    return base, part, sec


def _is_boost(std_num: str, query_refs: list[str], exact_boost: float) -> tuple[float, bool]:
    """Tiered IS boost (issue #4 P1-7): full designation > base+part > base.

    Any tier counts as an exact match (preserves base-query diversion);
    the multiplier separates `IS 101 (Part 2/Sec 6)` from its siblings
    instead of boosting every Part/Sec variant equally.
    """
    if not query_refs or not std_num:
        return 0.0, False
    sb, sp, ss = _parse_is_parts(std_num)
    if not sb:
        return 0.0, False
    best = 0.0
    for qr in query_refs:
        qb, qp, qs = _parse_is_parts(qr)
        if not qb or qb != sb:
            continue
        if qp and qp == sp and (not qs or not ss or qs == ss):
            best = max(best, exact_boost * 1.5)  # full designation
        elif qp and qp == sp:
            best = max(best, exact_boost * 1.25)  # same part, other section
        elif not qp:
            # Query names the base only: full marks when the doc is also
            # part-less, base marks when it refines into parts/sections.
            best = max(best, exact_boost * 1.5 if not sp else exact_boost)
        else:
            best = max(best, exact_boost * 0.5)  # same base, other part
    return (best, True) if best > 0 else (0.0, False)


def _normalize_is(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").upper().replace("–", "-").replace("—", "-"))
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    return s.strip()


def _fts_query(query: str) -> str:
    toks = [t for t in _TOKEN_RE.findall(query.lower())
            if len(t) > 2 and _normalise_token(t) not in _STOP]
    toks += [syn for t in list(toks) for syn in _SYNONYMS.get(t, ())]
    # keep IS digits glued: 'IS 101' -> 'IS101' token variant too
    extra = []
    for m in extract_is_numbers(query):
        extra.append("IS" + re.sub(r"\D", "", m))
    # FTS tokenizes punctuation in corpus designations (`IS 14543 : 2016`)
    # into separate words. Search the standard number as an exact phrase so
    # common terms such as `packing` cannot bury the requested designation.
    exact_numbers = [re.search(r"\d+", ref).group(0) for ref in extract_is_numbers(query)]
    exact_clause = " OR ".join(f'"{n}"' for n in exact_numbers if n)
    toks = toks + extra
    # de-dup, cap length, quote phrases safely
    seen, out = set(), []
    for t in toks:
        t = _FTS_RESERVED.sub("", t).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(f'"{t}"')
        if len(out) >= 16:
            break
    ordinary = " OR ".join(out)
    return f"({ordinary}) OR ({exact_clause})" if exact_clause and ordinary else (
        exact_clause or ordinary or '""')


def _token_overlap_score(query: str, text: str) -> float:
    q = _query_terms(query)
    d = _normalised_terms(text)
    if not q or not d:
        return 0.0
    return len(q & d) / (len(q) ** 0.5)


def _normalise_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _query_terms(query: str) -> set[str]:
    terms = set()
    for token in _TOKEN_RE.findall(query.lower()):
        if len(token) > 2 and token not in _STOP:
            terms.add(_normalise_token(token))
    return terms


def _expanded_terms(query_terms: set[str]) -> set[str]:
    out = set(query_terms)
    for term in query_terms:
        out.update(_normalise_token(s) for s in _SYNONYMS.get(term, ()))
    return out


def _normalised_terms(text: str) -> set[str]:
    return {_normalise_token(token) for token in _TOKEN_RE.findall(text.lower())
            if len(token) > 2}


def _best_local_window(query_terms: set[str], text: str,
                       window_size: int = 12) -> set[str]:
    """Return the tokens of the body-text window with the most query terms."""
    best: set[str] = set()
    best_hits = 0
    for segment in re.split(r"(?<=[.!?;:])\s+|[\r\n]+", text or ""):
        tokens = [_normalise_token(token) for token in _TOKEN_RE.findall(segment.lower())]
        for start in range(max(1, len(tokens) - window_size + 1)):
            window = set(tokens[start:start + window_size])
            hits = len(query_terms & window)
            if hits > best_hits:
                best, best_hits = window, hits
    return best


def _max_local_overlap(query_terms: set[str], text: str,
                       window_size: int = 12) -> int:
    """Return the strongest query-term support in a short body-text window."""
    return len(query_terms & _best_local_window(query_terms, text, window_size))


MAX_RELEVANCE_TERMS = 8


def _chunk_relevance(query_terms: set[str], row: dict,
                     semantic_score: float = 0.0) -> tuple[float, bool, int]:
    """Return (relevance, enough_terms, title_heading_hits).

    Synonyms can satisfy a query term (``mandatory`` matches ``compulsory``)
    but each original term counts at most once.
    """
    def matched(tokens: set[str]) -> set[str]:
        return {term for term in query_terms
                if term in tokens or any(_normalise_token(s) in tokens
                                         for s in _SYNONYMS.get(term, ()))}

    title_terms = matched(_normalised_terms(
        f"{row.get('title', '')} {row.get('heading', '')}"))
    body_terms = matched(_best_local_window(_expanded_terms(query_terms),
                                            row.get("chunk_text", "")))
    covered = title_terms | body_terms
    n = max(1, len(query_terms))
    # Long, multi-question messages would otherwise dilute every passage
    # below the threshold, so coverage is measured against at most
    # MAX_RELEVANCE_TERMS terms.
    relevance = max(min(1.0, len(covered) / min(n, MAX_RELEVANCE_TERMS)),
                    max(0.0, semantic_score))
    # Body-only support needs three terms in one short window; a title or
    # heading hit lowers that to two terms overall.
    enough = bool(query_terms) and (
        len(body_terms) >= min(3, n)
        or len(title_terms) >= min(2, n)
        or (title_terms and len(covered) >= min(2, n)))
    return relevance, enough, len(title_terms)


def _finish_diagnostics(diagnostics: dict | None, *, candidate_count: int,
                        selected: list[dict], reasons: dict[str, int]) -> None:
    if diagnostics is None:
        return
    diagnostics.update({
        "branch": "document_chunks",
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "rejected_count": max(0, candidate_count - len(selected)),
        "rejection_reasons": reasons,
        "selected": [
            {"rank": rank, "score": round(float(item.get("relevance", 0.0)), 6),
             "reason": item.get("selection_reason", "relevance_threshold_passed")}
            for rank, item in enumerate(selected[:12], 1)
        ],
    })


def search_rag(query: str, top_k: int = 5,
               db_path: str | Path | None = None,
               lexical_weight: float = 1.0, semantic_weight: float = 0.3,
               exact_boost: float = 50.0, semantic: bool = True,
               embedding_model: str = "",
               _conn: sqlite3.Connection | None = None,
               minimum_relevance: float = 0.25,
               diagnostics: dict | None = None) -> list[dict]:
    """Hybrid search. Empty list when DB missing/empty (never raises).

    Pass ``_conn`` to reuse one connection per request (issue #4 P1-10);
    caller-owned connections are never closed here.
    """
    from .rag_config import load_rag_config
    cfg = load_rag_config()
    db_path = str(db_path or cfg["db_path"])
    top_k = max(1, int(top_k or cfg["top_k"]))
    minimum_relevance = max(0.0, float(minimum_relevance))
    if not os.path.exists(db_path):
        _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
        return []
    q = (query or "").strip()
    if not q:
        _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
        return []
    query_terms = _query_terms(q)
    if not query_terms and not extract_is_numbers(q):
        _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
        return []
    q_is = extract_is_numbers(q)
    fts_q = _fts_query(q)
    candidate_limit = min(max(top_k * 20, 50), 300)

    own_conn = _conn is None
    if _conn is not None:
        conn = _conn
    else:
        if not os.path.exists(db_path):
            return []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
        except Exception:
            return []
    try:
        # Does the corpus exist?
        try:
            n = conn.execute("SELECT COUNT(*) c FROM corpus_chunks").fetchone()["c"]
        except Exception:
            _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
            return []
        if not n:
            _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
            return []
        lexical_rows: list[dict] = []
        used_fts = False
        try:
            cur = conn.execute(
                "SELECT c.id, c.doc_id, c.chunk_index, c.chunk_text, c.heading,"
                " c.char_start, c.char_end, c.token_count, c.standard_number,"
                " c.doc_type, c.source_url, bm25(corpus_chunks_fts) AS rank"
                " FROM corpus_chunks_fts JOIN corpus_chunks c ON c.id = corpus_chunks_fts.rowid"
                " WHERE corpus_chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_q, candidate_limit))
            for r in cur.fetchall():
                lexical_rows.append(dict(r))
            used_fts = True
        except Exception:
            pass

        model_name = embedding_model or cfg.get("embedding_model", "")
        dense_hits = (emb.dense_search(db_path, q, model_name, candidate_limit)
                      if semantic and model_name else [])
        if semantic and model_name and not dense_hits and lexical_rows:
            model = emb.get_model(model_name)
            if model is not None:
                try:
                    query_vector = model.encode([q], normalize_embeddings=True)[0]
                    texts = [row.get("chunk_text", "")[:2000] for row in lexical_rows]
                    vectors = model.encode(texts, normalize_embeddings=True)
                    dense_hits = sorted(
                        [(int(row["id"]), emb.cosine(query_vector, vector))
                         for row, vector in zip(lexical_rows, vectors, strict=True)],
                        key=lambda item: -item[1],
                    )
                except Exception:
                    dense_hits = []
        dense_scores = dict(dense_hits)
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_hits, 1)}
        rows_by_id = {int(row["id"]): row for row in lexical_rows}
        missing_ids = [chunk_id for chunk_id, _ in dense_hits if chunk_id not in rows_by_id]
        if missing_ids:
            placeholders = ",".join("?" for _ in missing_ids)
            for row in conn.execute(
                    f"SELECT * FROM corpus_chunks WHERE id IN ({placeholders})", missing_ids):
                item = dict(row)
                item["rank"] = 0.0
                rows_by_id[int(item["id"])] = item

        if not rows_by_id:
            # FTS5 can be missing or have no lexical hit. Keep a small LIKE
            # fallback for minimal SQLite builds and empty dense indexes.
            toks = sorted(query_terms)[:8]
            if not toks:
                _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
                return []
            where = " OR ".join(["chunk_text LIKE ?"] * len(toks))
            params = [f"%{t}%" for t in toks]
            try:
                cur = conn.execute(
                    f"SELECT c.*, 0.0 AS rank FROM corpus_chunks c WHERE {where} LIMIT ?",
                    (*params, candidate_limit))
                lexical_rows = [dict(row) for row in cur.fetchall()]
                rows_by_id.update({int(row["id"]): row for row in lexical_rows})
                used_fts = False
            except Exception:
                _finish_diagnostics(diagnostics, candidate_count=0, selected=[], reasons={})
                return []

        lexical_scores = {}
        for row in lexical_rows:
            try:
                value = -float(row.get("rank", 0.0) or 0.0) if used_fts else 0.0
            except (TypeError, ValueError):
                value = 0.0
            if not used_fts:
                value = _token_overlap_score(q, row.get("chunk_text", "")) * 10.0
            lexical_scores[int(row["id"])] = value
        lexical_ranks = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(
                sorted(lexical_scores.items(), key=lambda item: -item[1]), 1)
        }

        # Enrich the union of lexical and dense candidates with source metadata.
        doc_cache: dict = {}
        scored: list[dict] = []
        for chunk_id, r in rows_by_id.items():
            doc_id = r.get("doc_id")
            if doc_id not in doc_cache:
                try:
                    d = conn.execute(
                        "SELECT * FROM corpus_documents WHERE id=?", (doc_id,)).fetchone()
                    doc_cache[doc_id] = dict(d) if d else {}
                except Exception:
                    doc_cache[doc_id] = {}
            d = doc_cache[doc_id]
            std_num = r.get("standard_number") or d.get("standard_number", "")
            # Tiered IS boost (P1-7): full designation > base+part > base.
            boost, is_exact = _is_boost(std_num, q_is, exact_boost)
            lex = lexical_scores.get(chunk_id, 0.0)
            sem = dense_scores.get(chunk_id, 0.0)
            fused = lexical_weight * lex + boost + semantic_weight * sem * 10.0
            rrf = (lexical_weight / (60 + lexical_ranks[chunk_id])
                   if chunk_id in lexical_ranks else 0.0)
            rrf += (semantic_weight / (60 + dense_ranks[chunk_id])
                    if chunk_id in dense_ranks else 0.0)
            scored.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": r.get("chunk_index", 0),
                "chunk_text": r.get("chunk_text", ""),
                "heading": r.get("heading", ""),
                "char_start": r.get("char_start", 0),
                "char_end": r.get("char_end", 0),
                "standard_number": std_num,
                "title": d.get("title", ""),
                "department": d.get("department", ""),
                "committee": d.get("committee", ""),
                "doc_type": r.get("doc_type") or d.get("doc_type", ""),
                "source_url": r.get("source_url") or d.get("source_url", ""),
                "source_file": d.get("source_file", ""),
                "extraction_method": d.get("extraction_method", ""),
                "score": fused,
                "lexical": lex,
                "semantic": sem,
                "relevance": 0.0,
                "rrf_score": rrf,
                "exact_boost": boost,
                "exact_match": is_exact,
            })
        ordered = sorted(scored, key=lambda item: (
            not item["exact_match"], -item["rrf_score"], -item["score"]))

        rerank_model = cfg.get("reranker_model", "")
        pool = ordered[:candidate_limit]
        rerank_scores = emb.rerank_scores(
            q, [item["chunk_text"][:2000] for item in pool], rerank_model)
        if rerank_scores is not None:
            for item, score in zip(pool, rerank_scores, strict=True):
                item["rerank_score"] = score
            ordered = sorted(ordered, key=lambda item: (
                not item["exact_match"],
                -item.get("rerank_score", -1.0),
                -item["rrf_score"],
                -item["score"],
            ))
        reasons: dict[str, int] = {}
        candidate_count = len(ordered)
        relevant: list[dict] = []
        for item in ordered:
            relevance, enough_terms, title_hits = _chunk_relevance(
                query_terms, item, float(item.get("semantic", 0.0) or 0.0))
            item["relevance"] = round(relevance, 6)
            item["title_hits"] = title_hits
            item["rank_score"] = round(
                relevance * doc_type_weight(item.get("doc_type", ""))
                + 0.05 * title_hits + (0.5 if item.get("exact_match") else 0.0), 6)
            exact = bool(item.get("exact_match"))
            if exact or (relevance >= minimum_relevance and enough_terms):
                item["evidence_type"] = "document_chunk"
                item["selection_reason"] = (
                    "exact_designation" if exact else "relevance_threshold_passed")
                relevant.append(item)
            else:
                reason = "below_relevance_threshold"
                if relevance >= minimum_relevance and not enough_terms:
                    reason = "insufficient_query_term_overlap"
                reasons[reason] = reasons.get(reason, 0) + 1

        relevant.sort(key=lambda item: (-item["rank_score"], -item["rrf_score"],
                                        -item["score"]))
        # Remove duplicate chunk text (the same gazette schedule is attached to
        # several standards) and limit repeated passages from one source so it
        # cannot crowd out other relevant sources.
        selected: list[dict] = []
        seen_chunks: set[str] = set()
        per_standard: dict[str, int] = {}
        for item in relevant:
            standard = item.get("standard_number") or f"doc:{item.get('doc_id')}"
            key = " ".join((item.get("chunk_text") or "").lower().split())
            if key in seen_chunks:
                reasons["duplicate_chunk"] = reasons.get("duplicate_chunk", 0) + 1
                continue
            if per_standard.get(standard, 0) >= 3:
                reasons["source_diversity_limit"] = reasons.get(
                    "source_diversity_limit", 0) + 1
                continue
            seen_chunks.add(key)
            per_standard[standard] = per_standard.get(standard, 0) + 1
            selected.append(item)
        if len(selected) > top_k:
            reasons["top_k_limit"] = reasons.get("top_k_limit", 0) + len(selected) - top_k
            selected = selected[:top_k]
        for item in selected:
            if (item.get("doc_type") or "").lower() == "gazette" and not item.get("exact_match"):
                # A gazette notification lists many standards; it is filed under
                # one of them in the source manifest. Unless the user asked for
                # that standard, do not present the schedule as its text: the
                # passage itself names the standards it covers.
                item["related_standard"] = item.get("standard_number", "")
                item["standard_number"] = ""
                item["title"] = "BIS Gazette notification (standards established, revised or withdrawn)"
        _finish_diagnostics(diagnostics, candidate_count=candidate_count,
                            selected=selected, reasons=reasons)
        return selected
    finally:
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass
