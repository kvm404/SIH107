"""Production API (plan §5, Phase 3): FastAPI, server-side threads, owner tokens,
request IDs, redacted JSON logs, consent/erasure/export endpoints.

Legacy client-held `context` is accepted as a one-turn migration bridge only:
the server immediately mints a thread_id that clients must use afterwards.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .assistant import answer
from . import threads as threadmod
from .config import load as load_config
from .i18n_privacy import find_pii, redact
from . import metrics as metrics_mod

CFG = load_config()
import os as _os
DB_PATH = Path(_os.environ.get("BIS_OPS_DB",
               str(Path(__file__).resolve().parents[2] / "kb" / "ops.db")))

OPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads(
  id TEXT PRIMARY KEY, user_ref TEXT DEFAULT '', history_redacted_json TEXT DEFAULT '[]',
  rounds INTEGER DEFAULT 0, lang TEXT DEFAULT 'en',
  created_at TEXT, updated_at TEXT, expires_at TEXT, owner_token_hash TEXT);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, role TEXT,
  text_redacted TEXT, citations_json TEXT DEFAULT '[]', kind TEXT DEFAULT '',
  ms INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS consents(
  user_ref TEXT PRIMARY KEY, purpose TEXT, granted_at TEXT, expires_at TEXT, revoked_at TEXT);
CREATE TABLE IF NOT EXISTS profiles(
  user_ref TEXT PRIMARY KEY, fields_json TEXT DEFAULT '{}', updated_at TEXT);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, message_id INTEGER,
  rating INTEGER, note_redacted TEXT DEFAULT '', user_ref TEXT DEFAULT '',
  status TEXT DEFAULT 'pending', created_at TEXT);
CREATE TABLE IF NOT EXISTS kb_reviews(
  diff_id INTEGER PRIMARY KEY, publisher TEXT, approver TEXT, decided_at TEXT,
  CHECK (publisher <> approver));
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, target_ref TEXT, at TEXT);
"""

# ---- logging (redacted JSON, request_id) ----
class _JsonFmt(logging.Formatter):
    def format(self, record):
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "level": record.levelname, "msg": record.getMessage(),
                   **getattr(record, "ctx", {})}
        return json.dumps(payload, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFmt())
log = logging.getLogger("bis.api")
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False

_RETRIEVAL_BRANCHES = {
    "catalogue", "catalogue_fts", "catalogue_only", "document", "document_fts",
    "document_only", "fts", "fts_catalogue", "fts_document", "hybrid", "lexical",
    "semantic", "none", "no_results", "unavailable", "disabled",
}
_EVIDENCE_TYPES = {"document_chunk", "catalogue_record"}
_REJECTION_REASONS = {
    "below_min_score", "below_threshold", "candidate_cap", "duplicate",
    "below_relevance_threshold", "catalogue_limit", "combined_top_k_limit",
    "duplicate_chunk", "duplicate_evidence", "duplicate_source",
    "insufficient_query_term_overlap", "low_relevance", "low_score",
    "missing_query_terms", "no_lexical_match", "no_relevance_overlap",
    "not_selected", "outside_top_k", "query_mismatch", "relevance_threshold",
    "retrieval_disabled", "retrieval_error", "source_diversity",
    "source_diversity_limit", "source_limit", "threshold", "top_k_limit",
}
_SOURCE_ID_RE = re.compile(
    r"^(?:IS|ISO|IEC)\s*[A-Z0-9][A-Z0-9().:/_-]{0,50}"
    r"(?:\s+\(Part\s+\d+\))?(?::\d{4})?$", re.I)


def _bounded_int(value, *, minimum: int = 0, maximum: int = 1_000_000) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(minimum, min(maximum, number))


def _safe_identifier(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())[:64]
    return value if _SOURCE_ID_RE.fullmatch(value) else None


def _safe_retrieval_diagnostics(raw) -> dict | None:
    """Copy only bounded, non-content retrieval metadata into logs and responses."""
    if not isinstance(raw, dict):
        return None

    raw_branch = raw.get("branch", raw.get("retrieval_branch"))
    branch = raw_branch.lower() if isinstance(raw_branch, str) else ""
    if branch not in _RETRIEVAL_BRANCHES:
        branch = "other"

    raw_results = raw.get("results", raw.get("result_details", raw.get("candidates", [])))
    results = []
    if isinstance(raw_results, list):
        for item in raw_results[:20]:
            if not isinstance(item, dict):
                continue
            evidence_type = item.get("evidence_type", item.get("type"))
            if not isinstance(evidence_type, str) or evidence_type not in _EVIDENCE_TYPES:
                evidence_type = "other"
            result = {"evidence_type": evidence_type}

            rank = _bounded_int(item.get("rank"), minimum=1, maximum=1000)
            if rank is not None:
                result["rank"] = rank
            score = item.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                try:
                    score = float(score)
                    if math.isfinite(score):
                        result["score"] = round(max(-1_000_000.0, min(1_000_000.0, score)), 6)
                except (TypeError, ValueError, OverflowError):
                    pass
            source_id = _safe_identifier(item.get(
                "source_id", item.get("source_identifier", item.get("standard_number"))))
            if source_id:
                result["source_id"] = source_id
            if isinstance(item.get("selected"), bool):
                result["selected"] = item["selected"]
            reason = item.get("rejection_reason", item.get("rejection_reason_code"))
            if isinstance(reason, str) and reason:
                reason = reason.lower()
                result["rejection_reason"] = (
                    reason if reason in _REJECTION_REASONS else "other")
            results.append(result)

    selected_from_results = sum(r.get("selected") is True for r in results)
    rejected_from_results = sum(r.get("selected") is False for r in results)
    selected = _bounded_int(raw.get("selected_count"))
    rejected = _bounded_int(raw.get("rejected_count"))
    if selected is None:
        selected = selected_from_results
    if rejected is None:
        rejected = rejected_from_results

    reasons = {}
    raw_reasons = raw.get("rejection_reasons")
    if isinstance(raw_reasons, dict):
        for reason, count in list(raw_reasons.items())[:20]:
            if not isinstance(reason, str):
                continue
            reason = reason.lower()
            code = reason if reason in _REJECTION_REASONS else "other"
            amount = _bounded_int(count)
            if amount:
                reasons[code] = min(1_000_000, reasons.get(code, 0) + amount)
    if not reasons:
        for item in results:
            reason = item.get("rejection_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1

    retrieval_ms = None
    for key in ("retrieval_ms", "elapsed_ms", "latency_ms"):
        retrieval_ms = _bounded_int(raw.get(key), maximum=3_600_000)
        if retrieval_ms is not None:
            break

    return {
        "branch": branch,
        "selected_count": selected,
        "rejected_count": rejected,
        "rejection_reasons": reasons,
        "results": results,
        **({"retrieval_ms": retrieval_ms} if retrieval_ms is not None else {}),
    }


def _record_retrieval_telemetry(diagnostics: dict, response: dict, latency_ms: int) -> None:
    branch = diagnostics["branch"]
    metrics_mod.incr("rag_retrieval_total")
    metrics_mod.incr(f"rag_retrieval_branch_{branch}_total")
    metrics_mod.incr("rag_retrieval_selected_total", diagnostics["selected_count"])
    metrics_mod.incr("rag_retrieval_rejected_total", diagnostics["rejected_count"])
    outcome = "model_unavailable" if response.get("kind") in (
        "model_unavailable", "model_busy") else (
        "refused" if response.get("refused") else
        "needs_info" if response.get("needs_info") else "answered")
    metrics_mod.incr(f"rag_retrieval_outcome_{outcome}_total")
    for reason, count in diagnostics["rejection_reasons"].items():
        metrics_mod.incr(f"rag_retrieval_rejected_reason_{reason}_total", count)
    for result in diagnostics["results"]:
        metrics_mod.incr(f"rag_retrieval_result_type_{result['evidence_type']}_total")
        log.info("RAG retrieval result", extra={"ctx": {
            "event": "rag_retrieval_result", "branch": branch,
            "rank": result.get("rank"), "evidence_type": result["evidence_type"],
            "score": result.get("score"), "source_id": result.get("source_id"),
            "selected": result.get("selected"),
            "rejection_reason": result.get("rejection_reason")}})
    if "retrieval_ms" in diagnostics:
        metrics_mod.observe_retrieval_latency_ms(diagnostics["retrieval_ms"])
    log.info("RAG retrieval completed", extra={"ctx": {
        "event": "rag_retrieval", "branch": branch,
        "selected_count": diagnostics["selected_count"],
        "rejected_count": diagnostics["rejected_count"],
        "rejection_reasons": diagnostics["rejection_reasons"],
        "retrieval_ms": diagnostics.get("retrieval_ms"),
        "outcome": outcome, "latency_ms": latency_ms}})

def _admin_hashes() -> set[str]:
    import os
    return {_key_hash(k.strip()) for k in os.environ.get("BIS_ADMIN_API_KEYS", "").split(",") if k.strip()}


def _require_admin(key: Optional[str]) -> str:
    hashes = _admin_hashes()
    got = _key_hash(key) if key else ""
    if not key or not hashes or not any(hmac.compare_digest(got, h) for h in hashes):
        raise HTTPException(status_code=403, detail={
            "error": "admin key required", "code": "forbidden", "retryable": False})
    return got[:16]


def _audit(conn: sqlite3.Connection, actor: str, action: str, target: str) -> None:
    conn.execute("INSERT INTO audit_log(actor, action, target_ref, at) VALUES (?,?,?,?)",
                 (actor, action, target, _utcnow().isoformat()))
    conn.commit()


def _key_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(OPS_SCHEMA)
    return conn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---- models ----
class ChatIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    lang: Optional[str] = None
    thread_id: Optional[str] = None
    context: Optional[dict] = None  # deprecated bridge; thread_id authoritative.
    # One-turn migration only: accepted, truncated to last 4 turns, then the
    # server mints a thread_id clients must use afterwards. New code must
    # pass thread_id (see chat.py ThreadHandle) — context will be removed.
    force: bool = False
    new_topic: bool = False  # Design 3: ignore thread_id/context, mint fresh thread


class ThreadOut(BaseModel):
    thread_id: str
    owner_token: str
    expires_at: str


class TranscribeIn(BaseModel):
    audio_b64: str = Field(min_length=8, max_length=2_800_000)
    mime: str = "audio/webm"


class TitleIn(BaseModel):
    user_text: str = Field(min_length=1, max_length=2000)
    assistant_text: str = Field(default="", max_length=8000)
    lang: Optional[str] = None


class ConsentIn(BaseModel):
    user_ref: str = Field(min_length=1, max_length=128)
    purpose: str = "personalise BIS licensing guidance"


app = FastAPI(title="BIS Assistant API", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=CFG["api"]["cors_allow_origins"],
                   allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "X-Owner-Token", "X-User-Ref",
                                  "X-Request-ID", "X-Admin-Key"])


@app.middleware("http")
async def _rid(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or secrets.token_hex(8)
    t0 = time.time()
    try:
        resp = await call_next(request)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        resp = JSONResponse({**detail, "request_id": rid}, status_code=e.status_code)
    ms = int((time.time() - t0) * 1000)
    resp.headers["X-Request-ID"] = rid
    if resp.status_code >= 500 and request.url.path == "/chat":
        metrics_mod.incr("chat_5xx_total")
    log.info(f"{request.method} {request.url.path} -> {resp.status_code} {ms}ms",
             extra={"ctx": {"request_id": rid, "status": resp.status_code, "ms": ms}})
    return resp


def _get_thread(conn: sqlite3.Connection, tid: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=410, detail={
            "error": "unknown thread; start a new topic", "code": "thread_gone",
            "retryable": False})
    if row["expires_at"] and row["expires_at"] < _utcnow().isoformat():
        conn.execute("DELETE FROM messages WHERE thread_id=?", (tid,))
        conn.execute("DELETE FROM threads WHERE id=?", (tid,))
        conn.commit()
        raise HTTPException(status_code=410, detail={
            "error": "thread expired; start a new topic", "code": "thread_expired",
            "retryable": False})
    return row


def _check_owner(row: sqlite3.Row, token: Optional[str]) -> None:
    stored = row["owner_token_hash"] or ""
    got = _key_hash(token) if token else ""
    if not token or not stored or not hmac.compare_digest(got, stored):
        raise HTTPException(status_code=403, detail={
            "error": "owner token required", "code": "forbidden", "retryable": False})


@app.get("/health")
def health():
    from .rag_llm import is_configured, load_llm_config
    cfg = load_llm_config()
    speech = is_configured(cfg) or bool(os.environ.get("GROQ_API_KEY", "").strip())
    return {"ok": True, "speech": speech}


@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(metrics_mod.render_prometheus())


@app.post("/threads", response_model=ThreadOut)
def new_thread(x_user_ref: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        tid, token = secrets.token_hex(8), secrets.token_urlsafe(32)
        exp = (_utcnow() + timedelta(days=CFG["privacy"]["thread_ttl_days"])).isoformat()
        conn.execute("INSERT INTO threads(id, user_ref, created_at, updated_at, expires_at,"
                     " owner_token_hash) VALUES (?,?,?,?,?,?)",
                     (tid, x_user_ref or "", _utcnow().isoformat(), _utcnow().isoformat(),
                      exp, _key_hash(token)))
        conn.commit()
        return {"thread_id": tid, "owner_token": token, "expires_at": exp}
    finally:
        conn.close()


@app.get("/threads/{tid}")
def read_thread(tid: str, x_owner_token: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        row = _get_thread(conn, tid)
        _check_owner(row, x_owner_token)
        msgs = conn.execute("SELECT role, text_redacted, citations_json, kind, ms, created_at"
                            " FROM messages WHERE thread_id=? ORDER BY id", (tid,)).fetchall()
        return {"thread_id": tid, "rounds": row["rounds"], "lang": row["lang"],
                "messages": [dict(m) for m in msgs]}
    finally:
        conn.close()


@app.delete("/threads/{tid}")
def delete_thread(tid: str, x_owner_token: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        _check_owner(_get_thread(conn, tid), x_owner_token)
        conn.execute("DELETE FROM messages WHERE thread_id=?", (tid,))
        conn.execute("DELETE FROM threads WHERE id=?", (tid,))
        conn.commit()
        return {"deleted": tid}
    finally:
        conn.close()


@app.post("/chat")
def chat(body: ChatIn, request: Request,
         x_owner_token: Optional[str] = Header(default=None),
         x_user_ref: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        history, rounds, tid = [], 0, body.thread_id
        if body.new_topic:
            # Explicit fresh topic: drop any carried state, mint below.
            history, rounds, tid = [], 0, None
        elif tid:
            row = _get_thread(conn, tid)
            if row["owner_token_hash"]:
                _check_owner(row, x_owner_token)
            ctx = threadmod.normalize_context({
                "history": json.loads(row["history_redacted_json"] or "[]"),
                "rounds": row["rounds"]})
            history, rounds = ctx["history"], ctx["rounds"]
        elif isinstance(body.context, dict) and body.context.get("history"):
            log.warning("legacy context bridge used; minting thread_id — "
                        "clients must switch to thread_id",
                        extra={"ctx": {"request_id": request.headers.get("X-Request-ID", "")}})
            ctx = threadmod.normalize_context({
                "history": threadmod.bridge_history(
                    [redact(h)[:2000] for h in body.context["history"]]),
                "rounds": body.context.get("rounds", 0)})
            history, rounds = ctx["history"], ctx["rounds"]
        q = body.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail={
                "error": "empty query", "code": "bad_request", "retryable": False})
        if _wants_erasure(q):
            if not tid:
                raise HTTPException(status_code=400, detail={
                    "error": "owner token + thread required to erase",
                    "code": "bad_request", "retryable": False})
            row = conn.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
            if row is None:
                raise HTTPException(status_code=410, detail={
                    "error": "unknown thread; start a new topic", "code": "thread_gone",
                    "retryable": False})
            _check_owner(row, x_owner_token)
            out = _erase_thread(conn, tid)
            _audit(conn, "user:" + _key_hash(x_owner_token or tid)[:16], "erasure", "chat")
            return {"text": "Stored messages for this thread were erased.",
                    "refused": False, "kind": "erasure", "lang": "en",
                    "citations": [], "pii": find_pii(q), "needs_info": False,
                    "questions": [], "known": [], "assumptions": [],
                    "context": {"history": [], "rounds": 0},
                    "erased_threads": out["erased_threads"],
                    "thread_id": tid}
        lang = body.lang if body.lang in ("en", "hi") else None
        t0 = time.time()
        resp = answer(q, lang, {"history": history, "rounds": rounds, "force": body.force})
        ms = int((time.time() - t0) * 1000)
        retrieval_diagnostics = _safe_retrieval_diagnostics(
            resp.get("retrieval_diagnostics"))
        if retrieval_diagnostics is None:
            resp.pop("retrieval_diagnostics", None)
        else:
            # Never pass through arbitrary diagnostic keys or retrieved content.
            resp["retrieval_diagnostics"] = retrieval_diagnostics
        new_history = threadmod.normalize_context(resp.get("context"))["history"] \
            or threadmod.push_history(history, redact(q)[:2000])
        new_rounds = threadmod.rounds_from(resp.get("context"), default=rounds)
        if tid is None:  # mint server thread (bridge + fresh turns)
            tid = secrets.token_hex(8)
            token = secrets.token_urlsafe(32)
            exp = (_utcnow() + timedelta(days=CFG["privacy"]["thread_ttl_days"])).isoformat()
            conn.execute("INSERT INTO threads(id, user_ref, history_redacted_json, rounds,"
                         " lang, created_at, updated_at, expires_at, owner_token_hash)"
                         " VALUES (?,?,?,?,?,?,?,?,?)",
                         (tid, x_user_ref or "", json.dumps(new_history), new_rounds,
                          resp.get("lang", "en"), _utcnow().isoformat(),
                          _utcnow().isoformat(), exp, _key_hash(token)))
            resp["owner_token"] = token
            resp["expires_at"] = exp
        else:
            conn.execute("UPDATE threads SET history_redacted_json=?, rounds=?, updated_at=?"
                         " WHERE id=?", (json.dumps(new_history), new_rounds,
                                         _utcnow().isoformat(), tid))
        conn.execute("INSERT INTO messages(thread_id, role, text_redacted, citations_json,"
                     " kind, ms, created_at) VALUES (?,?,?,?,?,?,?)",
                     (tid, "user", redact(q)[:2000], "[]", "user", 0,
                      _utcnow().isoformat()))
        conn.execute("INSERT INTO messages(thread_id, role, text_redacted, citations_json,"
                     " kind, ms, created_at) VALUES (?,?,?,?,?,?,?)",
                     (tid, "assistant", redact(resp["text"])[:8000],
                      json.dumps(resp.get("citations", [])), resp.get("kind", ""),
                      ms, _utcnow().isoformat()))
        conn.commit()
        resp["thread_id"] = tid
        if retrieval_diagnostics is not None:
            _record_retrieval_telemetry(retrieval_diagnostics, resp, ms)
        log.info("chat response completed", extra={"ctx": {
            "kind": resp.get("kind") if resp.get("kind") in {
                "llm_answer", "model_unavailable", "model_busy", "grounding_refusal",
                "erasure", "refused"} else "other",
            "lang": resp.get("lang") if resp.get("lang") in {"en", "hi"} else "other",
            "needs_info": bool(resp.get("needs_info")), "ms": ms,
            "rag_mode": resp.get("rag_mode") if resp.get("rag_mode") in {
                "llm", "model unavailable", "model busy", "llm_grounding_guard",
                "rag", "none"} else "other",
            "rewritten": bool(resp.get("search_query")),
            "rag_used_llm": bool(resp.get("rag_used_llm", False)),
            "source_count": len(resp.get("sources") or resp.get("rag_evidence") or []),
            "pii": [k for k, v in find_pii(q).items() if v]}})
        metrics_mod.incr("chat_total")
        metrics_mod.observe_latency_ms(ms)
        if resp.get("kind") in ("model_unavailable", "model_busy"):
            metrics_mod.incr("model_unavailable_total")
            if resp.get("kind") == "model_busy":
                metrics_mod.incr("model_busy_total")
        elif resp.get("refused"):
            metrics_mod.incr("refused_total")
        else:
            metrics_mod.incr("answered_total")
        if resp.get("needs_info"):
            metrics_mod.incr("needs_info_total")
        return resp
    finally:
        conn.close()


@app.post("/transcribe")
def transcribe_clip(body: TranscribeIn):
    """Speech-to-text for the composer mic. 503 when no model can decode."""
    import base64 as _b64
    from .rag_llm import transcribe_audio
    mime = (body.mime or "audio/webm").split(";")[0].strip() or "audio/webm"
    try:
        data = _b64.b64decode(body.audio_b64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail={
            "error": "invalid audio", "code": "bad_request", "retryable": False})
    if not data or len(data) > 2_000_000:
        raise HTTPException(status_code=400, detail={
            "error": "audio too large or empty", "code": "bad_request",
            "retryable": False})
    try:
        text = transcribe_audio(data, mime)
    except Exception:
        log.exception("transcription failed")
        text = None
    if not text:
        raise HTTPException(status_code=503, detail={
            "error": "speech service unavailable", "code": "model_unavailable",
            "retryable": True})
    return {"text": text}


@app.post("/title")
def chat_title(body: TitleIn):
    """LLM sidebar title for an opening exchange. 503 when unavailable.

    Stateless and auth-free: takes the opening user/assistant texts only.
    Clients keep their local heuristic title on any failure.
    """
    from . import titles as titlemod
    lang = body.lang if body.lang in ("en", "hi") else "en"
    title = titlemod.generate_title(body.user_text, body.assistant_text, lang)
    if not title:
        raise HTTPException(status_code=503, detail={
            "error": "title model unavailable", "code": "model_unavailable",
            "retryable": True})
    log.info("title generated", extra={"ctx": {"title": title}})
    return {"title": title}


@app.post("/consent")
def consent(body: ConsentIn):
    conn = _db()
    try:
        granted = _utcnow()
        exp = (granted + timedelta(days=CFG["privacy"]["consent_valid_days"])).isoformat()
        conn.execute("INSERT INTO consents(user_ref, purpose, granted_at, expires_at, revoked_at)"
                     " VALUES (?,?,?,?,?) ON CONFLICT(user_ref) DO UPDATE SET purpose=excluded.purpose,"
                     " granted_at=excluded.granted_at, expires_at=excluded.expires_at, revoked_at=NULL",
                     (body.user_ref, body.purpose, granted.isoformat(), exp, None))
        conn.commit()
        return {"user_ref": body.user_ref, "granted_at": granted.isoformat(),
                "expires_at": exp, "purpose": body.purpose}
    finally:
        conn.close()


# Chat-side erase: real intent only — not substring "mera data" (data sheet, etc.).
_ERASE_INTENT = re.compile(
    r"(?:delete|erase|remove|wipe)\s+my\s+data|"
    r"mera\s+data\s+(?:mitao|mita\s*do|hatao|hatayen|hata\s*do|delete|erase)|"
    r"(?:mitao|hatao|hatayen)\s+mera\s+data",
    re.I,
)


def _wants_erasure(query: str) -> bool:
    ql = re.sub(r"\s+", " ", (query or "").strip().lower())
    return bool(_ERASE_INTENT.search(ql))


def _erase_thread(conn: sqlite3.Connection, tid: str) -> dict:
    """Capability is the owner token of one thread — never fan out on user_ref."""
    conn.execute("DELETE FROM messages WHERE thread_id=?", (tid,))
    conn.execute("DELETE FROM feedback WHERE thread_id=?", (tid,))
    conn.execute("DELETE FROM threads WHERE id=?", (tid,))
    conn.commit()
    return {"erased_threads": 1}


def _principal_from_owner(conn: sqlite3.Connection, token: Optional[str],
                          x_user_ref: Optional[str]) -> tuple[str, sqlite3.Row]:
    """Owner token is the capability for that thread only; X-User-Ref must match if sent."""
    if not token:
        raise HTTPException(status_code=403, detail={
            "error": "owner token required", "code": "forbidden", "retryable": False})
    row = conn.execute("SELECT * FROM threads WHERE owner_token_hash=?",
                       (_key_hash(token),)).fetchone()
    if row is None:
        raise HTTPException(status_code=403, detail={
            "error": "owner token required", "code": "forbidden", "retryable": False})
    principal = row["user_ref"] or ""
    if x_user_ref and x_user_ref != principal:
        raise HTTPException(status_code=403, detail={
            "error": "owner token does not match user", "code": "forbidden",
            "retryable": False})
    return principal, row


@app.delete("/me")
def erase_me(x_user_ref: Optional[str] = Header(default=None),
             x_owner_token: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        principal, row = _principal_from_owner(conn, x_owner_token, x_user_ref)
        out = _erase_thread(conn, row["id"])
        actor = principal or row["id"]
        _audit(conn, "user:" + _key_hash(actor)[:16], "erasure", "self")
        log.info("erasure", extra={"ctx": {"erased_threads": out["erased_threads"]}})
        return {"user_ref": principal, **out, "sla_hours": CFG["privacy"]["erasure_sla_hours"]}
    finally:
        conn.close()


@app.get("/me/export")
def export_me(x_user_ref: Optional[str] = Header(default=None),
              x_owner_token: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        principal, row = _principal_from_owner(conn, x_owner_token, x_user_ref)
        threads = [dict(row)]
        cons = []
        for t in threads:
            t.pop("owner_token_hash", None)
            t["messages"] = [dict(m) for m in conn.execute(
                "SELECT role, text_redacted, citations_json, kind, ms, created_at"
                " FROM messages WHERE thread_id=? ORDER BY id", (t["id"],)).fetchall()]
        return {"user_ref": principal, "threads": threads, "consents": cons}
    finally:
        conn.close()


class FeedbackIn(BaseModel):
    thread_id: str
    rating: int = Field(ge=-1, le=1)
    note: str = Field(default="", max_length=1000)


@app.post("/feedback")
def feedback(body: FeedbackIn, x_owner_token: Optional[str] = Header(default=None)):
    conn = _db()
    try:
        row = _get_thread(conn, body.thread_id)
        if row["owner_token_hash"]:
            _check_owner(row, x_owner_token)
        msg = conn.execute("SELECT id FROM messages WHERE thread_id=? AND role='assistant'"
                           " ORDER BY id DESC LIMIT 1", (body.thread_id,)).fetchone()
        conn.execute("INSERT INTO feedback(thread_id, message_id, rating, note_redacted,"
                     " user_ref, status, created_at) VALUES (?,?,?,?,?,?,?)",
                     (body.thread_id, msg["id"] if msg else None, body.rating,
                      redact(body.note)[:1000], row["user_ref"], "pending",
                      _utcnow().isoformat()))
        conn.commit()
        metrics_mod.incr("feedback_total")
        if body.rating < 0:
            metrics_mod.incr("feedback_neg_total")
        return {"ok": True, "status": "pending"}
    finally:
        conn.close()


@app.get("/kb/diff")
def kb_diff(x_admin_key: Optional[str] = Header(default=None)):
    import os
    admin = _require_admin(x_admin_key)
    kb = os.environ.get("BIS_KB_PATH", str(Path(__file__).resolve().parents[2] / "kb" / "bis.db"))
    from . import kb_store
    conn = kb_store.connect(kb)
    try:
        rows = conn.execute("SELECT * FROM pending_diffs WHERE status='pending' ORDER BY id").fetchall()
        return {"pending": [dict(r) for r in rows], "reviewed_by": admin}
    finally:
        conn.close()


class PublishIn(BaseModel):
    diff_id: int
    approve: bool
    publisher_key: str
    approver_key: str


@app.post("/kb/publish")
def kb_publish(body: PublishIn):
    import os
    pub, appr = _require_admin(body.publisher_key), _require_admin(body.approver_key)
    if pub == appr:
        raise HTTPException(status_code=400, detail={
            "error": "publisher and approver must be distinct", "code": "bad_request",
            "retryable": False})
    kb = os.environ.get("BIS_KB_PATH", str(Path(__file__).resolve().parents[2] / "kb" / "bis.db"))
    from . import kb_store
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from ingest import review as reviewmod
    kconn = kb_store.connect(kb)
    try:
        if body.approve:
            reviewmod.cmd_approve(kconn, body.diff_id, by=f"admin:{pub}")
        else:
            reviewmod.cmd_reject(kconn, body.diff_id)
    except reviewmod.ReviewError as e:
        code = {404: "not_found", 400: "bad_request"}.get(e.http_status, "conflict")
        raise HTTPException(status_code=e.http_status, detail={
            "error": str(e), "code": code, "retryable": False}) from e
    finally:
        kconn.close()
    conn = _db()
    try:
        conn.execute("INSERT INTO kb_reviews(diff_id, publisher, approver, decided_at)"
                     " VALUES (?,?,?,?)", (body.diff_id, pub, appr, _utcnow().isoformat()))
        _audit(conn, f"admin:{pub}", f"kb_publish:{body.diff_id}:{body.approve}", str(body.diff_id))
        conn.commit()
        return {"ok": True, "diff_id": body.diff_id, "approved": body.approve}
    finally:
        conn.close()
