"""In-process metrics (plan §7). Single-instance pilot; Prometheus scrapes /metrics.

Rollback: BIS_METRICS_ENABLED=false disables collection (Phase-6 flag).
"""
from __future__ import annotations
import os
import threading
from datetime import date

_lock = threading.Lock()
_counts: dict[str, int] = {}
_lat: list[int] = []
_rag_lat: list[int] = []


def enabled() -> bool:
    return os.environ.get("BIS_METRICS_ENABLED", "true").lower() in ("1", "true", "yes")


def incr(name: str, n: int = 1) -> None:
    if not enabled():
        return
    with _lock:
        _counts[name] = _counts.get(name, 0) + n


def observe_latency_ms(ms: int) -> None:
    if not enabled():
        return
    with _lock:
        _lat.append(ms)
        del _lat[:-500]


def observe_retrieval_latency_ms(ms: int) -> None:
    if not enabled():
        return
    with _lock:
        _rag_lat.append(ms)
        del _rag_lat[:-500]


def reset() -> None:  # tests only
    with _lock:
        _counts.clear()
        _lat.clear()
        _rag_lat.clear()


def _pct(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def kb_staleness_days() -> int:
    """Days since the BIS knowledge pages were last fetched (-1 if unknown)."""
    try:
        from pathlib import Path
        from .knowledge import knowledge_files, parse_knowledge_file
        root = Path(__file__).resolve().parents[2] / "data" / "knowledge"
        dates = [parse_knowledge_file(p)["retrieved"] for p in knowledge_files(root)]
        dates = [d for d in dates if d]
        if not dates:
            return -1
        return max(0, (date.today() - date.fromisoformat(max(dates)[:10])).days)
    except Exception:
        return -1


SERIES = ["chat_total", "answered_total", "refused_total", "needs_info_total",
          "model_unavailable_total",
          "feedback_total", "feedback_neg_total", "chat_5xx_total",
          "rag_retrieval_total", "rag_retrieval_selected_total",
          "rag_retrieval_rejected_total", "rag_retrieval_outcome_answered_total",
          "rag_retrieval_outcome_refused_total",
          "rag_retrieval_outcome_needs_info_total",
          "rag_retrieval_outcome_model_unavailable_total",
          "rag_retrieval_result_type_document_chunk_total",
          "rag_retrieval_result_type_catalogue_record_total"]


def snapshot() -> dict:
    from . import verifier
    with _lock:
        lat = list(_lat)
        rag_lat = list(_rag_lat)
        counts = {k: _counts.get(k, 0) for k in SERIES}
        counts.update({k: v for k, v in _counts.items() if k not in counts})
    out = dict(counts)
    out["chat_latency_p50_ms"] = _pct(lat, 0.5)
    out["chat_latency_p95_ms"] = _pct(lat, 0.95)
    out["rag_retrieval_latency_p50_ms"] = _pct(rag_lat, 0.5)
    out["rag_retrieval_latency_p95_ms"] = _pct(rag_lat, 0.95)
    out["citation_fail_total"] = verifier.VIOLATION_COUNT["n"]
    out["kb_staleness_days"] = kb_staleness_days()
    return out


def render_prometheus() -> str:
    lines = []
    for k in sorted(snapshot()):
        lines.append(f"bis_{k} {snapshot()[k]}")
    return "\n".join(lines) + "\n"
