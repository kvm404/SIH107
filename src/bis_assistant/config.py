"""Single config source: config.yaml + BIS_* env overrides. No secrets here."""
from __future__ import annotations
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # stdlib-only fallback (MVP core)
    yaml = None

_DEFAULTS = {
    "retrieval": {"direct_score": 15.0, "direct_margin": 5.0, "clarify_floor": 6.0,
                  "weak_floor": 3.0, "max_rounds": 2, "max_questions_per_turn": 2,
                  "scorer": "keyword", "kb_backend": "json",
                  "grounded_score": 10.0, "fusion_strong_score": 15.0},
    "privacy": {"thread_ttl_days": 7, "retention_days": 90,
                "consent_valid_days": 365, "erasure_sla_hours": 24},
    "api": {"port": 8000,
            "cors_allow_origins": ["http://127.0.0.1:5173", "http://localhost:5173"]},
    "ingest": {"crawl_delay_s": 2.0, "live_crawl_enabled": False,
               "refresh_days": 7, "staleness_alert_days": 14},
    "observability": {"trace_sample_rate": 0.05, "trace_retention_days": 14,
                      "latency_p95_ms": 2000},
    "rag": {"enabled": True, "db_path": "kb/bis_rag.db", "top_k": 5,
            "semantic": True, "embedding_model": "", "reranker_model": "",
            "weight_lexical": 1.0,
            "weight_semantic": 0.3, "exact_boost": 50.0},
    "llm": {"provider": "openai-compatible", "model": "", "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.2, "max_tokens": 900, "timeout_s": 20.0, "retries": 1,
            "fallback_model": "", "utility_model": ""},
}

_TYPES = {"port": int, "retention_days": int, "thread_ttl_days": int,
          "consent_valid_days": int, "erasure_sla_hours": int,
          "refresh_days": int, "staleness_alert_days": int, "trace_retention_days": int,
          "latency_p95_ms": int, "direct_score": float, "direct_margin": float,
          "clarify_floor": float, "weak_floor": float, "max_rounds": int, "max_questions_per_turn": int,
          "crawl_delay_s": float, "trace_sample_rate": float,
          "top_k": int, "max_tokens": int, "temperature": float, "timeout_s": float,
          "weight_lexical": float, "weight_semantic": float, "exact_boost": float,
          "enabled": bool, "semantic": bool, "live_crawl_enabled": bool,
          "retries": int, "grounded_score": float, "fusion_strong_score": float}


# File-content cache keyed by (path, mtime) (issue #4 P1-10): retrieval and
# provider configuration may both read the shared file during one chat turn.
# Only file content is cached — BIS_* env overrides are applied live on
# every load(), so monkeypatched env in tests keeps working.
_FILE_CACHE: dict = {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_file(p: Path) -> dict:
    if yaml is None:
        return {}
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    key = str(p.resolve())
    ent = _FILE_CACHE.get(key)
    if ent is not None and ent[0] == mtime:
        return ent[1]
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    _FILE_CACHE[key] = (mtime, data)
    return data


def load(path: str | Path | None = None) -> dict:
    """Load config.yaml (repo root default), apply BIS_* env overrides."""
    cfg = {s: dict(v) for s, v in _DEFAULTS.items()}
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "config.yaml"
    if p.exists():
        cfg = _deep_merge(cfg, _read_file(p))
    for section, keys in cfg.items():
        for key in keys:
            env = os.environ.get(f"BIS_{section.upper()}_{key.upper()}")
            if env is None:
                continue
            cast = _TYPES.get(key, str)
            if cast is bool:
                cfg[section][key] = env.lower() in ("1", "true", "yes")
            elif key in ("cors_allow_origins",):
                cfg[section][key] = [o.strip() for o in env.split(",")]
            else:
                try:
                    cfg[section][key] = cast(env)
                except ValueError:
                    pass
    return cfg
