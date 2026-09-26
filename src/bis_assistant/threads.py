"""Turn-context mechanics: the single owner of thread history/rounds/force.

Previously this shape (``{"history", "rounds", "force"}``) was rebuilt inline in
``assistant.answer`` and re-implemented with different truncation in
``server.chat`` (bridge ``[-4:]`` vs stored ``[-6:]``), while ``cli`` carried a
third variant. This module owns the mechanics — normalize, reset, force, push,
combine — behind one interface. Redaction, persistence, tokens, and expiry stay
with their existing owners; only the shape arithmetic lives here.
"""
from __future__ import annotations

HISTORY_LIMIT = 8  # server-kept redacted entries per thread (4 user/assistant pairs)
BRIDGE_LIMIT = 4  # legacy client-held context turns accepted per call (unchanged)


def new_context(force: bool = False) -> dict:
    return {"history": [], "rounds": 0, "force": bool(force)}


def normalize_context(context: dict | None) -> dict:
    ctx = context or {}
    return {"history": list(ctx.get("history", [])),
            "rounds": int(ctx.get("rounds", 0)),
            "force": bool(ctx.get("force", False))}


def reset_context(force: bool = False) -> dict:
    return new_context(force=force)


def with_force(ctx: dict) -> dict:
    return {"history": list(ctx.get("history", [])),
            "rounds": int(ctx.get("rounds", 0)), "force": True}


def push_history(history: list, entry: str, limit: int = HISTORY_LIMIT) -> list:
    """Bounded append for persisted (redacted) thread history."""
    return (list(history) + [entry])[-limit:]


def push_turn(history: list, user_text: str, assistant_text: str = "",
              limit: int = HISTORY_LIMIT) -> list:
    """Append one exchange as "User: ..." / "Assistant: ..." entries.

    The model needs its own previous answer to resolve follow-ups such as
    "which labs test it?". An empty assistant text (errors, refusals) is
    skipped.
    """
    out = list(history) + [f"User: {user_text}"]
    if assistant_text:
        out.append(f"Assistant: {assistant_text}")
    return out[-limit:]


def append_turn(history: list, entry: str) -> list:
    """Unbounded append for the in-memory dialog turn (persistence truncates)."""
    return list(history) + [entry]


def bridge_history(items: list, limit: int = BRIDGE_LIMIT) -> list:
    return list(items)[-limit:]


def next_rounds(rounds: int) -> int:
    return int(rounds) + 1


def rounds_from(ctx: dict | None, default: int = 0) -> int:
    return int((ctx or {}).get("rounds", default))


def combined_query(history: list, query: str) -> str:
    return " ".join(list(history) + [query]).strip()
