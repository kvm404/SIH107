"""LLM-generated chat titles. Standalone helper; never feeds chat answers.

``generate_title`` takes the opening exchange of a topic and returns a short
sidebar title, or None when no LLM is available (callers keep their local
heuristic title). Title text is sanitised: one line, no quotes, no em dashes.
"""
from __future__ import annotations

import re

MAX_TITLE_CHARS = 48

_SYSTEM = """\
You generate short chat titles. Rules for every title:
- 2 to 6 plain words naming the topic (product, standard, or question focus).
- No quotes, no trailing punctuation, no emojis, no em dashes.
- Never write "Question about", "Chat about", or "Conversation".
- Reply with the title only, nothing else.
- Use the same language as the conversation."""


def clean_title(raw: str | None) -> str:
    """Sanitise raw model output into a sidebar title ("" when unusable)."""
    if not raw:
        return ""
    line = raw.strip().splitlines()[0].strip()
    line = re.sub(r"^[\s\"'«»“”*#\-:]+", "", line)
    line = re.sub(r"[\s\"'«»“”*#:;,.!?]+$", "", line)
    line = line.replace("—", " ").replace("–", " ")
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > MAX_TITLE_CHARS:
        cut = line[:MAX_TITLE_CHARS]
        sp = cut.rfind(" ")
        line = (cut[:sp] if sp > 15 else cut).rstrip()
    if len(line) < 2 or line.lower().startswith(
            ("question about", "chat about", "conversation")):
        return ""
    return line


def generate_title(user_text: str, assistant_text: str = "",
                   lang: str = "en", cfg: dict | None = None) -> str | None:
    """Return an LLM title for an opening exchange. None when unavailable."""
    u = (user_text or "").strip()[:1000]
    if not u:
        return None
    try:
        from .rag_config import load_llm_config
        from .rag_llm import chat_complete, is_configured, utility_config
        # Reasoning models spend part of the budget before the title; a tight
        # cap truncated titles to fragments like "CR".
        cfg = utility_config(cfg or load_llm_config(), max_tokens=96)
        if not is_configured(cfg):
            return None
        convo = f"USER: {u}"
        a = (assistant_text or "").strip()[:1500]
        if a:
            convo += f"\nASSISTANT: {a}"
        if lang == "hi":
            convo += "\n(Conversation language: Hindi. Title in Hindi.)"
        out = chat_complete([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": convo + "\nTitle:"}], cfg)
        title = clean_title(out)
        return title or None
    except Exception:
        return None
