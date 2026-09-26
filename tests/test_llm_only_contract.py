"""The chat path must not answer without model-generated text."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _configured_llm() -> dict:
    return {
        "provider": "openai-compatible",
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://llm.example.test/v1",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout_s": 1.0,
        "retries": 0,
    }


def test_valid_chat_turn_is_generated_by_llm_even_without_lab_hits(monkeypatch):
    from bis_assistant import assistant, rag_llm

    cfg = _configured_llm()
    seen: list[list[dict]] = []

    monkeypatch.setattr(assistant, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(assistant, "_rag_lookup",
                        lambda *_args, **_kwargs: ([], {"enabled": True}))
    monkeypatch.setattr(rag_llm, "chat_complete",
                        lambda messages, _cfg=None: seen.append(messages) or "MODEL ANSWER")

    response = assistant.answer("What is your name?", "en")

    assert response["text"].startswith("MODEL ANSWER")
    assert response["rag_used_llm"] is True
    assert len(seen) == 1
    system = next(m["content"] for m in seen[0] if m["role"] == "system")
    user = next(m["content"] for m in seen[0] if m["role"] == "user")
    assert "use only the bis evidence" in " ".join(system.lower().split())
    assert "what is your name?" in user.lower()
    assert re.search(r"\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}", user)


def test_unconfigured_model_returns_offline_message_not_clarification(monkeypatch):
    from bis_assistant import assistant

    cfg = _configured_llm() | {"model": "", "api_key": ""}
    monkeypatch.setattr(assistant, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(assistant, "_rag_lookup",
                        lambda *_args, **_kwargs: ([], {"enabled": False}))

    response = assistant.answer("steel bottle", "en")

    assert response["kind"] == "model_unavailable"
    assert response["rag_used_llm"] is False
    assert "unavailable" in response["text"].lower() or "offline" in response["text"].lower()
    assert response["questions"] == []
    assert response["citations"] == []


def test_lab_passages_are_prompt_context_never_the_answer(monkeypatch):
    from bis_assistant import assistant, rag_llm

    cfg = _configured_llm()
    evidence = [{
        "standard_number": "IS 14478:2026",
        "title": "Plain bearings",
        "doc_type": "lab",
        "chunk_text": "RAW LAB PASSAGE THAT MUST NOT BE SHOWN AS THE ANSWER",
        "source_url": "https://www.bis.gov.in/know-your-standard",
    }]
    seen: list[list[dict]] = []
    monkeypatch.setattr(assistant, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(assistant, "_rag_lookup",
                        lambda *_args, **_kwargs: (evidence, {"enabled": True}))
    monkeypatch.setattr(rag_llm, "chat_complete",
                        lambda messages, _cfg=None: seen.append(messages)
                        or "SYNTHESIZED MODEL ANSWER [Source 1]")

    response = assistant.answer("What does IS 14478 cover?", "en")

    user = next(m["content"] for m in seen[0] if m["role"] == "user")
    assert response["text"] == "SYNTHESIZED MODEL ANSWER[1]"
    assert "RAW LAB PASSAGE" not in response["text"]
    assert "RAW LAB PASSAGE" in user
    assert response["sources"]
    assert response["rag_used_llm"] is True


def test_model_failure_does_not_return_raw_lab_passage(monkeypatch):
    from bis_assistant import assistant, rag_llm

    cfg = _configured_llm()
    evidence = [{
        "standard_number": "IS 14478:2026",
        "title": "Plain bearings",
        "doc_type": "lab",
        "chunk_text": "RAW LAB PASSAGE THAT MUST NOT BE SHOWN AS THE ANSWER",
        "source_url": "https://www.bis.gov.in/know-your-standard",
    }]
    monkeypatch.setattr(assistant, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(assistant, "_rag_lookup",
                        lambda *_args, **_kwargs: (evidence, {"enabled": True}))
    monkeypatch.setattr(rag_llm, "chat_complete", lambda *_args, **_kwargs: None)

    response = assistant.answer("What does this standard require?", "en")

    assert response["kind"] == "model_unavailable"
    assert response["rag_used_llm"] is False
    assert "RAW LAB PASSAGE" not in response["text"]
    assert "unavailable" in response["text"].lower() or "offline" in response["text"].lower()
