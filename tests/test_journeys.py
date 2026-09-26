"""Representative user questions all use the same model-backed chat path."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize("query", [
    "What is your name?",
    "What is the current date and time?",
    "नल के पानी का मानक कौन सा है?",
    "I manufacture 9W B22 self-ballasted LED bulbs. Which standard and is CRS needed?",
    "How do I verify HUID on gold jewellery I bought?",
    "Tell me about plain bearings in IS 14478.",
])
def test_each_question_is_sent_to_the_model(query, monkeypatch):
    from bis_assistant import assistant, rag_llm

    cfg = {"provider": "openai-compatible", "model": "test-model", "api_key": "k",
           "base_url": "https://llm.example.test/v1", "temperature": 0.0,
           "max_tokens": 256, "timeout_s": 1.0, "retries": 0}
    calls = []
    monkeypatch.setattr(assistant, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(assistant, "_rag_lookup", lambda *_a, **_kw: ([], {"enabled": True}))
    monkeypatch.setattr(rag_llm, "chat_complete",
                        lambda messages, _cfg=None: calls.append(messages) or "MODEL GENERATED ANSWER")

    response = assistant.answer(query)

    assert response["text"] == "MODEL GENERATED ANSWER"
    assert response["kind"] == "llm_answer"
    # Hindi questions first get a small English search-query rewrite call.
    answer_calls = [c for c in calls if "search query" not in c[0]["content"]]
    assert len(answer_calls) == 1
    assert query.lower() in answer_calls[0][1]["content"].lower()
