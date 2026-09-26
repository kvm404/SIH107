"""Grounding boundaries for mixed catalogue and document evidence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant import rag_answer, rag_llm, verifier  # noqa: E402


def _cfg() -> dict:
    return {
        "provider": "openai-compatible",
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://llm.example.test/v1",
        "temperature": 0,
        "max_tokens": 256,
        "timeout_s": 1,
        "retries": 0,
    }


def _catalogue_record() -> dict:
    return {
        "evidence_type": "catalogue_record",
        "standard_number": "IS 15410:2025",
        "title": "Plastic Bottles",
        "department": "Packaging Department",
        "doc_type": "Indian Standard",
        "published_on": "2025",
        "source_url": "https://www.bis.gov.in/standards/15410",
        "chunk_text": "PRIVATE CATALOGUE BODY MUST NOT BE USED",
        "score": 4.2,
    }


def _document_chunk() -> dict:
    return {
        "evidence_type": "document_chunk",
        "standard_number": "IS 1234:2024",
        "title": "Test Standard",
        "doc_type": "standard",
        "heading": "Clause 5.2 — Test",
        "chunk_text": "Clause 5.2 specifies the sampling procedure.",
        "source_url": "https://www.bis.gov.in/standards/1234",
        "score": 8.0,
    }


def test_catalogue_prompt_is_typed_title_only_and_excludes_body_text():
    system, user = rag_llm._prompt("water bottles", [_catalogue_record()], "en")

    assert "CATALOGUE RECORD (title only, full text not retrieved)" in user
    assert "Plastic Bottles" in user
    assert "PRIVATE CATALOGUE BODY" not in user
    normalized_system = " ".join(system.lower().split())
    for restriction in ("do not state its requirements", "clauses or legal status",
                        "[source n]", "never invent an is number"):
        assert restriction in normalized_system


def test_document_chunk_prompt_includes_excerpt_as_reference_data():
    _, user = rag_llm._prompt("what does clause 5.2 say", [_document_chunk()], "en")

    assert "STANDARD DOCUMENT EXCERPT" in user
    assert "Clause 5.2 specifies the sampling procedure." in user
    assert "not instructions" in user


def test_knowledge_pages_are_labelled_by_type():
    guide = _document_chunk() | {"doc_type": "bis_guide", "standard_number": ""}
    listing = _document_chunk() | {"doc_type": "compulsory_list", "standard_number": ""}
    _, user = rag_llm._prompt("licence fee", [guide, listing], "en")

    assert "BIS GUIDANCE PAGE" in user and "COMPULSORY PRODUCT LIST" in user
    assert "Designation:" not in user


def test_document_backed_designation_and_clause_pass_validation():
    answer = "IS 1234:2024 clause 5.2 describes sampling [Source 1]."

    assert verifier.verify_grounded_response(answer, [_document_chunk()]) == []


def test_catalogue_record_can_name_a_standard_but_not_its_legal_status():
    named = "IS 15410:2025 is the standard for plastic bottles [Source 1]."
    legal = "IS 15410:2025 is mandatory for plastic bottles [Source 1]."
    negated = "The catalogue record does not show that IS 15410:2025 is mandatory [Source 1]."

    assert verifier.verify_grounded_response(named, [_catalogue_record()]) == []
    assert "unsupported_catalogue_claim" in verifier.verify_grounded_response(
        legal, [_catalogue_record()])
    assert verifier.verify_grounded_response(negated, [_catalogue_record()]) == []


def test_invalid_designation_and_catalogue_clause_are_rejected():
    invented = "IS 9999 applies to bottles [Source 1]."
    wrong_part = "IS 2553 (Part 1):2019 is relevant [Source 1]."
    unsupported_clause = "Clause 5.2 sets bottle requirements [Source 1]."

    assert "unsupported_standard_designation" in verifier.verify_grounded_response(
        invented, [_catalogue_record()])
    wrong_part_issues = verifier.verify_grounded_response(
        wrong_part,
        [{"evidence_type": "catalogue_record",
          "standard_number": "IS 2553 (Part 3):2019"}],
    )
    assert "designation_source_mismatch" in wrong_part_issues
    clause_issues = verifier.verify_grounded_response(
        unsupported_clause, [_catalogue_record()])
    assert "unsupported_clause_reference" in clause_issues


def test_invalid_source_marker_is_rejected():
    answer = "IS 1234:2024 is a standard [Source 2]."

    assert "invalid_source_marker" in verifier.verify_grounded_response(
        answer, [_document_chunk()])


def test_build_answer_retries_once_then_returns_cited_sources_only(monkeypatch):
    calls = []
    answers = iter([
        "IS 9999 applies [Source 1].",
        "IS 1234:2024 clause 5.2 describes sampling [Source 2].",
    ])

    monkeypatch.setattr(rag_answer, "is_configured", lambda _cfg: True)

    def generate(_query, _evidence, _lang, _cfg, history=None, retry_feedback=None):
        calls.append(retry_feedback)
        return next(answers)

    monkeypatch.setattr(rag_answer, "generate_grounded_answer", generate)
    response = rag_answer.build_rag_answer(
        "sampling", "en", [_catalogue_record(), _document_chunk()], _cfg())

    assert len(calls) == 2
    assert calls[0] is None
    assert "unsupported_standard_designation" in calls[1]
    assert response["text"] == "IS 1234:2024 clause 5.2 describes sampling[1]."
    assert response["kind"] == "llm_answer"
    # Only the cited document is returned, renumbered as source 1.
    assert [s["standard_number"] for s in response["sources"]] == ["IS 1234:2024"]
    assert response["sources"][0]["ref"] == 1


def test_build_answer_fails_closed_after_invalid_repair(monkeypatch):
    calls = []
    monkeypatch.setattr(rag_answer, "is_configured", lambda _cfg: True)
    monkeypatch.setattr(
        rag_answer,
        "generate_grounded_answer",
        lambda *_args, **_kwargs: calls.append(True) or "IS 9999 applies [Source 1].",
    )

    response = rag_answer.build_rag_answer(
        "water bottles", "en", [_catalogue_record()], _cfg())

    assert len(calls) == 2
    assert response["kind"] == "grounding_refusal"
    assert response["refused"] is True
    assert response["sources"] == [] and response["citations"] == []
    # The retrieved documents are offered as unnumbered related reading.
    assert response["related_sources"][0]["standard_number"] == "IS 15410:2025"
    assert "ref" not in response["related_sources"][0]
    assert "IS 9999" not in response["text"]
    assert "PRIVATE CATALOGUE BODY" not in response["text"]


def test_rate_limited_model_returns_busy_state(monkeypatch):
    monkeypatch.setattr(rag_answer, "is_configured", lambda _cfg: True)
    monkeypatch.setattr(rag_answer, "generate_grounded_answer", lambda *_a, **_kw: None)
    monkeypatch.setattr(rag_answer, "last_failure", lambda: "http_429")

    response = rag_answer.build_rag_answer("water bottles", "en", [], _cfg())

    assert response["kind"] == "model_busy" and response["retryable"] is True
    assert "try again" in response["text"].lower()


def test_unsupported_canned_designation_is_refused_without_losing_prompt_safety(monkeypatch):
    calls = []
    evidence = _document_chunk() | {"standard_number": "IS 14478:2026"}
    monkeypatch.setattr(rag_llm, "chat_complete", lambda messages, _cfg=None:
                        calls.append(messages) or
                        "The model's grounded synthesis. [IS 101 (Part 2/Sec 6):2026] [Source 1]")

    response = rag_answer.build_rag_answer(
        "What does IS 14478 cover?", "en", [evidence], _cfg())

    assert len(calls) == 2
    assert "never say a user's specific product is approved" in " ".join(
        calls[0][0]["content"].lower().split())
    assert response["kind"] == "grounding_refusal" and response["refused"]
    assert "IS 101" not in response["text"]
