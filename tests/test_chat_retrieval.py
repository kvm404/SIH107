"""Regression tests for the chat retrieval merge and its safe diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from bis_assistant.rag_store import connect_rag, now

WATER_BOTTLE_QUERY = (
    "I want to start a water bottle plastic company. What standards do I care about?"
)


def _insert_catalogue(conn, standard_id: int, number: str, title: str,
                      department: str = "PLASTICS DEPARTMENT") -> None:
    conn.execute(
        "INSERT INTO catalogue_standards VALUES (?,?,?,?,?,?,?,?)",
        (standard_id, number, f"{number} {title}", title, department,
         "PLASTICS PACKAGING", "Product Specification", "2025-01-01"),
    )


def _insert_document(conn, standard_id: int, number: str, title: str,
                     chunks: list[str]) -> None:
    cur = conn.execute(
        "INSERT INTO corpus_documents(source_file, standard_id, standard_number,"
        " title, department, committee, doc_type, category, source_url, imported_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (f"Files/{standard_id}.txt", standard_id, number, title, "", "", "gazette",
         "gazette", f"https://example.invalid/{standard_id}", now()),
    )
    for index, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO corpus_chunks(doc_id, chunk_index, chunk_text, heading,"
            " token_count, standard_number, doc_type, source_url)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (cur.lastrowid, index, chunk, "", len(chunk.split()), number, "gazette",
             f"https://example.invalid/{standard_id}"),
        )


@pytest.fixture
def water_bottle_db(tmp_path):
    db = tmp_path / "water_bottle_rag.db"
    conn = connect_rag(db)
    try:
        _insert_catalogue(
            conn, 15410, "IS 15410:2025",
            "Plastic bottles/containers for packaging of natural mineral water "
            "and packaged drinking water - specification",
        )
        _insert_catalogue(
            conn, 6719, "IS 6719:2026",
            "Moulded PVC unit outsole for footwear",
        )
        # These deliberately reproduce the distracting one-topic FTS hits:
        # one shares only "water", one only "plastic" with the query.
        _insert_document(
            conn, 18474, "IS 18474 (Part 5):2026", "Transfusion equipment",
            ["Water-based cleaning fluids for medical transfusion equipment "
             "and other clinical use."],
        )
        _insert_document(
            conn, 6719, "IS 6719:2026", "Moulded PVC unit outsole",
            ["Plastic PVC outsole material and footwear construction."],
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _rag_cfg(monkeypatch, db, **overrides):
    from bis_assistant import assistant

    cfg = {
        "enabled": True,
        "db_path": str(db),
        "top_k": 5,
        "catalogue_top_k": 5,
        "minimum_relevance": 0.25,
        "weight_lexical": 1.0,
        "weight_semantic": 0.3,
        "exact_boost": 50.0,
        "semantic": False,
        "embedding_model": "",
        **overrides,
    }
    monkeypatch.setattr(assistant, "load_rag_config", lambda: cfg)
    return cfg


def test_water_bottle_query_prefers_catalogue_and_rejects_irrelevant_chunks(
        water_bottle_db, monkeypatch):
    _rag_cfg(monkeypatch, water_bottle_db)
    from bis_assistant.assistant import _rag_lookup

    evidence, cfg = _rag_lookup(WATER_BOTTLE_QUERY)

    assert evidence
    assert evidence[0]["standard_number"] == "IS 15410:2025"
    assert evidence[0]["evidence_type"] == "catalogue_record"
    assert evidence[0]["metadata_only"] is True
    assert evidence[0]["chunk_text"] == ""
    assert "full standard text was not retrieved" in evidence[0]["metadata_notice"]
    assert not any(item["standard_number"] in {
        "IS 18474 (Part 5):2026", "IS 6719:2026"
    } and item["evidence_type"] == "document_chunk" for item in evidence)

    diagnostics = cfg["retrieval_diagnostics"]
    assert diagnostics["branch"] == "hybrid"
    assert diagnostics["branches"]["document_chunks"]["rejected_count"] >= 1
    assert diagnostics["branches"]["document_chunks"]["rejection_reasons"].get(
        "insufficient_query_term_overlap", 0) >= 1
    assert diagnostics["rejection_reasons"].get("missing_query_terms", 0) >= 1
    assert diagnostics["selected"][0]["branch"] == "catalogue_record"
    assert diagnostics["selected"][0]["rank"] == 1
    assert isinstance(diagnostics["selected"][0]["score"], float)
    assert diagnostics["results"][0]["selected"] is True
    assert diagnostics["results"][0]["source_id"] == "IS 15410:2025"
    serialized = repr(diagnostics)
    assert WATER_BOTTLE_QUERY not in serialized
    assert "Plastic PVC outsole material" not in serialized

    from bis_assistant.server import _safe_retrieval_diagnostics
    safe = _safe_retrieval_diagnostics(diagnostics)
    assert safe["branch"] == "hybrid"
    assert safe["results"][0]["evidence_type"] == "catalogue_record"
    assert safe["results"][0]["source_id"] == "IS 15410:2025"
    assert safe["rejection_reasons"].get("missing_query_terms", 0) >= 1


def test_chat_returns_model_text_and_safe_retrieval_diagnostics(water_bottle_db,
                                                                  monkeypatch):
    _rag_cfg(monkeypatch, water_bottle_db)
    from bis_assistant import assistant

    monkeypatch.setattr(assistant, "load_llm_config", lambda: {"model": "m"})
    monkeypatch.setattr(assistant, "is_configured", lambda _cfg: True)
    seen = {}

    def model_generated(query, lang, evidence, cfg, history=None):
        seen["evidence"] = evidence
        return {"text": "MODEL-GENERATED RESPONSE", "kind": "llm_answer"}

    monkeypatch.setattr(assistant, "build_rag_answer", model_generated)
    response = assistant.answer(WATER_BOTTLE_QUERY)

    assert response["text"] == "MODEL-GENERATED RESPONSE"
    assert any(e["evidence_type"] == "catalogue_record" for e in seen["evidence"])
    assert response["retrieval_diagnostics"]["selected_count"] == len(seen["evidence"])
    assert WATER_BOTTLE_QUERY not in repr(response["retrieval_diagnostics"])
    assert all("chunk_text" not in row for row in response["retrieval_diagnostics"]["selected"])


def test_combined_top_k_diagnostics_count_mixed_remaining_candidates(monkeypatch):
    from bis_assistant import assistant, catalogue_search, rag_retriever

    cfg = {
        "enabled": True,
        "top_k": 2,
        "catalogue_top_k": 5,
        "minimum_relevance": 0.25,
        "semantic": False,
    }
    monkeypatch.setattr(assistant, "load_rag_config", lambda: cfg)

    documents = [
        {"evidence_type": "document_chunk", "relevance": 0.9,
         "chunk_id": 1, "chunk_index": 0, "standard_number": "IS 1"},
        {"evidence_type": "document_chunk", "relevance": 0.7,
         "chunk_id": 2, "chunk_index": 0, "standard_number": "IS 2"},
        {"evidence_type": "document_chunk", "relevance": 0.5,
         "chunk_id": 3, "chunk_index": 0, "standard_number": "IS 3"},
    ]
    catalogue = [
        {"relevance": 0.8, "standard_id": 4, "standard_number": "IS 4"},
        {"relevance": 0.6, "standard_id": 5, "standard_number": "IS 5"},
    ]

    def search_documents(*_args, diagnostics, **_kwargs):
        diagnostics.update(candidate_count=len(documents), rejection_reasons={})
        return documents

    def search_catalogue(*_args, diagnostics, **_kwargs):
        diagnostics.update(candidate_count=len(catalogue), rejection_reasons={})
        return catalogue

    monkeypatch.setattr(rag_retriever, "search_rag", search_documents)
    monkeypatch.setattr(catalogue_search, "search_catalogue", search_catalogue)

    evidence, result_cfg = assistant._rag_lookup("query")

    assert [item["relevance"] for item in evidence] == [0.9, 0.8]
    diagnostics = result_cfg["retrieval_diagnostics"]
    branch_reasons = diagnostics["branches"]
    combined_top_k_rejections = sum(
        branch["rejection_reasons"].get("combined_top_k_limit", 0)
        for branch in branch_reasons.values()
    )
    assert combined_top_k_rejections == diagnostics["rejection_reasons"].get(
        "outside_top_k", 0)


def test_document_results_are_deduplicated_and_diversified(tmp_path):
    from bis_assistant.rag_retriever import search_rag

    db = tmp_path / "diverse.db"
    conn = connect_rag(db)
    try:
        _insert_document(conn, 1, "IS 1:2025", "Plastic water bottles", [
            "Plastic bottles for packaged drinking water, technical description one.",
            "Plastic bottles for packaged drinking water, technical description one.",
            "Plastic bottles for packaged drinking water, technical description two.",
            "Plastic bottles for packaged drinking water, technical description three.",
            "Plastic bottles for packaged drinking water, technical description four.",
        ])
        _insert_document(conn, 2, "IS 2:2025", "Water bottles", [
            "Reusable water bottle construction and material requirements.",
        ])
        conn.commit()
    finally:
        conn.close()

    diagnostics = {}
    result = search_rag("plastic water bottle requirements", top_k=5,
                        db_path=db, semantic=False, diagnostics=diagnostics)

    assert result
    assert all(item["evidence_type"] == "document_chunk" for item in result)
    # Gazette schedules are filed under one standard but list many, so
    # non-exact hits keep that filing only as related_standard.
    def filed(item):
        return item.get("related_standard") or item["standard_number"]

    assert len([item for item in result if filed(item) == "IS 1:2025"]) <= 3
    assert len({" ".join(item["chunk_text"].split()) for item in result}) == len(result)
    assert any(filed(item) == "IS 2:2025" for item in result)
    assert diagnostics["branch"] == "document_chunks"
    assert diagnostics["selected_count"] == len(result)
    assert diagnostics["rejection_reasons"].get("duplicate_chunk", 0) >= 1
    assert diagnostics["rejection_reasons"].get("source_diversity_limit", 0) >= 1


def test_semantic_score_and_scattered_cross_references_do_not_qualify_document(
        tmp_path, monkeypatch):
    from bis_assistant import rag_retriever

    db = tmp_path / "cross_reference_false_positive.db"
    conn = connect_rag(db)
    try:
        _insert_document(
            conn, 8181, "IS 8181:2026", "Specification for repositor, Iris",
            ["Aluminium/Plastic Caps are cross-referenced for infusion bottles. "
             "Another unrelated section mentions bottles used for cold water."]
        )
        chunk_id = conn.execute("SELECT id FROM corpus_chunks").fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    # Keep the semantic signal deterministic while reproducing the real index's
    # misleading cosine score for this unrelated, cross-reference-heavy chunk.
    monkeypatch.setattr(
        rag_retriever.emb, "dense_search", lambda *_args: [(chunk_id, 0.643)])
    result = rag_retriever.search_rag(
        WATER_BOTTLE_QUERY, top_k=5, db_path=db, semantic=True,
        embedding_model="deterministic-test-model",
    )

    assert not any(item["standard_number"] == "IS 8181:2026" for item in result)


def test_retrieval_controls_read_config_and_clamp_environment_overrides(monkeypatch):
    from bis_assistant import rag_config

    monkeypatch.setattr(rag_config, "_cfg_section", lambda _name: {
        "minimum_relevance": 0.4, "catalogue_top_k": 7,
    })
    monkeypatch.delenv("BIS_RAG_MINIMUM_RELEVANCE", raising=False)
    monkeypatch.delenv("BIS_RAG_CATALOGUE_TOP_K", raising=False)
    configured = rag_config.load_rag_config()
    assert configured["minimum_relevance"] == 0.4
    assert configured["catalogue_top_k"] == 7

    monkeypatch.setenv("BIS_RAG_MINIMUM_RELEVANCE", "3")
    monkeypatch.setenv("BIS_RAG_CATALOGUE_TOP_K", "1000")
    overridden = rag_config.load_rag_config()
    assert overridden["minimum_relevance"] == 1.0
    assert overridden["catalogue_top_k"] == 100

    monkeypatch.setenv("BIS_RAG_MINIMUM_RELEVANCE", "invalid")
    monkeypatch.setenv("BIS_RAG_CATALOGUE_TOP_K", "0")
    fallback = rag_config.load_rag_config()
    assert fallback["minimum_relevance"] == 0.25
    assert fallback["catalogue_top_k"] == 1
