"""Optional dense-index configuration and offline fallback contracts."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def clear_model_caches():
    from bis_assistant import rag_embeddings

    rag_embeddings._embedding_model.cache_clear()
    rag_embeddings._cross_encoder.cache_clear()
    yield
    rag_embeddings._embedding_model.cache_clear()
    rag_embeddings._cross_encoder.cache_clear()


def test_config_uses_practical_default_but_allows_explicit_disable(monkeypatch):
    from bis_assistant import rag_config

    monkeypatch.setattr(rag_config, "_cfg_section", lambda name: {"semantic": True})
    monkeypatch.delenv("BIS_RAG_EMBEDDING_MODEL", raising=False)
    assert rag_config.load_rag_config()["embedding_model"] == "BAAI/bge-small-en-v1.5"

    monkeypatch.setenv("BIS_RAG_EMBEDDING_MODEL", "")
    assert rag_config.load_rag_config()["embedding_model"] == ""


def test_runtime_models_are_loaded_cache_only(monkeypatch):
    from bis_assistant import rag_embeddings

    calls = []

    class FakeModel:
        pass

    def fake_sentence_transformer(name, **kwargs):
        calls.append((name, kwargs))
        return FakeModel()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = fake_sentence_transformer
    module.CrossEncoder = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    assert rag_embeddings.get_model("example/model") is not None
    assert rag_embeddings.get_model("example/model") is not None
    assert rag_embeddings.get_model("example/model", allow_download=True) is not None
    assert rag_embeddings._cross_encoder("example/reranker") is not None
    assert calls == [
        ("example/model", {"local_files_only": True}),
        ("example/model", {"local_files_only": False}),
        ("example/reranker", {"local_files_only": True}),
    ]


def test_missing_optional_runtime_degrades_to_lexical(monkeypatch):
    from bis_assistant import rag_embeddings

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert rag_embeddings.get_model("missing/from-cache") is None
    assert rag_embeddings.semantic_score("query", "passage", "missing/from-cache") == 0.0


def test_dense_search_error_returns_no_dense_hits(tmp_path, monkeypatch):
    from bis_assistant import rag_embeddings
    from bis_assistant.rag_store import connect_rag

    db = tmp_path / "rag.db"
    conn = connect_rag(db)
    try:
        doc_id = conn.execute(
            "INSERT INTO corpus_documents(source_file, imported_at) VALUES (?,?)",
            ("Files/sample.txt", "test"),
        ).lastrowid
        chunk_id = conn.execute(
            "INSERT INTO corpus_chunks(doc_id, chunk_index, chunk_text) VALUES (?,?,?)",
            (doc_id, 0, "cached dense passage"),
        ).lastrowid
        conn.execute(
            "INSERT INTO corpus_embeddings(chunk_id, model_name, vector) VALUES (?,?,?)",
            (chunk_id, "model", bytes(4)),
        )
        conn.commit()
    finally:
        conn.close()
    # Keep this test independent of whether numpy is installed in the test env.
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    def fail_index(*args):
        raise RuntimeError("broken dense index")

    monkeypatch.setattr(rag_embeddings, "_dense_index", fail_index)
    assert rag_embeddings.dense_search(db, "query", "model", 5) == []


def _write_minimal_corpus(root: Path) -> None:
    (root / "Files").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "Files" / "sample.txt").write_text(
        "Pressure vessels require traceable BIS evidence and a safe test method.\n",
        encoding="utf-8",
    )
    (root / "data" / "standards_metadata.ndjson").write_text(
        json.dumps({"standardId": 1, "standardNumber": "IS 1:2024",
                    "standardName": "Pressure vessels"}) + "\n",
        encoding="utf-8",
    )
    (root / "data" / "files.ndjson").write_text(
        json.dumps({"path": "sample.pdf", "sourceStandardId": 1,
                    "sourceStandardNumber": "IS 1:2024",
                    "sourceRef": "gazette.pdf", "url": "https://example.test/source"}) + "\n",
        encoding="utf-8",
    )
    (root / "data" / "standard_documents.ndjson").write_text("", encoding="utf-8")
    (root / "conversion_manifest.json").write_text(json.dumps({"records": [
        {"text": "Files/sample.txt", "sourcePdfFilename": "sample.pdf"}
    ]}), encoding="utf-8")


def test_failed_index_build_keeps_corpus_lexically_searchable(tmp_path, monkeypatch):
    from bis_assistant import rag_embeddings
    from bis_assistant.rag_ingest import import_corpus
    from bis_assistant.rag_retriever import search_rag

    corpus = tmp_path / "corpus"
    _write_minimal_corpus(corpus)

    def unavailable_model(name, *, allow_download=False):
        assert allow_download is True
        return None

    monkeypatch.setattr(rag_embeddings, "get_model", unavailable_model)
    db = tmp_path / "rag.db"
    stats = import_corpus(corpus, db, embedding_model="missing/from-cache")

    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    assert stats["embedded_chunks"] == 0
    assert "install sentence-transformers" in stats["embedding_error"]
    hits = search_rag("pressure vessels", db_path=db, semantic=False)
    top = hits[0] if hits else {}
    assert (top.get("related_standard") or top.get("standard_number")) == "IS 1:2024"
