"""Ingest the BIS RAG text corpus into SQLite (documents + chunks + FTS + catalogue).

Usage:
  PYTHONPATH=src python scripts/import_rag_corpus.py --corpus new_data/bis-rag-text-corpus-2026-09-18 --db kb/bis_rag.db
  Add --embedding-model BAAI/bge-small-en-v1.5 to build a local dense index.
  BIS knowledge pages in data/knowledge/ are imported too (--knowledge "" skips them).

Idempotent per source_file (re-import replaces that document's chunks).
Preserves unmatched TXT files (empty provenance) instead of dropping them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant.rag_ingest import import_corpus  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Import BIS RAG text corpus into SQLite")
    ap.add_argument("--corpus", required=True,
                    help="path to extracted corpus dir (contains Files/, data/, *.json)")
    ap.add_argument("--db", default="kb/bis_rag.db",
                    help="target SQLite file (default kb/bis_rag.db)")
    ap.add_argument("--embedding-model",
                    default=os.environ.get("BIS_RAG_EMBEDDING_MODEL", ""),
                    help="optional sentence-transformers model for a dense index")
    ap.add_argument("--knowledge",
                    default=str(Path(__file__).resolve().parents[1] / "data" / "knowledge"),
                    help="BIS knowledge Markdown directory (default data/knowledge)")
    args = ap.parse_args()
    corpus = Path(args.corpus)
    if not (corpus / "Files").is_dir() or not (corpus / "data").is_dir():
        raise SystemExit(f"corpus dir {corpus} must contain Files/ and data/")
    stats = import_corpus(corpus, args.db, embedding_model=args.embedding_model,
                          knowledge_dir=args.knowledge or None)
    print(f"imported {stats.get('documents',0)} documents, {stats.get('chunks',0)} chunks, "
          f"{stats.get('catalogue',0)} catalogue rows -> {args.db}")
    print(f"  knowledge: {stats.get('knowledge_documents', 0)} pages, "
          f"{stats.get('knowledge_chunks', 0)} chunks")
    if args.embedding_model:
        print(f"  dense index: {stats.get('embedded_chunks', 0)} chunks using {args.embedding_model}")
    print(f"  matched={stats.get('matched',0)} unmatched={stats.get('unmatched',0)} "
          f"skipped_empty={stats.get('skipped_empty',0)} fts={stats.get('fts',False)}")


if __name__ == "__main__":
    main()
