"""Parse and chunk the BIS knowledge pages in ``data/knowledge``.

Each file is Markdown with a small front matter block::

    ---
    title: ...
    source_url: https://www.bis.gov.in/...
    doc_type: bis_guide | compulsory_list | lab_directory
    ---
    ## Section
    <!-- source: https://... -->
    text...

Chunks follow ``## `` sections. Long sections split on line boundaries so a
list row (one product, one lab) is never cut in half, and every chunk keeps
its section heading and the source URL of the page it came from.
"""
from __future__ import annotations

import re
from pathlib import Path

KNOWLEDGE_DOC_TYPES = ("bis_guide", "compulsory_list", "lab_directory")
CHUNK_WORDS = 220

_SOURCE_RE = re.compile(r"^<!--\s*source:\s*(\S+)\s*-->$")


def parse_knowledge_file(path: Path) -> dict:
    """Return {title, source_url, doc_type, retrieved, sections: [...]}."""
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    meta[key.strip()] = value.strip()
            body = text[end + 4:]
    doc_type = meta.get("doc_type", "bis_guide")
    if doc_type not in KNOWLEDGE_DOC_TYPES:
        doc_type = "bis_guide"
    sections: list[dict] = []
    current = {"heading": "", "source_url": meta.get("source_url", ""), "lines": []}
    for line in body.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if any(ln.strip() for ln in current["lines"]):
                sections.append(current)
            current = {"heading": line[3:].strip(),
                       "source_url": meta.get("source_url", ""), "lines": []}
            continue
        m = _SOURCE_RE.match(line.strip())
        if m:
            current["source_url"] = m.group(1)
            continue
        current["lines"].append(line.rstrip())
    if any(ln.strip() for ln in current["lines"]):
        sections.append(current)
    return {
        "title": meta.get("title") or path.stem.replace("-", " ").title(),
        "source_url": meta.get("source_url", ""),
        "doc_type": doc_type,
        "retrieved": meta.get("retrieved", ""),
        "sections": sections,
    }


def chunk_sections(sections: list[dict], max_words: int = CHUNK_WORDS) -> list[dict]:
    """Split sections into line-aligned chunks: [{chunk_text, heading, source_url}]."""
    chunks: list[dict] = []
    for section in sections:
        lines = [ln for ln in section["lines"] if ln.strip()]
        # Lines that give context to the rows under them (e.g. the QCO a list
        # of products falls under) are repeated at the top of later chunks.
        context = ""
        buf: list[str] = []
        words = 0

        def flush() -> None:
            nonlocal buf, words
            if buf:
                chunks.append({"chunk_text": "\n".join(buf).strip(),
                               "heading": section["heading"],
                               "source_url": section["source_url"]})
            buf, words = ([context] if context else []), len(context.split())

        for line in lines:
            n = len(line.split())
            if buf and words + n > max_words and words > len(context.split()):
                flush()
            if not line.startswith("- "):
                context = line if len(line.split()) <= 80 else ""
            buf.append(line)
            words += n
        if buf and (len(buf) > 1 or buf[0] != context):
            chunks.append({"chunk_text": "\n".join(buf).strip(),
                           "heading": section["heading"],
                           "source_url": section["source_url"]})
    return chunks


def knowledge_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())
