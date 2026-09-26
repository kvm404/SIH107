"""Fetch official BIS web pages and write them as knowledge Markdown for RAG.

Usage:
  python scripts/build_knowledge.py            # writes data/knowledge/generated/
  python scripts/build_knowledge.py --out DIR

Stdlib only (plus the optional `pdftotext` binary for the recognised-lab
list). Each output file has a small front matter block and `## ` sections.
A `<!-- source: URL -->` line under a heading records the page that section
came from, so every chunk keeps a precise citation URL.

The generated files are committed. Re-run this script when BIS updates its
pages, review the diff, then re-import the corpus
(scripts/import_rag_corpus.py picks up data/knowledge automatically).
"""
from __future__ import annotations

import argparse
import html
import http.client
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (BIS-Assistant knowledge builder)"
BIS = "https://www.bis.gov.in"
DELAY_S = 1.0

# (output file, title, [(section title or "", url), ...])
PROSE = [
    ("about-bis.md", "About the Bureau of Indian Standards", [
        ("About BIS and its offices", f"{BIS}/the-bureau/about-bis/"),
    ]),
    ("product-certification.md", "BIS product certification (ISI mark licence)", [
        ("Products under compulsory certification", f"{BIS}/product-certification/products-under-compulsory-certification/"),
        ("Product certification FAQ", f"{BIS}/product-certification/product-certification-faq/"),
        ("BIS Standard Mark and licence number", f"{BIS}/fmcs/certification-process/bis-standard-mark/"),
        ("Know Your Standard portal", f"{BIS}/know-your-standard/"),
    ]),
    ("crs.md", "Compulsory Registration Scheme (CRS) for electronics, IT and solar goods", [
        ("About CRS", "https://www.crsbis.in/BIS/about-crs.do"),
        ("CRS registration steps", "https://www.crsbis.in/BIS/registration-page.do"),
    ]),
    ("fmcs.md", "Foreign Manufacturers Certification Scheme (FMCS)", [
        ("FMCS overview", f"{BIS}/fmcs/fmcs-overview/"),
        ("About FMCS", f"{BIS}/fmcs/certification-process/aboutfmcs/"),
        ("FMCS FAQ", f"{BIS}/fmcs/fmcs-faqs/"),
    ]),
    ("hallmarking.md", "Hallmarking of gold and silver jewellery (HUID)", [
        ("Hallmarking overview", f"{BIS}/hallmarking-overview/"),
        ("Hallmarking FAQ", f"{BIS}/hallmarking-overview/hallmarking-faqs/hallmarking-faq/"),
        ("Hallmarking FAQ for consumers", f"{BIS}/hallmarking-overview/hallmarking-faqs/mandatory/"),
        ("Hallmarking consumer protection", f"{BIS}/hallmarking-overview/consumer-protection/"),
    ]),
    ("consumer-complaints.md", "Consumer protection and complaints to BIS", [
        ("Consumer overview", f"{BIS}/consumer-overview/"),
        ("How to lodge a complaint", f"{BIS}/consumer-overview/consumer-protection/"),
        ("Online complaint registration", f"{BIS}/consumer-overview/online-complaint-registration/"),
        ("Consumer FAQ", f"{BIS}/consumer-overview/for-consumers-faq/"),
    ]),
    ("bis-care-app.md", "BIS Care mobile app", [
        ("BIS Care app features", f"{BIS}/bis-apps/"),
    ]),
    ("laboratories.md", "BIS laboratories, recognised labs and testing", [
        ("Laboratory services overview", f"{BIS}/laboratorys/laboratory-services-overview/"),
        ("Laboratory FAQ", f"{BIS}/laboratorys/laboratory-services-overview/laboratory-faq/"),
        ("How a lab applies for BIS recognition", f"{BIS}/laboratorys/how-to-apply-for-bis-recognition/"),
        ("BIS laboratory directory", f"{BIS}/directory/laboratory/"),
    ]),
    ("management-systems.md", "BIS management systems certification (ISO 9001 and others)", [
        ("Management systems certification scheme", f"{BIS}/system-certification-overview/systems-certification/"),
        ("Systems under certification", f"{BIS}/system-certification-overview/systems-under-certification/"),
    ]),
    ("training-and-outreach.md", "BIS training (NITS), young professionals and students", [
        ("National Institute of Training for Standardization (NITS)", f"{BIS}/training-2/overview-of-nits/"),
        ("Training programmes", f"{BIS}/training-2/training-programmes/"),
        ("Manak Pravardhak programme for young professionals", f"{BIS}/manak-pravardhak-programme/"),
    ]),
]

SCHEME_I = f"{BIS}/product-certification/products-under-compulsory-certification/scheme-i-mark-scheme/"
SCHEME_II = f"{BIS}/product-certification/products-under-compulsory-certification/scheme-ii-registration-scheme/"
SCHEME_IV = f"{BIS}/product-certification/products-under-compulsory-certification/scheme-4/"
UPCOMING = f"{BIS}/upcoming-qcos-notified-and-due-for-implementation/"
GROUP1_PAGE = f"{BIS}/laboratorys/list-of-bis-recognized-lab/"


# ---------------------------------------------------------------------------
# fetching

def fetch_bytes(url: str, attempts: int = 4) -> bytes:
    """GET with a polite delay and a few retries (BIS resets busy connections)."""
    for attempt in range(attempts):
        time.sleep(DELAY_S * (2 ** attempt))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (OSError, http.client.HTTPException):
            if attempt + 1 == attempts:
                raise
    raise AssertionError("unreachable")


def fetch(url: str) -> str:
    sep = "&" if "?" in url else "?"
    full = url + (f"{sep}lang=en" if url.startswith(BIS) else "")
    return fetch_bytes(full).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# prose extraction

_BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br",
          "section", "article", "table", "ul", "ol"}
_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "form",
         "svg", "button", "select"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        if tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in _BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("### ")
        if tag == "li":
            self.out.append("- ")

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        if tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


_DROP_LINE = re.compile(
    r"^(read more.*|click here.*|pause|play|scan to download|explore|user manual|"
    r"overview|-|\||##+|###|- /|home|.*\(size\s*[–-].*\)|know your standard portal video|"
    r"bis care (app|video)|bureau of indian standards|statutory notifications)$", re.I)


def page_text(raw_html: str) -> tuple[str, str]:
    """Return (main text, last-updated string) for a BIS WordPress page."""
    s = raw_html
    for pat in (r'<div[^>]+class="[^"]*entry-content[^"]*"', r"<main\b", r"<article\b"):
        m = re.search(pat, s)
        if m:
            s = s[m.start():]
            break
    s = re.split(r"<footer\b", s)[0]
    p = _Text()
    p.feed(s)
    lines = [" ".join(ln.split()) for ln in html.unescape("".join(p.out)).splitlines()]
    lines = [ln for ln in lines if ln]
    # Content starts after the last breadcrumb separator ("- /").
    crumbs = [i for i, ln in enumerate(lines) if ln == "- /"]
    if crumbs:
        lines = lines[crumbs[-1] + 2:]
    updated = ""
    body: list[str] = []
    for ln in lines:
        if ln.startswith("Last Updated on"):
            updated = ln.replace("Last Updated on", "").strip()
            break
        if ln in ("Pause", "Play"):
            break
        if _DROP_LINE.match(ln.strip(" -")) or _DROP_LINE.match(ln):
            continue
        ln = re.sub(r"\s*\(size\s*[–-][^)]*\)", "", ln)
        ln = ln.replace("[at]", "@").replace("[dot]", ".").replace("[-]", "-")
        ln = re.sub(r"^### ", "", ln)
        if body and body[-1] == ln:
            continue
        body.append(ln)
    # Drop the page's own title line (repeated as our section heading).
    if body and len(body[0]) < 90 and not body[0].endswith("."):
        body = body[1:]
    text = "\n".join(body)
    text = re.sub(r"\n- (?=\n)", "", text)
    return text.strip(), updated


# ---------------------------------------------------------------------------
# tables (rowspan/colspan aware)

class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[dict]]] = []
        self.row: list[dict] | None = None
        self.cell: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            def span(key: str) -> int:
                return int(re.sub(r"\D", "", a.get(key) or "1") or 1)
            self.cell = {"text": "", "rowspan": span("rowspan"), "colspan": span("colspan")}
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += " "

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.cell["text"] = " ".join(html.unescape(self.cell["text"]).split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.tables:
                self.tables[-1].append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def _grid(rows: list[list[dict]]) -> list[list[str]]:
    out: list[list[str]] = []
    pending: dict[int, list] = {}
    for cells in rows:
        line: list[str] = []
        col, ci = 0, 0
        while ci < len(cells) or any(c >= col for c in pending):
            if col in pending:
                remaining, text = pending[col]
                line.append(text)
                if remaining > 1:
                    pending[col] = [remaining - 1, text]
                else:
                    del pending[col]
                col += 1
                continue
            if ci >= len(cells):
                break
            c = cells[ci]
            ci += 1
            for _ in range(c["colspan"]):
                line.append(c["text"])
                if c["rowspan"] > 1:
                    pending[col] = [c["rowspan"] - 1, c["text"]]
                col += 1
        out.append(line)
    return out


def tables(raw_html: str) -> list[list[list[str]]]:
    p = _Tables()
    p.feed(raw_html)
    return [_grid(t) for t in p.tables if len(t) > 1]


# ---------------------------------------------------------------------------
# writers

def front(title: str, url: str, doc_type: str, updated: str = "") -> str:
    lines = ["---", f"title: {title}", f"source_url: {url}", f"doc_type: {doc_type}",
             f"retrieved: {date.today().isoformat()}"]
    if updated:
        lines.append(f"source_last_updated: {updated}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _qco(text: str, limit: int = 420) -> str:
    text = re.sub(r"^\d+\.\s*", "", text.strip())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _is_heading_row(row: list[str]) -> bool:
    return len(set(row)) == 1 and bool(row[0])


def build_scheme_i(raw: str) -> str:
    grid = max(tables(raw), key=len)
    out = [front("Products under compulsory BIS certification: Scheme-I (ISI Mark)",
                 SCHEME_I, "compulsory_list"),
           "# Products under compulsory BIS certification: Scheme-I (ISI Mark)\n",
           "The Central Government makes BIS certification compulsory for these products "
           "through Quality Control Orders (QCOs). They must carry the ISI mark under a "
           "valid BIS licence. Each line gives the Indian Standard, the product and the "
           "Quality Control Order.\n"]
    category, last_order = "General", ""
    for row in grid[1:]:
        if _is_heading_row(row):
            if row[0].startswith("*"):
                continue
            category, last_order = row[0].rstrip(" :"), ""
            out.append(f"\n## {category}\n<!-- source: {SCHEME_I} -->")
            continue
        if len(row) < 4:
            continue
        is_no, product, order = row[1].strip(), row[2].strip(), row[3]
        if not is_no.upper().startswith("IS") and not re.match(r"\d", is_no):
            continue
        if order != last_order:
            out.append(f"Quality Control Order details: {_qco(order)}")
            last_order = order
        status = ""
        low = (order + " " + category).lower()
        if "withdrawal of" in low or "de-notified" in low or "denotified" in low:
            status = " Status: QCO withdrawn or de-notified; check the notification."
        out.append(f"- {is_no}: {product}. Compulsory under {_order_name(order)}.{status}")
    return "\n".join(out) + "\n"


_ORDER_TITLE = re.compile(
    r"([A-Z][A-Za-z ,&/’'\-]*?\s*\((?:Quality\s*Control|Requirements?\s+(?:for|of)\s+"
    r"Compulsory\s+Registration)\s*\)\s*(?:(?:Second|Third)?\s*Amendment\s+)?Order,?\s*\d{4})")


def _order_name(order: str) -> str:
    """Short order name, e.g. 'Toys (Quality Control) Order, 2020'."""
    text = re.sub(r"^\d+\.\s*", "", order.strip())
    m = _ORDER_TITLE.search(text)
    if m:
        name = re.sub(r"^(?:Amendments?\s+to\s+the\s+|Superseded\s+by\s+)", "",
                      m.group(1).strip(" .“\""), flags=re.I)
        return " ".join(name.split())
    m = re.search(r"^(.*?\bOrder,?\s*\d{4})", text)
    name = m.group(1) if m else text[:100]
    return name.strip(" .") or "a Quality Control Order"


def build_scheme_ii(raw: str) -> str:
    out = [front("Products under compulsory registration: Scheme-II (CRS, Registration Mark)",
                 SCHEME_II, "compulsory_list"),
           "# Products under compulsory registration: Scheme-II (CRS)\n",
           "These products need registration with BIS under the Compulsory Registration "
           "Scheme (Scheme-II, self-declaration of conformity) and must carry the Standard "
           "Mark with a unique R-number. Each line gives the product category, the "
           "Indian Standard and the order.\n"]
    for grid in tables(raw):
        header = " ".join(grid[0]).lower()
        if "product category" not in header:
            continue
        group = ""
        rows = [r for r in grid[1:] if len(r) >= 5 and not _is_heading_row(r)]
        for r in rows:
            order = _qco(r[4], 260)
            name = _order_name(r[4])
            if name.startswith("a "):
                name = group or "Compulsory Registration Order"
            if name != group:
                group = name
                out.append(f"\n## {name[:120]}\n<!-- source: {SCHEME_II} -->")
            out.append(f"- {r[3]}: {r[1]} ({r[2]}). Order: {order}.")
    return "\n".join(out) + "\n"


def build_scheme_iv(raw: str) -> str:
    grid = max(tables(raw), key=len)
    out = [front("Products under compulsory certification: Scheme-IV (Certificate of Conformity)",
                 SCHEME_IV, "compulsory_list"),
           "# Products under compulsory certification: Scheme-IV (Certificate of Conformity)\n",
           f"## Scheme-IV products\n<!-- source: {SCHEME_IV} -->"]
    for r in grid[1:]:
        if len(r) >= 4 and not _is_heading_row(r):
            out.append(f"- {r[1]}. Essential requirement: {r[2]} QCO: {_qco(r[3])}.")
    return "\n".join(out) + "\n"


def build_upcoming(raw: str) -> str:
    grid = max(tables(raw), key=len)
    out = [front("Upcoming Quality Control Orders: notified and due for implementation",
                 UPCOMING, "compulsory_list"),
           "# Upcoming Quality Control Orders (QCOs)\n",
           "These QCOs have been notified but are not yet in force. From the enforcement "
           "date, the product must carry the BIS Standard Mark under a licence.\n"]
    ministry = ""
    for r in grid[1:]:
        if len(r) < 6 or _is_heading_row(r):
            continue
        if r[1] != ministry:
            ministry = r[1]
            out.append(f"\n## {ministry}\n<!-- source: {UPCOMING} -->")
        std = r[4] if r[4].upper().startswith("IS") else f"IS {r[4]}"
        out.append(f"- {r[2]}: {std}. Enforcement date: {r[5]}.")
    return "\n".join(out) + "\n"


_LAB_ROW = re.compile(
    r"^\s*(\d+)\.\s+(.+?)\s{2,}([A-Z][A-Za-z.&() ]+?)\s{2,}(Private|Govt\.?|Government|PSU|Public|Semi[- ]Govt\.?)\s+"
    r"(\d{6,8})\s+(\d{2}\.\d{2}\.\d{4})\s*(.*)$")


_STATES = {
    "a.p.": "Andhra Pradesh", "ap": "Andhra Pradesh", "andhra": "Andhra Pradesh",
    "andhra pradesh": "Andhra Pradesh", "daman": "Dadra & Nagar Haveli and Daman & Diu",
    "h.p.": "Himachal Pradesh", "himachal": "Himachal Pradesh", "harayana": "Haryana",
    "jammu &": "Jammu & Kashmir", "karnatka": "Karnataka", "m.p": "Madhya Pradesh",
    "m.p.": "Madhya Pradesh", "mp": "Madhya Pradesh", "madhya": "Madhya Pradesh",
    "maharshtra": "Maharashtra", "orissa": "Odisha", "rajasth": "Rajasthan",
    "tamilnadu": "Tamil Nadu", "u.p": "Uttar Pradesh", "u.p.": "Uttar Pradesh",
    "up": "Uttar Pradesh", "utter pradesh": "Uttar Pradesh", "uttrakhand": "Uttarakhand",
    "w.b.": "West Bengal", "w.b": "West Bengal", "wb": "West Bengal",
}


def _state(raw: str) -> str:
    key = " ".join(raw.split()).lower()
    return _STATES.get(key, " ".join(raw.split()))


def parse_group1(text: str) -> list[dict]:
    labs: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _LAB_ROW.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(2).strip()
        remark = m.group(7).strip()
        j = i + 1
        # Name continues in the name column on following non-blank lines.
        while j < len(lines) and lines[j].strip() and not _LAB_ROW.match(lines[j]):
            head = lines[j][:48].strip()
            tail = lines[j][90:].strip()
            if head and not head.startswith("Sl."):
                name += " " + head
            if tail and len(remark) < 120:
                remark += " " + tail
            j += 1
        suspended = bool(re.match(r"(?i)suspended", remark)) and "revoked" not in remark.lower()[:40]
        labs.append({"name": " ".join(name.split()), "state": _state(m.group(3)),
                     "type": m.group(4), "osl": m.group(5), "valid_until": m.group(6),
                     "currently_suspended": suspended})
        i = j
    return labs


def build_group1(pdf_url: str) -> tuple[str, list[dict]]:
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not installed")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "g1.pdf"
        pdf.write_bytes(fetch_bytes(pdf_url))
        txt = Path(tmp) / "g1.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        labs = parse_group1(txt.read_text(errors="replace"))
    by_state: dict[str, list[dict]] = {}
    for lab in labs:
        by_state.setdefault(lab["state"], []).append(lab)
    out = [front("BIS recognised laboratories (Group-1) by state", pdf_url, "lab_directory"),
           "# BIS recognised laboratories (Group-1) by state\n",
           "Laboratories recognised by BIS under its Laboratory Recognition Scheme. Test "
           "reports from these labs are accepted for BIS certification. The products each "
           "lab can test are listed on the LIMS portal (lims.bis.gov.in, search by IS number).\n"]
    for state in sorted(by_state):
        out.append(f"\n## Recognised labs in {state}\n<!-- source: {pdf_url} -->")
        for lab in by_state[state]:
            note = " Currently suspended." if lab["currently_suspended"] else ""
            out.append(f"- {lab['name']} ({lab['type']}), OSL code {lab['osl']}, "
                       f"recognition valid up to {lab['valid_until']}.{note}")
    return "\n".join(out) + "\n", labs


def build_prose(title: str, sections: list[tuple[str, str]]) -> str:
    parts, first_url, updated_any = [], sections[0][1], ""
    for heading, url in sections:
        text, updated = page_text(fetch(url))
        updated_any = updated_any or updated
        if not text:
            print(f"  ! empty page: {url}", file=sys.stderr)
            continue
        parts.append(f"## {heading}\n<!-- source: {url} -->\n{text}\n")
    return front(title, first_url, "bis_guide", updated_any) + f"# {title}\n\n" + "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "knowledge" / "generated"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fname, title, sections in PROSE:
        (out / fname).write_text(build_prose(title, sections), encoding="utf-8")
        print(f"wrote {fname}")
    for fname, url, builder in (
            ("compulsory-scheme-i.md", SCHEME_I, build_scheme_i),
            ("compulsory-scheme-ii-crs.md", SCHEME_II, build_scheme_ii),
            ("compulsory-scheme-iv.md", SCHEME_IV, build_scheme_iv),
            ("upcoming-qcos.md", UPCOMING, build_upcoming)):
        (out / fname).write_text(builder(fetch(url)), encoding="utf-8")
        print(f"wrote {fname}")

    page = fetch(GROUP1_PAGE)
    m = re.search(r'href="([^"]+Group[_-]?1[^"]*\.pdf)"', page, re.I)
    if m:
        try:
            md, labs = build_group1(m.group(1))
            (out / "recognised-labs.md").write_text(md, encoding="utf-8")
            (ROOT / "data" / "recognised_labs.json").write_text(
                json.dumps({"source_url": m.group(1), "retrieved": date.today().isoformat(),
                            "labs": labs}, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")
            print(f"wrote recognised-labs.md ({len(labs)} labs)")
        except Exception as exc:  # optional: keep the rest of the build
            print(f"  ! skipped Group-1 lab list: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
