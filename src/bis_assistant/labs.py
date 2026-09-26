"""Find laboratories that test a product's Indian Standard (BIS LIMS).

Lab questions ("where can I test helmets?", "lab in Pune for LED lamps")
need the IS number first. It comes from the query itself or from the
retrieved compulsory-product lists and catalogue rows. The labs for that
standard come from the BIS LIMS "search by IS number" page, joined with the
Group-1 recognised lab list for each lab's state.

Lookups use the committed cache ``data/lims_labs.json`` first (built by
``scripts/prefetch_lims.py``), then a live LIMS request when
``BIS_LIMS_LIVE`` is not ``0``. Live results are cached in memory only.
The result is one evidence row the model can cite like any other source.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("bis.labs")

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "lims_labs.json"
LABS_PATH = ROOT / "data" / "recognised_labs.json"
LIMS_SEARCH = "https://lims.bis.gov.in/home/search_is_number/"
LIMS_PORTAL = "https://lims.bis.gov.in/"

_LAB_QUERY_RE = re.compile(
    r"\b(labs?|laborator(?:y|ies)|test(?:ing|ed)?\s+(?:lab|centre|center|facilit)|"
    r"where\s+(?:can|do|to)\s+(?:i|we)\s+(?:get\s+)?(?:it\s+|my\s+\w+\s+)?test|"
    r"test\s+report|get\s+(?:\w+\s+){0,3}tested)\b"
    r"|प्रयोगशाला|लैब|लैब्स|परीक्षण|जांच\s+केंद्र",
    re.IGNORECASE)
_IS_RE = re.compile(r"\bIS\s*[:/]?\s*(\d{2,5})(?:\s*\(\s*Part\s*([0-9A-Za-z]+)[^)]*\))?",
                    re.IGNORECASE)
_LAB_WORDS = re.compile(
    r"\b(labs?|laborator(?:y|ies)|testing|tests?|tested|centres?|centers?|facilit(?:y|ies)|"
    r"suggest|recommend|nearby|near|in|at|for|my|a|an|the|to|where|can|i|we|get|find|"
    r"which|bis|recogni[sz]ed|approved|please|any|some|good|best)\b",
    re.IGNORECASE)

# Major cities to states, to put nearby labs first. Lab names often carry the
# city too, which is checked directly.
_CITY_STATE = {
    "pune": "Maharashtra", "mumbai": "Maharashtra", "nagpur": "Maharashtra",
    "nashik": "Maharashtra", "thane": "Maharashtra", "aurangabad": "Maharashtra",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu",
    "hyderabad": "Telangana", "secunderabad": "Telangana",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "kolkata": "West Bengal", "durgapur": "West Bengal", "delhi": "Delhi",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh", "lucknow": "Uttar Pradesh",
    "kanpur": "Uttar Pradesh", "sahibabad": "Uttar Pradesh",
    "gurugram": "Haryana", "gurgaon": "Haryana", "faridabad": "Haryana", "sonipat": "Haryana",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat", "rajkot": "Gujarat",
    "jaipur": "Rajasthan", "chandigarh": "Punjab", "mohali": "Punjab", "ludhiana": "Punjab",
    "kochi": "Kerala", "thiruvananthapuram": "Kerala", "bhopal": "Madhya Pradesh",
    "indore": "Madhya Pradesh", "patna": "Bihar", "guwahati": "Assam",
    "bhubaneswar": "Odisha", "dehradun": "Uttarakhand", "raipur": "Chhattisgarh",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "jammu": "Jammu & Kashmir",
}
_STATE_NAMES = sorted(set(_CITY_STATE.values()) | {
    "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Kerala", "Gujarat",
    "Rajasthan", "Punjab", "Haryana", "Uttar Pradesh", "West Bengal", "Bihar", "Odisha",
    "Assam", "Madhya Pradesh", "Andhra Pradesh", "Uttarakhand", "Himachal Pradesh"})
LIMS_PAGE_ROWS = 30

_memory: dict[str, dict] = {}
_lock = threading.Lock()


def is_lab_query(query: str) -> bool:
    return bool(_LAB_QUERY_RE.search(query or ""))


def lims_url(base: str, part: str = "") -> str:
    params = {"is_number__doc_no": base}
    if part:
        params["is_number__part"] = part
    return LIMS_SEARCH + "?" + urllib.parse.urlencode(params)


def _clean(cell: str) -> str:
    cell = re.sub(r"<[^>]+>", " ", cell)
    return " ".join(html.unescape(cell).split())


def parse_lims(page: str, limit: int = 40) -> list[dict]:
    """Parse LIMS search rows. Each row embeds a charges modal with its own
    table, so rows are split on their ``tr_<id>`` markers and the modal is
    removed before reading cells."""
    page = re.sub(r"<!--.*?-->", " ", page, flags=re.S)
    segments = re.split(r'<tr\s+id="tr_\d+"[^>]*>', page)[1:]
    rows: list[dict] = []
    for seg in segments[:limit]:
        head = re.split(r'<div\s+class="modal', seg, maxsplit=1)[0]
        cells = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</t[dh]>", head, re.S | re.I)]
        if len(cells) < 5:
            continue
        charge = re.search(r"fa-inr[^>]*>\s*</i>\s*([\d,]+)", seg)
        tail = seg[seg.rfind("</div>"):] if "</div>" in seg else seg
        valid = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]{2},?\s+\d{4})\b", tail)
        rows.append({
            "lab": cells[1],
            "osl": cells[2] if cells[2].lower() not in ("none", "-") else "",
            "is_number": cells[3],
            "product": cells[4],
            "scope": cells[5] if len(cells) > 5 and cells[5] not in ("-", "") else "",
            "charges_inr": charge.group(1) if charge else "",
            "valid_until": valid.group(1) if valid else "",
        })
    return rows


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _lab_states() -> dict[str, str]:
    data = _load_json(LABS_PATH)
    return {lab.get("osl", ""): lab.get("state", "") for lab in data.get("labs", [])
            if lab.get("osl")}


def _cache_key(base: str, part: str = "") -> str:
    return f"IS {base}" + (f" (Part {part})" if part else "")


def lookup(base: str, part: str = "", live: bool | None = None,
           timeout_s: float = 8.0, use_cache: bool = True) -> list[dict] | None:
    """Labs testing IS <base> (Part <part>). None when unknown and offline."""
    key = _cache_key(base, part)
    if use_cache:
        with _lock:
            if key in _memory:
                return _memory[key]["rows"]
        cached = _load_json(CACHE_PATH).get("standards", {}).get(key)
        if cached is not None:
            return cached.get("rows", [])
    if live is None:
        live = os.environ.get("BIS_LIMS_LIVE", "1") != "0"
    if not live:
        return None
    try:
        req = urllib.request.Request(lims_url(base, part),
                                     headers={"User-Agent": "BIS-Assistant/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            rows = parse_lims(r.read().decode("utf-8", "replace"))
    except Exception as exc:  # network trouble must never break chat
        log.warning("LIMS lookup failed", extra={"ctx": {"reason": type(exc).__name__}})
        return None
    with _lock:
        _memory[key] = {"rows": rows, "at": time.time()}
    return rows


def product_query(query: str) -> str:
    """The query without lab words and place names: what the product is."""
    text = _LAB_WORDS.sub(" ", query or "")
    for place in list(_CITY_STATE) + [st.lower() for st in _STATE_NAMES]:
        text = re.sub(rf"\b{re.escape(place)}\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _product_terms(query: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", product_query(query).lower())}


def standards_for_query(query: str, evidence: list[dict], limit: int = 2) -> list[tuple[str, str, str]]:
    """Return up to ``limit`` (base, part, label) IS numbers for a lab query.

    Order: IS numbers named in the query, then the compulsory-list row that
    best matches the product words, then catalogue records.
    """
    found: list[tuple[str, str, str]] = []

    def add(base: str, part: str, label: str) -> None:
        if base and all(base != b or part != p for b, p, _ in found):
            found.append((base, part or "", label))

    for m in _IS_RE.finditer(query or ""):
        add(m.group(1), m.group(2) or "", "")
    if found:
        return found[:limit]
    terms = _product_terms(query)
    if not terms:
        return []
    best: list[tuple[int, str, str, str, str]] = []
    for item in evidence:
        if item.get("doc_type") != "compulsory_list":
            continue
        for line in str(item.get("chunk_text", "")).splitlines():
            if not line.startswith("- "):
                continue
            words = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", line.lower())}
            score = len(terms & words)
            m = _IS_RE.search(line)
            if score and m:
                label = line[2:].split(". Compulsory")[0].split(": ", 1)[-1][:90]
                best.append((score, m.group(1), m.group(2) or "", label,
                             str(item.get("heading", ""))))
    if best:
        top = max(b[0] for b in best)
        leaders = [b for b in best if b[0] == top]
        categories = {b[4] for b in leaders}
        if len(categories) > 1:
            # Ambiguous product word ("helmets": riders, industrial, police):
            # one standard per category so the answer can show each.
            seen: set[str] = set()
            for _, base, part, label, heading in leaders:
                if heading not in seen:
                    seen.add(heading)
                    add(base, part, label)
                if len(found) >= 3:
                    return found
            return found
        for _, base, part, label, _ in leaders:
            add(base, part, label)
            if len(found) >= limit:
                return found
    for item in evidence:
        if item.get("evidence_type") != "catalogue_record":
            continue
        words = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(item.get("title", "")).lower())}
        m = _IS_RE.search(str(item.get("standard_number", "")))
        if m and terms & words:
            add(m.group(1), m.group(2) or "", str(item.get("title", ""))[:90])
        if len(found) >= limit:
            break
    return found[:limit]


def _with_year(designation: str, labs: list[dict]) -> str:
    """Add the edition most labs are recognised for: LIMS writes "IS 4151 (2015)"."""
    years: dict[str, int] = {}
    for lab in labs:
        m = re.search(r"\((\d{4})(?:\.0)?\)", lab.get("is_number", ""))
        if m:
            years[m.group(1)] = years.get(m.group(1), 0) + 1
    if not years:
        return designation
    return f"{designation}:{max(years, key=years.get)}"


def _place(query: str) -> tuple[str, str]:
    """(city, state) named in the query, lower-case city; '' when absent."""
    low = (query or "").lower()
    for city, state in _CITY_STATE.items():
        if re.search(rf"\b{city}\b", low):
            return city, state
    for state in _STATE_NAMES:
        if state.lower() in low:
            return "", state
    return "", ""


def lab_evidence(query: str, evidence: list[dict]) -> list[dict]:
    """Evidence rows listing labs for the product in a lab question."""
    if not is_lab_query(query):
        return []
    states = _lab_states()
    city, wanted_state = _place(query)
    rows: list[dict] = []
    for base, part, label in standards_for_query(query, evidence):
        labs = lookup(base, part)
        if labs is None:
            continue
        designation = _with_year(_cache_key(base, part), labs)
        if not labs:
            text = (f"BIS LIMS lists no recognised laboratory for {designation} "
                    "at the time of lookup.")
        else:
            def nearness(lab: dict) -> int:
                name = lab.get("lab", "").lower()
                if city and city in name:
                    return 0
                if wanted_state and states.get(lab.get("osl", "")) == wanted_state:
                    return 1
                return 2
            labs = sorted(labs, key=nearness)
            lines = [f"Laboratories that test {designation}"
                     + (f" ({label})" if label else "") + ", from BIS LIMS:"]
            if wanted_state:
                near = sum(1 for lab in labs if nearness(lab) < 2)
                lines.append(f"Labs in or near {city.title() or wanted_state}: {near} listed first."
                             if near else
                             f"No lab for this standard is listed in {wanted_state}; "
                             "the labs below are elsewhere in India.")
            for lab in labs[:12]:
                state = states.get(lab.get("osl", ""), "")
                bits = [lab["lab"] + (f", {state}" if state else "")]
                if lab.get("scope"):
                    bits.append(f"scope: {lab['scope'][:80]}")
                if lab.get("charges_inr"):
                    bits.append(f"testing charges Rs. {lab['charges_inr']} excluding taxes")
                if lab.get("valid_until"):
                    bits.append(f"recognition valid until {lab['valid_until']}")
                lines.append("- " + "; ".join(bits))
            if len(labs) >= LIMS_PAGE_ROWS:
                lines.append("- The LIMS portal lists more laboratories for this standard.")
            elif len(labs) > 12:
                lines.append(f"- and {len(labs) - 12} more on the LIMS portal.")
            text = "\n".join(lines)
        rows.append({
            "evidence_type": "document_chunk",
            "doc_type": "lab_directory",
            "standard_number": designation,
            "title": f"BIS LIMS: laboratories testing {designation}",
            "heading": label or designation,
            "chunk_text": text,
            "source_url": lims_url(base, part),
            "source_file": "",
            "chunk_id": f"lims:{designation}",
            "relevance": 1.0,
            "rank_score": 2.0,
            "exact_match": True,
            "selection_reason": "lab_lookup",
        })
    return rows
