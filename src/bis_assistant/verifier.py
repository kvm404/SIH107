"""Grounding checks for model answers against the evidence they cite.

The model cites evidence with ``[Source N]`` markers. ``verify_grounded_response``
checks the high-risk claim forms: every IS designation must be supported by a
source cited in the same sentence, clause numbers must appear in a cited
excerpt, and catalogue records (title-only metadata) cannot carry legal or
certification claims on their own. ``finalize_citations`` then renumbers the
markers to the sources actually cited, for display.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("bis.verifier")

# Case-sensitive IS so English "is 1 litre" is not Indian Standard 1.
IS_RE = re.compile(r"(?<![A-Za-z])IS\s*(\d+(?:-\d+)?)")
# "[Source 2]", "[Source 1, 3]", "[Sources 1 and 2]", "[Source 1][Source 2]".
SOURCE_RE = re.compile(r"\[\s*Sources?\s*(\d+(?:\s*(?:,|and|&)\s*(?:Source\s*)?\d+)*)\s*\]",
                       re.IGNORECASE)
# Case-sensitive "IS": the English word "is" followed by a number
# ("the fee is 1000") is not an Indian Standard.
STANDARD_DESIGNATION_RE = re.compile(
    r"(?<![A-Za-z])IS\s*[:\-]?\s*(?P<base>\d+(?:-\d+)?)"
    r"(?:\s*(?P<qualifier>\([^\n)]{1,80}\)))?"
    r"(?:\s*:\s*(?P<year>\d{4}))?",
)
CLAUSE_REF_RE = re.compile(
    r"\b(?:clause|section)\s*(?:no\.?\s*)?(\d+(?:\.\d+)*(?:\([a-z0-9]+\))?)",
    re.IGNORECASE,
)

# Claims a title-only catalogue record can never support by itself.
_UNSUPPORTED_CATALOGUE_CLAIM_RE = re.compile(
    r"\b(?:requires?|required|must|shall|mandatory|mandated|compulsory|QCO|"
    r"quality\s+control\s+order|approved|certified|compliance|compliant|"
    r"in\s+force|licen[cs]e\s+is\s+needed)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|cannot|can['’]t|does\s+not|doesn['’]t|do\s+not|"
    r"don['’]t|did\s+not|is\s+not|isn['’]t|are\s+not|aren['’]t|"
    r"without|unable\s+to|insufficient|unknown)\b",
    re.IGNORECASE,
)
_CONTRAST_BOUNDARY_RE = re.compile(
    r"\b(?:but|however|yet|nevertheless|instead)\b", re.IGNORECASE)
VIOLATION_COUNT = {"n": 0}  # grounding failures, surfaced via /metrics


def evidence_type(evidence: dict) -> str:
    """Return the stable evidence kind, failing safely for untyped records."""
    kind = str(evidence.get("evidence_type", "")).strip().lower()
    if kind in ("document_chunk", "catalogue_record"):
        return kind
    # Older retrieval rows have no explicit type. Non-empty body text is the
    # only safe signal that the row is a retrieved document passage.
    return "document_chunk" if str(evidence.get("chunk_text", "")).strip() else "catalogue_record"


_SOURCE_IS_RE = re.compile(r"\b[Ii][Ss](?=\s*[:\-]?\s*\d)")


def _norm_source(value: str) -> str:
    """Source data spells designations loosely ("Is 2347:2023"); model text
    is checked strictly, so upper-case the prefix in source strings only."""
    return _SOURCE_IS_RE.sub("IS", value or "")


def normalize_markers(text: str) -> str:
    """Accept marker variants models emit: 【Source 1】, [source 1], [Source1]."""
    text = re.sub(r"[【［]\s*(Sources?\s*[\d,\sand&]+?)\s*[】］]", r"[\1]", text or "")
    return text


def _standard_key(value: str) -> str:
    match = IS_RE.search(value or "")
    return match.group(1) if match else ""


def _designation_key(value: str) -> tuple[str, str, str, str]:
    match = STANDARD_DESIGNATION_RE.search(value or "")
    if not match:
        return "", "", "", ""
    qualifier = match.group("qualifier") or ""
    part = re.search(r"\bpart\s*(\d+)", qualifier, re.IGNORECASE)
    section = re.search(r"\bsec(?:tion)?\s*(\d+)", qualifier, re.IGNORECASE)
    return (
        match.group("base"),
        part.group(1) if part else "",
        section.group(1) if section else "",
        match.group("year") or "",
    )


def _designation_matches(mentioned: str, source: dict) -> bool:
    mentioned_key = _designation_key(mentioned)
    source_key = _designation_key(_norm_source(str(source.get("standard_number", ""))))
    if not mentioned_key[0] or mentioned_key[0] != source_key[0]:
        return False
    # When the answer names a part, section, or edition explicitly, it must be
    # present in the source's designation or passage. A conflicting designation
    # in the source metadata always wins over incidental cross-references.
    support_text = _norm_source(" ".join((
        str(source.get("chunk_text", "")),
        str(source.get("published_on", "")),
    )))
    support_keys = [
        _designation_key(match.group(0))
        for match in STANDARD_DESIGNATION_RE.finditer(support_text)
        if match.group("base") == mentioned_key[0]
    ]
    for index, mentioned_value in enumerate(mentioned_key[1:], 1):
        if not mentioned_value:
            continue
        source_value = source_key[index]
        if source_value:
            if source_value != mentioned_value:
                return False
        elif not any(candidate[index] == mentioned_value for candidate in support_keys):
            if index == 3 and str(source.get("published_on", "")).strip()[:4] == mentioned_value:
                continue
            return False
    return True


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    """Find a compact sentence span around a match for local citation checks."""
    left = 0
    for boundary in re.finditer(r"[.!?]+\s+|\n+", text):
        if boundary.end() <= position:
            left = boundary.end()
        elif boundary.start() >= position:
            return left, boundary.start()
    return left, len(text)


def _positive_unsupported_catalogue_claim(text: str) -> bool:
    for match in _UNSUPPORTED_CATALOGUE_CLAIM_RE.finditer(text):
        sentence_start, _ = _sentence_bounds(text, match.start())
        before = text[sentence_start:match.start()]
        # A negation in an earlier sentence or contrastive clause does not
        # negate this claim. Keep the local window to avoid distant scope.
        before = _CONTRAST_BOUNDARY_RE.split(before)[-1][-48:]
        if not _NEGATION_RE.search(before):
            return True
    return False


def marker_indices(text: str) -> list[int]:
    """1-based source numbers referenced by markers in ``text``, in order."""
    out: list[int] = []
    for marker in SOURCE_RE.finditer(text):
        out.extend(int(n) for n in re.findall(r"\d+", marker.group(1)))
    return out


def _cited_rows(sentence: str, rows: list[dict]) -> list[dict]:
    return [rows[i - 1] for i in marker_indices(sentence) if 1 <= i <= len(rows)]


def _row_designations(row: dict) -> set[str]:
    """IS base numbers a row supports: its designation plus, for document
    excerpts, every designation printed in the excerpt."""
    keys = {_standard_key(_norm_source(str(row.get("standard_number", ""))))}
    if evidence_type(row) == "document_chunk":
        keys.update(m.group("base") for m in
                    STANDARD_DESIGNATION_RE.finditer(_norm_source(str(row.get("chunk_text", "")))))
    keys.discard("")
    return keys


def _supports(mention: str, row: dict) -> bool:
    key = _designation_key(mention)
    if not key[0]:
        return False
    if _designation_matches(mention, row):
        return True
    if evidence_type(row) != "document_chunk":
        return False
    # The excerpt itself names the standard (guidance pages, product lists).
    for match in STANDARD_DESIGNATION_RE.finditer(_norm_source(str(row.get("chunk_text", "")))):
        if _designation_matches(mention, {"standard_number": match.group(0),
                                          "chunk_text": row.get("chunk_text", "")}):
            return True
    return False


def verify_grounded_response(text: str, evidence: list[dict],
                             query: str = "") -> list[str]:
    """Return stable issue codes for unsupported claims ([] when grounded).

    This validates citation linkage and high-risk claim forms, not semantic
    entailment. On failure the caller requests one bounded repair.
    Designations the user typed may be restated without a marker ("IS 10500
    alone is not enough"); a cited mention must still match its source.
    """
    violations: list[str] = []
    rows = list(evidence)
    known: set[str] = set()
    for row in rows:
        known |= _row_designations(row)

    if any(i < 1 or i > len(rows) for i in marker_indices(text)):
        violations.append("invalid_source_marker")

    # A designation needs a supporting citation where it is first grounded;
    # repeat mentions ("the evidence does not say IS 17803 covers flasks")
    # may omit the marker once the same standard (and part) is cited.
    unmarked: list[tuple[str, str]] = []
    grounded: set[tuple[str, str]] = set()
    asked = {_designation_key(m.group(0))[:2]
             for m in STANDARD_DESIGNATION_RE.finditer((query or "").upper())}
    for mention in STANDARD_DESIGNATION_RE.finditer(text):
        key = _designation_key(mention.group(0))[:2]
        start, end = _sentence_bounds(text, mention.start())
        cited = _cited_rows(text[start:end], rows)
        if not cited and key in asked:
            continue
        if mention.group("base") not in known:
            violations.append("unsupported_standard_designation")
            continue
        if not cited:
            unmarked.append(key)
        elif not any(_supports(mention.group(0), row) for row in cited):
            violations.append("designation_source_mismatch")
        else:
            grounded.add(key)
    if any(key not in grounded for key in unmarked):
        violations.append("standard_without_source_marker")

    for clause in CLAUSE_REF_RE.finditer(text):
        start, end = _sentence_bounds(text, clause.start())
        clause_text = clause.group(1).lower()
        supported = any(
            evidence_type(row) == "document_chunk"
            and re.search(rf"(?<!\d){re.escape(clause_text)}(?!\d)",
                          str(row.get("chunk_text", "")), re.IGNORECASE)
            for row in _cited_rows(text[start:end], rows))
        if not supported:
            violations.append("unsupported_clause_reference")

    checked: set[tuple[int, int]] = set()
    for marker in SOURCE_RE.finditer(text):
        span = _sentence_bounds(text, marker.start())
        if span in checked:
            continue
        checked.add(span)
        sentence = text[span[0]:span[1]]
        cited = _cited_rows(sentence, rows)
        if (cited and all(evidence_type(row) == "catalogue_record" for row in cited)
                and _positive_unsupported_catalogue_claim(sentence)):
            violations.append("unsupported_catalogue_claim")

    # A substantive answer built on evidence must cite it.
    if rows and not marker_indices(text) and len(text.split()) > 60:
        violations.append("missing_source_markers")

    if violations:
        VIOLATION_COUNT["n"] += 1
    # Stable, bounded codes only: never raw answer, query or evidence text.
    return list(dict.fromkeys(violations))


def finalize_citations(text: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Renumber markers to the cited sources and return (text, cited rows).

    ``[Source 3] ... [Source 1]`` becomes ``[1] ... [2]`` and the returned
    rows are evidence[2], evidence[0]. Out-of-range markers are dropped.
    """
    order: list[int] = []
    for i in marker_indices(text):
        if 1 <= i <= len(evidence) and i not in order:
            order.append(i)
    new_number = {old: new for new, old in enumerate(order, 1)}

    def replace(match: re.Match) -> str:
        nums = []
        for n in re.findall(r"\d+", match.group(1)):
            mapped = new_number.get(int(n))
            if mapped and mapped not in nums:
                nums.append(mapped)
        return "".join(f"[{n}]" for n in nums)

    out = SOURCE_RE.sub(replace, text)
    out = re.sub(r"[ \t]+(\[\d+\])", r"\1", out)   # "claim [1]" -> "claim[1]"
    out = re.sub(r"(\[\d+\])[ \t]+([.,;:])", r"\1\2", out)
    return out, [evidence[i - 1] for i in order]


def strip_markers(text: str) -> str:
    """Text without any citation markers (for history and titles)."""
    text = SOURCE_RE.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text)
