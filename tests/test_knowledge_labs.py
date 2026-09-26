"""BIS knowledge pages, lab lookup, query rewriting and model fallback."""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bis_assistant import knowledge, labs, rag_llm  # noqa: E402
from bis_assistant.rag_ingest import import_knowledge  # noqa: E402
from bis_assistant.rag_store import connect_rag  # noqa: E402

PAGE = """---
title: Products under compulsory BIS certification: Scheme-I (ISI Mark)
source_url: https://www.bis.gov.in/scheme-i/
doc_type: compulsory_list
retrieved: 2026-09-26
---

# Products under compulsory BIS certification

## Domestic Pressure Cooker
<!-- source: https://www.bis.gov.in/scheme-i/#cooker -->
Quality Control Order details: Domestic Pressure Cooker (Quality Control) Order, 2020
- IS 2347:2017: Domestic Pressure Cooker. Compulsory under Domestic Pressure Cooker (Quality Control) Order, 2020.

## Helmet for riders of Two Wheeler Motor Vehicles
- IS 4151: 2015: Helmet for riders of Two Wheeler Motor Vehicles. Compulsory under Helmet (Quality Control) Order, 2020.
"""


def test_knowledge_file_sections_keep_their_source_url(tmp_path):
    path = tmp_path / "list.md"
    path.write_text(PAGE, encoding="utf-8")
    doc = knowledge.parse_knowledge_file(path)
    assert doc["doc_type"] == "compulsory_list" and doc["retrieved"] == "2026-09-26"
    assert [s["heading"] for s in doc["sections"]] == [
        "Domestic Pressure Cooker", "Helmet for riders of Two Wheeler Motor Vehicles"]
    assert doc["sections"][0]["source_url"].endswith("#cooker")
    # A section without its own marker falls back to the page URL.
    assert doc["sections"][1]["source_url"] == "https://www.bis.gov.in/scheme-i/"


def test_long_sections_split_on_lines_and_repeat_context():
    lines = ["Quality Control Order details: Steel (Quality Control) Order, 2020"]
    lines += [f"- IS {1000 + i}: Steel product number {i} with a long description here."
              for i in range(80)]
    chunks = knowledge.chunk_sections([{"heading": "Steel", "source_url": "u", "lines": lines}],
                                      max_words=120)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk["heading"] == "Steel"
        assert chunk["chunk_text"].startswith("Quality Control Order details")
        # Every product row stays whole.
        assert all(line.startswith(("- IS", "Quality")) for line in chunk["chunk_text"].splitlines())


def test_import_knowledge_is_idempotent_and_drops_removed_pages(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "list.md").write_text(PAGE, encoding="utf-8")
    (root / "old.md").write_text(PAGE.replace("compulsory_list", "bis_guide"), encoding="utf-8")
    conn = connect_rag(tmp_path / "rag.db")
    try:
        first = import_knowledge(conn, root)
        assert first == {"knowledge_documents": 2, "knowledge_chunks": 4}
        (root / "old.md").unlink()
        second = import_knowledge(conn, root)
        assert second["knowledge_documents"] == 1
        docs = [r["source_file"] for r in conn.execute("SELECT source_file FROM corpus_documents")]
        assert docs == ["knowledge/list.md"]
        row = conn.execute("SELECT doc_type, source_url FROM corpus_chunks ORDER BY id LIMIT 1").fetchone()
        assert row["doc_type"] == "compulsory_list" and row["source_url"].endswith("#cooker")
    finally:
        conn.close()


def test_committed_knowledge_pages_parse_and_cite_official_sites():
    root = ROOT / "data" / "knowledge"
    files = knowledge.knowledge_files(root)
    assert len(files) >= 10
    for path in files:
        doc = knowledge.parse_knowledge_file(path)
        assert doc["sections"], path
        for section in doc["sections"]:
            host = section["source_url"].split("/")[2]
            assert host.endswith(("bis.gov.in", "crsbis.in")), (path, section["source_url"])


# --- LIMS labs -------------------------------------------------------------------

LIMS_HTML = """
<table><tr><th>S.No.</th><th>Lab Name</th></tr>
<tr id="tr_1">
  <td>1</td><td>PRESTO LABORATORIES PRIVATE LIMITED</th><td>8199006</th>
  <td>IS 4151 (2015)</td><td>Protective helmets for motorcycle riders</td><td>All</td>
  <!-- <td>commented</td> -->
  <td><i class="fa fa-inr" aria-hidden="true"></i>14000 <a href="#">View breakup</a>
    <div class="modal fade"><div class="modal-body"><table><tr><td>4.1</td><td>01 Jan, 2024</td></tr></table></div></div>
  </td><td>21 Aug, 2029</td><td></td>
</tr>
<tr id="tr_2">
  <td>2</td><td>BIS, Central Laboratory (CL)</th><td>None</th>
  <td>IS 4151 (2015)</td><td>Protective helmets for motorcycle riders</td><td>-</td>
  <td></td><td>-</td><td></td>
</tr>
</table>
"""


def test_parse_lims_skips_charge_modals():
    rows = labs.parse_lims(LIMS_HTML)
    assert [r["lab"] for r in rows] == ["PRESTO LABORATORIES PRIVATE LIMITED",
                                        "BIS, Central Laboratory (CL)"]
    assert rows[0]["osl"] == "8199006" and rows[0]["charges_inr"] == "14000"
    assert rows[0]["valid_until"] == "21 Aug, 2029"
    assert rows[1]["osl"] == "" and rows[1]["charges_inr"] == ""


def test_lab_query_detection():
    assert labs.is_lab_query("Suggest a BIS recognised lab in Pune to test helmets")
    assert labs.is_lab_query("where can I get my LED lamps tested?")
    assert labs.is_lab_query("हेलमेट की जांच के लिए प्रयोगशाला")
    assert not labs.is_lab_query("Is ISI mandatory for pressure cookers?")
    assert labs.product_query("Suggest a lab in Pune to test helmets") == "helmets"


def _list_row(heading: str, line: str) -> dict:
    return {"doc_type": "compulsory_list", "heading": heading, "chunk_text": line}


def test_standards_for_query_prefers_named_is_then_product_rows():
    assert labs.standards_for_query("labs for IS 16102 (Part 1)", []) == [("16102", "1", "")]
    evidence = [
        _list_row("Domestic Pressure Cooker",
                  "- IS 2347:2017: Domestic Pressure Cooker. Compulsory under X Order, 2020."),
        _list_row("Helmet for riders", "- IS 4151: 2015: Helmet for riders. Compulsory under Y Order, 2020."),
    ]
    found = labs.standards_for_query("lab to test pressure cookers", evidence)
    assert found[0][:2] == ("2347", "")


def test_ambiguous_product_word_returns_one_standard_per_category():
    evidence = [
        _list_row("Helmet for riders", "- IS 4151: 2015: Helmet for riders. Compulsory under Y Order, 2020."),
        _list_row("Helmet for Police", "- IS 2925:1984: Industrial safety helmets. Compulsory under Z Order, 2023.\n"
                  "- IS 9562:1980: Non- metal helmet for Police Force. Compulsory under Z Order, 2023."),
    ]
    found = labs.standards_for_query("lab for helmets", evidence)
    assert [f[0] for f in found] == ["4151", "2925"]


def test_lab_evidence_lists_nearby_labs_first(monkeypatch):
    rows = [
        {"lab": "Delhi Test House", "osl": "1", "scope": "", "charges_inr": "100",
         "valid_until": "", "is_number": "IS 4151 (2015)"},
        {"lab": "Pune Labs LLP, Pune", "osl": "2", "scope": "", "charges_inr": "",
         "valid_until": "", "is_number": "IS 4151 (1993.0)"},
        {"lab": "Noida Lab", "osl": "3", "scope": "", "charges_inr": "",
         "valid_until": "", "is_number": "IS 4151 (2015)"},
    ]
    monkeypatch.setattr(labs, "lookup", lambda base, part="": rows)
    monkeypatch.setattr(labs, "_lab_states", lambda: {"1": "Delhi", "2": "Maharashtra"})
    out = labs.lab_evidence("lab in Pune to test IS 4151 helmets", [])
    assert len(out) == 1 and out[0]["doc_type"] == "lab_directory"
    text = out[0]["chunk_text"]
    assert text.index("Pune Labs LLP") < text.index("Delhi Test House")
    assert "Rs. 100" in text and out[0]["source_url"].startswith(labs.LIMS_SEARCH)
    # The edition most labs test to becomes the designation, so "IS 4151:2015" verifies.
    assert out[0]["standard_number"] == "IS 4151:2015"
    assert labs.lab_evidence("Is ISI mandatory for cookers?", []) == []


def test_committed_lims_cache_has_labs_for_demo_standards():
    cache = json.loads((ROOT / "data" / "lims_labs.json").read_text())
    assert cache["standards"]["IS 4151"]["rows"]
    assert cache["standards"]["IS 2347"]["rows"]


# --- query rewriting + model fallback ----------------------------------------------

def test_rewrite_only_for_hindi_or_follow_ups():
    assert rag_llm.needs_rewrite("नल के पानी का मानक?", "hi", [])
    assert rag_llm.needs_rewrite("which labs can test it?", "en", ["User: pressure cooker"])
    assert not rag_llm.needs_rewrite("Is ISI mandatory for pressure cookers?", "en", [])
    assert rag_llm.needs_rewrite("Helmet ke liye kaunsa manak hai?", "en", [])
    assert not rag_llm.needs_rewrite("What is the fee and how long does it take?", "en", [])
    assert not rag_llm.needs_rewrite(
        "Please explain the full licence procedure for a cement plant in Gujarat", "en",
        ["User: hello"])


def _cfg(**extra) -> dict:
    return {"provider": "openai-compatible", "model": "main", "api_key": "k",
            "base_url": "https://llm.example.test/v1", "retries": 0, "timeout_s": 1,
            **extra}


def test_rate_limited_primary_falls_back_to_second_model(monkeypatch):
    seen = []

    def sender(messages, cfg):
        seen.append(cfg["model"])
        if cfg["model"] == "main":
            raise urllib.error.HTTPError("u", 429, "busy", {"retry-after": "30"}, None)
        return "fallback answer"

    monkeypatch.setitem(rag_llm._SENDERS, "openai-compatible", sender)
    out = rag_llm.chat_complete([{"role": "user", "content": "q"}], _cfg(fallback_model="backup"))
    assert out == "fallback answer" and seen == ["main", "backup"]


def test_failure_reason_is_reported_when_all_models_fail(monkeypatch):
    def sender(messages, cfg):
        raise urllib.error.HTTPError("u", 429, "busy", {"retry-after": "30"}, None)

    monkeypatch.setitem(rag_llm._SENDERS, "openai-compatible", sender)
    assert rag_llm.chat_complete([{"role": "user", "content": "q"}], _cfg()) is None
    assert rag_llm.last_failure() == "http_429"


def test_utility_config_prefers_small_model():
    cfg = rag_llm.utility_config(_cfg(utility_model="small", fallback_model="backup"))
    assert cfg["model"] == "small" and cfg["max_tokens"] == 80 and cfg["fallback_model"] == "main"
    assert rag_llm.utility_config(_cfg())["model"] == "main"


def test_groq_gpt_oss_uses_low_reasoning(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(rag_llm, "_post_json", fake_post)
    cfg = _cfg(model="openai/gpt-oss-20b", base_url="https://api.groq.com/openai/v1")
    assert rag_llm._send_openai_compatible([{"role": "user", "content": "q"}], cfg) == "ok"
    assert seen["reasoning_effort"] == "low" and seen["include_reasoning"] is False


# --- scheme process pairing -----------------------------------------------------------

def test_compulsory_hit_brings_its_scheme_process(tmp_path, monkeypatch):
    from bis_assistant import assistant
    db = tmp_path / "rag.db"
    conn = connect_rag(db)
    try:
        root = tmp_path / "knowledge"
        (root / "generated").mkdir(parents=True)
        (root / "curated").mkdir()
        (root / "generated" / "crs.md").write_text(
            "---\ntitle: CRS\nsource_url: https://www.crsbis.in/BIS/about-crs.do\n---\n"
            "## CRS registration steps\n- Get product tested from BIS recognized lab\n",
            encoding="utf-8")
        import_knowledge(conn, root)
    finally:
        conn.close()
    monkeypatch.setenv("BIS_RAG_DB_PATH", str(db))
    hit = {"doc_type": "compulsory_list",
           "source_file": "knowledge/generated/compulsory-scheme-ii-crs.md"}
    extra = assistant._scheme_process_evidence([hit])
    assert len(extra) == 1 and extra[0]["heading"] == "CRS registration steps"
    assert extra[0]["source_url"].startswith("https://www.crsbis.in")
    assert assistant._scheme_process_evidence([{"doc_type": "bis_guide"}]) == []
    # A "how do I register under CRS" question gets the steps without a list hit.
    asked = assistant._scheme_process_evidence(
        [{"doc_type": "bis_guide"}], "How to apply on the CRS portal for LED lamps?")
    assert asked and asked[0]["heading"] == "CRS registration steps"
    assert assistant._scheme_process_evidence(asked + [hit]) == []
