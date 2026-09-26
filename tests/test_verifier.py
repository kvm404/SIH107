"""Grounding verifier: designations need a cited source; citations renumber."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant import verifier  # noqa: E402

LIST_ROW = {
    "evidence_type": "document_chunk", "doc_type": "compulsory_list", "standard_number": "",
    "chunk_text": "- IS 17526 : 2021: Domestic Stainless steel vacuum flask/bottle. "
                  "Compulsory under Insulated Flask (Quality Control) Order, 2023.",
}
CATALOGUE = {"evidence_type": "catalogue_record", "standard_number": "IS 17526:2021",
             "title": "DOMESTIC STAINLESS STEEL VACUUM FLASK/BOTTLE - SPECIFICATION"}
GUIDE = {"evidence_type": "document_chunk", "doc_type": "bis_guide", "standard_number": "",
         "chunk_text": "The application fee is Rs. 1000."}


def test_english_is_before_a_number_is_not_a_standard():
    text = "The application fee is 1000 rupees [Source 3]. Capacity is 1 litre."
    assert verifier.verify_grounded_response(text, [LIST_ROW, CATALOGUE, GUIDE]) == []


def test_designation_supported_by_excerpt_that_names_it():
    text = "Steel vacuum flasks follow IS 17526:2021 [Source 1]."
    assert verifier.verify_grounded_response(text, [LIST_ROW]) == []


def test_designation_needs_marker_and_real_support():
    ev = [LIST_ROW, CATALOGUE, GUIDE]
    assert "standard_without_source_marker" in verifier.verify_grounded_response(
        "Use IS 17526:2021.", ev)
    assert "unsupported_standard_designation" in verifier.verify_grounded_response(
        "Use IS 9999 [Source 1].", ev)
    assert "designation_source_mismatch" in verifier.verify_grounded_response(
        "Use IS 17526:2021 [Source 3].", ev)
    assert "invalid_source_marker" in verifier.verify_grounded_response(
        "Fee [Source 7].", ev)


def test_catalogue_alone_cannot_carry_legal_claims():
    ev = [LIST_ROW, CATALOGUE]
    assert "unsupported_catalogue_claim" in verifier.verify_grounded_response(
        "IS 17526:2021 is mandatory for flasks [Source 2].", ev)
    # Naming the standard and its product from the title is fine.
    assert verifier.verify_grounded_response(
        "IS 17526:2021 is the specification for stainless steel vacuum flasks [Source 2].", ev) == []
    # The compulsory list cited alongside supports the legal claim.
    assert verifier.verify_grounded_response(
        "IS 17526:2021 is mandatory for flasks [Source 1, 2].", ev) == []


def test_long_answer_without_markers_is_flagged():
    text = " ".join(["BIS certification involves several steps."] * 15)
    assert "missing_source_markers" in verifier.verify_grounded_response(text, [GUIDE])
    assert verifier.verify_grounded_response("Which product do you make?", [GUIDE]) == []


def test_finalize_citations_renumbers_to_cited_sources():
    text, cited = verifier.finalize_citations(
        "A [Source 3]. B [Source 1][Source 3]. C [Sources 1 and 2].", [LIST_ROW, CATALOGUE, GUIDE])
    assert text == "A[1]. B[2][1]. C[2][3]."
    assert cited == [GUIDE, LIST_ROW, CATALOGUE]
    assert verifier.strip_markers(text) == "A. B. C."


def test_loose_source_spelling_and_repeat_mentions_are_accepted():
    catalogue = {"evidence_type": "catalogue_record", "standard_number": "Is 2347:2023",
                 "title": "DOMESTIC PRESSURE COOKER - SPECIFICATION"}
    assert verifier.verify_grounded_response(
        "A catalogue record for IS 2347:2023 exists [Source 1].", [catalogue]) == []
    text = ("IS 17526:2021 covers steel vacuum flasks [Source 1]. "
            "The evidence does not say IS 17526 covers glass flasks.")
    assert verifier.verify_grounded_response(text, [LIST_ROW]) == []
    # A standard never cited anywhere still needs its marker.
    assert "standard_without_source_marker" in verifier.verify_grounded_response(
        "The evidence mentions IS 17526.", [LIST_ROW])


def test_full_width_markers_are_normalised():
    assert verifier.normalize_markers("Fee is Rs. 1000【Source 1】.") == "Fee is Rs. 1000[Source 1]."
    assert verifier.marker_indices(verifier.normalize_markers("x 【Source 2】")) == [2]


def test_user_typed_designation_may_be_restated_without_marker():
    rows = [{"standard_number": "IS 14543:2024", "title": "Packaged drinking water",
             "chunk_text": "IS 14543 Packaged drinking water", "doc_type": "catalogue"}]
    text = ("Packaged drinking water is covered by IS 14543 [Source 1]. "
            "IS 10500 alone does not cover packaged water.")
    assert verifier.verify_grounded_response(text, rows) == [
        "unsupported_standard_designation"]
    assert verifier.verify_grounded_response(text, rows, "Is is 10500 enough?") == []
    # A cited restatement must still match its source.
    cited = "IS 10500 applies [Source 1]."
    assert verifier.verify_grounded_response(cited, rows, "IS 10500?") == [
        "unsupported_standard_designation"]
