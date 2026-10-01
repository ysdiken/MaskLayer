"""
Unit tests for span_merger.py and pipeline._apply_masks().

No ML model required — NER spans are constructed manually so tests run
fast and offline.

Coverage:
  - Regex-dominant labels always win over NER on overlap.
  - NER spans win for non-deterministic types when they are longer.
  - Containment: outer span wins.
  - Coreference: same canonical_id assigned to identical texts.
  - _apply_masks: left-to-right numbering, coreference reuse, RTL replacement.
  - Ablation helpers: regex_only, ner_only.
"""

import pytest

from app.masking.span_merger import merge, regex_only, ner_only, _assign_canonical_ids
from app.masking.pipeline import _apply_masks
from app.models.schemas import Span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rspan(start, end, label, confidence=0.99, text=None):
    return Span(
        start=start, end=end, label=label, source="regex",
        confidence=confidence, text=text or "x" * (end - start),
    )

def nspan(start, end, label, confidence=0.85, text=None):
    return Span(
        start=start, end=end, label=label, source="ner",
        confidence=confidence, text=text or "x" * (end - start),
    )


# ---------------------------------------------------------------------------
# merge() — overlap resolution
# ---------------------------------------------------------------------------

class TestMergeOverlapResolution:

    def test_no_spans_returns_empty(self):
        assert merge([], []) == []

    def test_non_overlapping_spans_all_kept(self):
        r = [rspan(0, 5, "TC_No")]
        n = [nspan(10, 15, "Person")]
        result = merge(r, n)
        assert len(result) == 2

    def test_regex_dominant_beats_ner_on_overlap(self):
        # TC_No (regex-dominant) overlaps with a NER Person span
        r = [rspan(0, 11, "TC_No", confidence=0.99)]
        n = [nspan(0, 15, "Person", confidence=0.85)]   # longer, but NER
        result = merge(r, n)
        assert len(result) == 1
        assert result[0].label == "TC_No"
        assert result[0].source == "regex"

    def test_regex_dominant_beats_ner_even_when_shorter(self):
        r = [rspan(2, 8, "IBAN", confidence=0.99)]
        n = [nspan(0, 10, "Person", confidence=0.99)]   # longer NER
        result = merge(r, n)
        assert len(result) == 1
        assert result[0].label == "IBAN"

    def test_longer_ner_beats_shorter_ner_on_overlap(self):
        # Two NER spans overlap: longer one wins regardless of confidence
        n1 = nspan(5, 15, "Person",  confidence=0.90)   # shorter, higher conf
        n2 = nspan(0, 20, "Company", confidence=0.85)   # longer, lower conf
        result = merge([], [n1, n2])
        assert len(result) == 1
        assert result[0].label == "Company"

    def test_containment_outer_wins(self):
        r = [rspan(5, 10, "TC_No", confidence=0.99)]
        n = [nspan(0, 15, "Person", confidence=0.85)]    # contains r
        # TC_No is dominant, wins despite being contained
        result = merge(r, n)
        assert len(result) == 1
        assert result[0].label == "TC_No"

    def test_two_ner_spans_overlap_longer_wins(self):
        n1 = nspan(0, 10, "Person", confidence=0.85)
        n2 = nspan(5, 20, "Company", confidence=0.85)
        result = merge([], [n1, n2])
        assert len(result) == 1
        assert result[0].label == "Company"   # longer (15 chars vs 10)

    def test_equal_length_higher_confidence_wins(self):
        n1 = nspan(0, 10, "Person",  confidence=0.70)
        n2 = nspan(0, 10, "Company", confidence=0.90)
        result = merge([], [n1, n2])
        assert len(result) == 1
        assert result[0].label == "Company"

    def test_result_sorted_by_start(self):
        r = [rspan(20, 30, "Email"), rspan(0, 10, "TC_No")]
        result = merge(r, [])
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_adjacent_spans_both_kept(self):
        # Adjacent (not overlapping) spans must both survive
        r = [rspan(0, 5, "TC_No"), rspan(5, 10, "Email")]
        result = merge(r, [])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Coreference (_assign_canonical_ids)
# ---------------------------------------------------------------------------

class TestCoreference:

    def test_identical_text_same_label_gets_same_canonical_id(self):
        spans = [
            rspan(0, 10, "Person", text="Ahmet Yılmaz"),
            rspan(50, 62, "Person", text="Ahmet Yılmaz"),
        ]
        result = _assign_canonical_ids(spans)
        assert result[0].canonical_id == result[1].canonical_id

    def test_case_insensitive_coreference(self):
        spans = [
            rspan(0, 12, "Person", text="Ahmet Yılmaz"),
            rspan(50, 62, "Person", text="ahmet yılmaz"),
        ]
        result = _assign_canonical_ids(spans)
        assert result[0].canonical_id == result[1].canonical_id

    def test_different_labels_no_coreference(self):
        spans = [
            rspan(0, 5, "Person",  text="Ahmet"),
            rspan(20, 25, "Company", text="Ahmet"),
        ]
        result = _assign_canonical_ids(spans)
        assert result[0].canonical_id != result[1].canonical_id

    def test_different_texts_no_coreference(self):
        spans = [
            rspan(0, 12, "Person", text="Ahmet Yılmaz"),
            rspan(50, 59, "Person", text="Ayşe Kaya"),
        ]
        result = _assign_canonical_ids(spans)
        assert result[0].canonical_id != result[1].canonical_id

    def test_all_spans_get_canonical_id(self):
        spans = [rspan(0, 5, "TC_No", text="12345"), rspan(10, 15, "Email", text="a@b.c")]
        result = _assign_canonical_ids(spans)
        assert all(s.canonical_id is not None for s in result)


# ---------------------------------------------------------------------------
# _apply_masks — placeholder numbering and coreference
# ---------------------------------------------------------------------------

class TestApplyMasks:
    TEXT = "Ahmet Yılmaz aradı. Ahmet Yılmaz tekrar aradı."
    # Positions (approximate — we'll use exact spans):
    # "Ahmet Yılmaz" at 0..13 and 20..33

    def test_single_span_replaced(self):
        spans = [nspan(0, 5, "Person", text="Ahmet")]
        masked, mapping = _apply_masks("Ahmet geldi.", spans)
        assert "{Person_1}" in masked
        assert "Ahmet" not in masked
        assert mapping["{Person_1}"] == "Ahmet"

    def test_two_different_entities_numbered_left_to_right(self):
        spans = [
            nspan(0, 5, "Person",  text="Ahmet"),
            nspan(10, 15, "Person", text="Mehmet"),
        ]
        masked, mapping = _apply_masks("Ahmet ve  Mehmet geldi.", spans)
        # Left-to-right: Ahmet=1, Mehmet=2
        assert "{Person_1}" in masked
        assert "{Person_2}" in masked
        assert mapping["{Person_1}"] == "Ahmet"
        assert mapping["{Person_2}"] == "Mehmet"

    def test_coreferent_spans_get_same_number(self):
        # Two spans with same canonical_id → same placeholder
        s1 = nspan(0, 12, "Person", text="Ahmet Yılmaz")
        s2 = nspan(20, 32, "Person", text="Ahmet Yılmaz")
        # Assign canonical_ids manually (merger would do this)
        s1 = s1.model_copy(update={"canonical_id": "Person::ahmet yılmaz"})
        s2 = s2.model_copy(update={"canonical_id": "Person::ahmet yılmaz"})

        text = "Ahmet Yılmaz aradı. Ahmet Yılmaz tekrar aradı."
        masked, mapping = _apply_masks(text, [s1, s2])

        # Both occurrences replaced with the SAME placeholder
        assert masked.count("{Person_1}") == 2
        assert "{Person_2}" not in masked
        assert mapping.get("{Person_1}") == "Ahmet Yılmaz"

    def test_empty_spans_returns_original(self):
        text = "Hiçbir kişisel veri yok."
        masked, mapping = _apply_masks(text, [])
        assert masked == text
        assert mapping == {}

    def test_rtl_replacement_preserves_offsets(self):
        # Two spans at known offsets; make sure RTL substitution doesn't shift them
        text = "TC: 10000000146 IBAN: TR330006100519786457841326"
        tc_span   = rspan(4, 15, "TC_No",   text="10000000146")
        iban_span = rspan(22, 48, "IBAN",    text="TR330006100519786457841326")
        masked, mapping = _apply_masks(text, [tc_span, iban_span])
        assert "{TC_No_1}" in masked
        assert "{IBAN_1}" in masked
        assert "10000000146" not in masked
        assert "TR330006100519786457841326" not in masked

    def test_different_label_types_independent_counters(self):
        spans = [
            nspan(0, 5,   "Person",  text="Ahmet"),
            rspan(10, 22, "IBAN",    text="TR123456789012345678901234"),
            nspan(30, 36, "Person",  text="Mehmet"),
        ]
        masked, mapping = _apply_masks(
            "Ahmet    TR123456789012345678901234    Mehmet", spans
        )
        assert "{Person_1}" in masked   # Ahmet
        assert "{IBAN_1}" in masked
        assert "{Person_2}" in masked   # Mehmet
        # No cross-contamination between Person and IBAN counters
        assert "{Person_3}" not in masked


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

class TestAblationHelpers:

    def test_regex_only_returns_only_regex_spans(self):
        r = [rspan(0, 5, "TC_No")]
        n = [nspan(10, 15, "Person")]
        result = regex_only(r)
        assert all(s.source == "regex" for s in result)
        assert len(result) == 1

    def test_ner_only_returns_only_ner_spans(self):
        r = [rspan(0, 5, "TC_No")]
        n = [nspan(10, 15, "Person")]
        result = ner_only(n)
        assert all(s.source == "ner" for s in result)
        assert len(result) == 1
