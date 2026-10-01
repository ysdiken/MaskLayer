"""
Tests for the evaluation harness: markup loader (offset correctness) and the
scorer (strict/relaxed/char-level metrics). No model needed — all offline.
"""

from dataclasses import dataclass

import pytest

from eval.dataset import parse_markup, load_dataset, GOLD_DIR
from eval.scorer import Report, score_doc


@dataclass
class S:
    """Minimal span stand-in for the scorer (matches the _SpanLike protocol)."""
    start: int
    end: int
    label: str


# ---------------------------------------------------------------------------
# Markup loader
# ---------------------------------------------------------------------------

class TestParseMarkup:

    def test_offsets_are_correct(self):
        text, spans = parse_markup("TC [[10000000146|TC_No]] sahibi [[Ahmet|Person]].")
        assert text == "TC 10000000146 sahibi Ahmet."
        assert len(spans) == 2
        tc, person = spans
        assert text[tc.start:tc.end] == "10000000146"
        assert tc.label == "TC_No"
        assert text[person.start:person.end] == "Ahmet"
        assert person.label == "Person"

    def test_clean_text_has_no_markup(self):
        text, _ = parse_markup("a [[b|Person]] c [[d e|Company]] f")
        assert "[[" not in text and "]]" not in text
        assert text == "a b c d e f"

    def test_surface_with_spaces_and_punctuation(self):
        text, spans = parse_markup("X [[Mavi Ada Teknoloji A.Ş.|Company]] Y")
        assert spans[0].text == "Mavi Ada Teknoloji A.Ş."
        assert text[spans[0].start:spans[0].end] == "Mavi Ada Teknoloji A.Ş."

    def test_no_markup_returns_plain_text(self):
        text, spans = parse_markup("hiçbir etiket yok")
        assert text == "hiçbir etiket yok"
        assert spans == []

    def test_unannotated_text_preserved_between_spans(self):
        # Statute names / aliases are intentionally left unannotated.
        text, spans = parse_markup(
            "[[Şirket A.Ş.|Company]] 6698 sayılı Kanun uyarınca işlem yapar."
        )
        assert "6698 sayılı Kanun" in text
        assert len(spans) == 1


# ---------------------------------------------------------------------------
# Gold corpus loads and is self-consistent
# ---------------------------------------------------------------------------

class TestGoldCorpus:

    def test_corpus_loads(self):
        docs = load_dataset()
        assert len(docs) >= 6, "expected the synthetic gold corpus to be present"

    def test_every_span_text_matches_offsets(self):
        for doc in load_dataset():
            for s in doc.spans:
                assert doc.text[s.start:s.end] == s.text, (
                    f"{doc.doc_id}: span text {s.text!r} != slice "
                    f"{doc.text[s.start:s.end]!r}"
                )

    def test_corpus_has_meaningful_span_count(self):
        total = sum(len(d.spans) for d in load_dataset())
        assert total >= 40


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class TestScorer:

    def test_perfect_match(self):
        gold = [S(0, 5, "Person"), S(10, 21, "TC_No")]
        pred = [S(0, 5, "Person"), S(10, 21, "TC_No")]
        r = Report(mode="t")
        score_doc(r, gold, pred)
        p, rc, f = r.micro(strict=True)
        assert (p, rc, f) == (1.0, 1.0, 1.0)
        assert r.leak_rate == 0.0

    def test_false_positive(self):
        gold = [S(0, 5, "Person")]
        pred = [S(0, 5, "Person"), S(10, 15, "Company")]  # extra
        r = Report()
        score_doc(r, gold, pred)
        p, rc, f = r.micro(strict=True)
        assert rc == 1.0          # recall unaffected
        assert p == 0.5           # one of two preds is wrong
        assert r.strict["Company"].fp == 1

    def test_false_negative_is_a_leak(self):
        gold = [S(0, 5, "Person"), S(10, 21, "TC_No")]
        pred = [S(0, 5, "Person")]  # missed the TC
        r = Report()
        score_doc(r, gold, pred)
        assert r.strict["TC_No"].fn == 1
        # 16 gold PII chars total; only the 5 Person chars masked, the 11 TC
        # chars leak → leak_rate = 11/16, char_recall = 5/16.
        assert r.char_recall == pytest.approx(5 / 16)
        assert r.leak_rate == pytest.approx(11 / 16)

    def test_boundary_mismatch_strict_vs_relaxed(self):
        # Predicted span overlaps but boundaries differ (e.g. Case_No over-capture).
        gold = [S(9, 18, "Case_No")]          # "2024/1456"
        pred = [S(0, 18, "Case_No")]          # "Esas No: 2024/1456"
        r = Report()
        score_doc(r, gold, pred)
        # Strict: boundary differs → miss + false positive
        assert r.micro(strict=True)[2] == 0.0
        # Relaxed: same label + overlap → counted as a hit
        assert r.relaxed["Case_No"].tp == 1
        # Char level: all gold chars covered → no leak
        assert r.leak_rate == 0.0

    def test_label_mismatch_same_span(self):
        gold = [S(0, 5, "Person")]
        pred = [S(0, 5, "Company")]
        r = Report()
        score_doc(r, gold, pred)
        assert r.strict["Person"].fn == 1
        assert r.strict["Company"].fp == 1
        # Char level is label-agnostic → still masked, no leak
        assert r.leak_rate == 0.0

    def test_macro_averages_per_label_f1(self):
        gold = [S(0, 5, "Person"), S(10, 21, "TC_No")]
        pred = [S(0, 5, "Person")]  # TC_No F1 = 0, Person F1 = 1
        r = Report()
        score_doc(r, gold, pred)
        _, _, macro_f1 = r.macro(strict=True)
        assert macro_f1 == pytest.approx(0.5)
