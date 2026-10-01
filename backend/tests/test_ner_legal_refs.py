"""
Unit tests for ner_engine.py legal-reference suppression.

Background: WikiANN has no LAW label, so statute/regulation names like
"5846 sayılı Fikir ve Sanat Eserleri Kanunu" get classified as ORG and
masked as {Company_N}. These are public citations, not PII — the
suppression rules (A: keyword in span, B: keyword right after a Company
span, C: overlap with a "<no> sayılı …" citation) must drop them while
leaving real company and person names untouched.

No ML model required — same stub pipeline approach as test_ner_chunking.py.
"""

import re

import pytest

from app.masking.ner_engine import (
    NerEngine,
    _is_blocklisted,
    _legal_citation_zones,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubTokenizer:
    """Whitespace tokenizer: one token per word, real character offsets."""

    def __call__(self, text, **kwargs):
        return {
            "offset_mapping": [
                (m.start(), m.end()) for m in re.finditer(r"\S+", text)
            ]
        }


class StubNerPipeline:
    """Emits an HF-style entity dict for every occurrence of each name."""

    def __init__(self, names: dict[str, str]):
        self.names = names  # surface form → entity_group (PER/LOC/ORG)
        self.tokenizer = StubTokenizer()

    def __call__(self, chunk: str):
        ents = []
        for name, group in self.names.items():
            start = 0
            while True:
                idx = chunk.find(name, start)
                if idx == -1:
                    break
                ents.append({
                    "entity_group": group,
                    "score": 0.95,
                    "word": name,
                    "start": idx,
                    "end": idx + len(name),
                })
                start = idx + len(name)
        return ents


def detect_surfaces(text: str, names: dict[str, str]) -> list[str]:
    """Run detect() with a stub pipeline; return the surviving span texts."""
    eng = NerEngine()
    eng._pipeline = StubNerPipeline(names)
    eng._available = True
    eng._load_attempted = True
    return [text[s.start: s.end] for s in eng.detect(text)]


# ---------------------------------------------------------------------------
# Rule A — law keyword inside the span
# ---------------------------------------------------------------------------

class TestKeywordInSpan:

    def test_full_law_name_suppressed(self):
        text = "İşbu sözleşme 5846 sayılı Fikir ve Sanat Eserleri Kanunu hükümlerine tabidir."
        assert detect_surfaces(text, {"Fikir ve Sanat Eserleri Kanunu": "ORG"}) == []

    def test_law_name_with_case_suffix_suppressed(self):
        text = "Türk Borçlar Kanunu'nun 27. maddesi uyarınca işlem yapılır."
        assert detect_surfaces(text, {"Türk Borçlar Kanunu'nun": "ORG"}) == []

    def test_yonetmelik_suppressed(self):
        text = "İş Sağlığı ve Güvenliği Yönetmeliği kapsamında denetim yapıldı."
        assert detect_surfaces(text, {"İş Sağlığı ve Güvenliği Yönetmeliği": "ORG"}) == []

    def test_anayasa_suppressed(self):
        text = "Başvuru Anayasa Mahkemesi tarafından reddedildi."
        assert detect_surfaces(text, {"Anayasa Mahkemesi": "ORG"}) == []


# ---------------------------------------------------------------------------
# Rule B — Company span immediately followed by a law keyword
# ---------------------------------------------------------------------------

class TestKeywordFollowsSpan:

    def test_title_part_before_kanunu_suppressed(self):
        # Model tags only the title words; "Kanunu" stays outside the span.
        text = "İşbu sözleşme Fikir ve Sanat Eserleri Kanunu hükümlerine tabidir."
        assert detect_surfaces(text, {"Fikir ve Sanat Eserleri": "ORG"}) == []

    def test_person_followed_by_kanun_not_suppressed(self):
        # Rule B is Company-only: a Person span must survive even when the
        # next word is "Kanun…" — suppression here would be a PII leak.
        text = "Ahmet Yılmaz, Kanun hükümlerine uymakla yükümlüdür."
        assert detect_surfaces(text, {"Ahmet Yılmaz": "PER"}) == ["Ahmet Yılmaz"]


# ---------------------------------------------------------------------------
# Rule C — span inside a "<no> sayılı …" citation zone
# ---------------------------------------------------------------------------

class TestCitationZones:

    def test_zone_covers_full_citation(self):
        text = "Veriler 6698 sayılı Kişisel Verilerin Korunması Kanunu uyarınca işlenir."
        zones = _legal_citation_zones(text)
        assert len(zones) == 1
        z_start, z_end = zones[0]
        assert text[z_start: z_end] == "6698 sayılı Kişisel Verilerin Korunması Kanunu"

    def test_inner_title_word_suppressed(self):
        # A fragment inside the citation, not adjacent to the keyword itself.
        text = "İşbu sözleşme 5846 sayılı Fikir ve Sanat Eserleri Kanunu hükümlerine tabidir."
        assert detect_surfaces(text, {"Sanat Eserleri": "ORG"}) == []

    def test_no_zone_without_sayili(self):
        assert _legal_citation_zones("Mavi Ada Teknoloji A.Ş. bir yazılım şirketidir.") == []


# ---------------------------------------------------------------------------
# Real PII near legal references must survive
# ---------------------------------------------------------------------------

class TestRealEntitiesSurvive:

    def test_company_in_same_sentence_as_citation_still_masked(self):
        text = (
            "Mavi Ada Teknoloji A.Ş., 6698 sayılı Kişisel Verilerin Korunması "
            "Kanunu uyarınca veri sorumlusudur."
        )
        surfaces = detect_surfaces(text, {
            "Mavi Ada Teknoloji": "ORG",
            "Kişisel Verilerin Korunması Kanunu": "ORG",
        })
        assert surfaces == ["Mavi Ada Teknoloji"]

    def test_person_in_same_sentence_as_law_still_masked(self):
        text = "Elif Şahin, İş Kanunu kapsamında işe iade davası açtı."
        surfaces = detect_surfaces(text, {
            "Elif Şahin": "PER",
            "İş Kanunu": "ORG",
        })
        assert surfaces == ["Elif Şahin"]


# ---------------------------------------------------------------------------
# Statute abbreviations (blocklist)
# ---------------------------------------------------------------------------

class TestParentheticalAlias:
    """Bare ("X") definitions (real-contract style) → defined terms, not PII (v0.5.1)."""

    def test_extracts_parenthetical_terms(self):
        from app.masking.pipeline import _extract_contract_aliases
        text = 'Taşıt Kredisi’nin (“Kredi”) ... A.Ş (“Biz”) ... müşterimiz (“Siz”)'
        aliases = _extract_contract_aliases(text)
        assert "KREDI" in aliases
        assert "BIZ" in aliases
        assert "SIZ" in aliases

    def test_does_not_catch_bundan_boyle_clause(self):
        # The existing "(bundan böyle 'Şirket' ...)" clause has text between the
        # paren and the quote, so the bare-parenthetical rule must not fire on it
        # (the dedicated clause rule still extracts ŞİRKET).
        from app.masking.pipeline import _extract_contract_aliases
        text = 'Mavi Ada A.Ş. (bundan böyle “Şirket” olarak anılacaktır)'
        aliases = _extract_contract_aliases(text)
        # "Şirket".upper() → "ŞIRKET" (ASCII I; Python upper() is not Turkish-locale-aware)
        assert "ŞIRKET" in aliases


class TestStatuteAbbreviations:

    def test_kvkk_blocklisted(self):
        assert _is_blocklisted("KVKK")
        assert _is_blocklisted("FSEK")
        assert _is_blocklisted("Resmi Gazete")

    def test_kvkk_span_suppressed(self):
        text = "KVKK uyarınca açık rıza alınması zorunludur."
        assert detect_surfaces(text, {"KVKK": "ORG"}) == []

    def test_kvkk_with_straight_apostrophe_suffix_suppressed(self):
        # Apostrophe splitting must handle the ASCII ' (OCR output) so the
        # root "KVKK" reaches the blocklist check.
        text = "KVKK'nın 5. maddesi açık rızayı düzenler."
        assert detect_surfaces(text, {"KVKK'nın": "ORG"}) == []
