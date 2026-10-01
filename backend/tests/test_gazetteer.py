"""
Tests for the gazetteer detector + public-institution suppression.

Offline — the gazetteer is pure pattern/dictionary matching, no model.
"""

import pytest

from app.masking.gazetteer import engine
from app.masking.ner_engine import is_institution
from app.masking.span_merger import merge
from app.models.schemas import Span


def companies(text):
    return [s for s in engine.detect(text) if s.label == "Company"]

def persons(text):
    return [s for s in engine.detect(text) if s.label == "Person"]

def locations(text):
    return [s for s in engine.detect(text) if s.label == "Location"]


# ---------------------------------------------------------------------------
# Company legal-form detection
# ---------------------------------------------------------------------------

class TestCompanyDetection:

    def test_as_with_trailing_period(self):
        text = "Düzenleyen: Yıldız Tekstil A.Ş. tarafından hazırlandı."
        spans = companies(text)
        assert len(spans) == 1
        assert spans[0].text == "Yıldız Tekstil A.Ş."   # period included
        assert spans[0].source == "gazetteer"

    def test_all_caps_company(self):
        text = "KURUM: DEMİR İNŞAAT A.Ş."
        spans = companies(text)
        assert len(spans) == 1
        assert spans[0].text == "DEMİR İNŞAAT A.Ş."

    def test_holding(self):
        spans = companies("davalı Nur Holding A.Ş. arasında")
        assert len(spans) == 1
        assert spans[0].text == "Nur Holding A.Ş."

    def test_ltd_sti(self):
        spans = companies("taraf Teknova Bilişim Ltd. Şti. olarak")
        assert len(spans) == 1
        assert spans[0].text.startswith("Teknova Bilişim Ltd")

    def test_law_name_not_matched(self):
        # "Türk Ticaret Kanunu" has no legal-form suffix → not a company.
        assert companies("6102 sayılı Türk Ticaret Kanunu uyarınca") == []

    def test_boundary_excludes_label_word(self):
        # The capture must not swallow a preceding lowercase label.
        text = "şirket Mavi Ada Teknoloji A.Ş. adına"
        spans = companies(text)
        assert spans[0].text == "Mavi Ada Teknoloji A.Ş."

    def test_suffix_not_matched_midword(self):
        # "AŞAĞIDAKİ" must NOT yield "LÜTFEN AŞ" — the bare-AŞ suffix may only
        # match a complete legal-form token (real-doc leak, fixed).
        assert companies("İSE; LÜTFEN AŞAĞIDAKİ BEYANI") == []

    def test_leading_initial_kept(self):
        # "T. GARANTİ BANKASI A.Ş." captured whole, not dropping the "T.".
        spans = companies("Banka: T. GARANTİ BANKASI A.Ş. tarafından")
        assert spans and spans[0].text == "T. GARANTİ BANKASI A.Ş."

    def test_apostrophe_after_suffix_ok(self):
        spans = companies("Mavi Ada A.Ş.'nin yetkilisi")
        assert spans and spans[0].text == "Mavi Ada A.Ş."


# ---------------------------------------------------------------------------
# Person lookup
# ---------------------------------------------------------------------------

class TestPersonLookup:

    def test_known_firstname_plus_surname(self):
        spans = persons("Toplantıya Ahmet Yılmaz katıldı.")
        assert len(spans) == 1
        assert spans[0].text == "Ahmet Yılmaz"

    def test_unknown_firstname_not_matched(self):
        # "Mavi" is not a first name → no person span.
        assert persons("Mavi Ada projesi") == []

    def test_requires_two_tokens(self):
        # A lone first name is too ambiguous to mask.
        assert persons("Sayın Ahmet, hoş geldiniz.") == []

    def test_company_keyword_not_a_surname(self):
        # "Nur Holding" / "Pınar Emlak" must NOT be tagged Person (the keyword
        # would leak into the span); the company detector handles the real org.
        assert persons("davalı Nur Holding A.Ş. arasında") == []
        assert persons("Pınar Emlak A.Ş. ile") == []

    def test_honorific_not_a_surname(self):
        assert persons("Sayın Ahmet Bey ile görüşüldü") == []


# ---------------------------------------------------------------------------
# Location lookup
# ---------------------------------------------------------------------------

class TestLocationLookup:

    def test_province_detected(self):
        spans = locations("İşlem İstanbul şubesinde yapıldı.")
        assert [s.text for s in spans] == ["İstanbul"]

    def test_district_detected(self):
        spans = locations("Kadıköy mağazamız açıldı.")
        assert [s.text for s in spans] == ["Kadıköy"]

    def test_non_place_not_matched(self):
        assert locations("Teknoloji geliştirme departmanı") == []

    def test_place_in_institution_name_skipped(self):
        # "Ankara" inside a court name must NOT be tagged (it's not PII here).
        text = "Ankara 3. Asliye Ticaret Mahkemesi kararı"
        assert locations(text) == []


# ---------------------------------------------------------------------------
# Institution suppression
# ---------------------------------------------------------------------------

class TestIsInstitution:

    @pytest.mark.parametrize("name", [
        "Ankara 3. Asliye Ticaret Mahkemesi",
        "Beyoğlu 5. Noterliği",
        "İstanbul Valiliği",
        "Çevre ve Şehircilik Bakanlığı",
        "Kadıköy Belediyesi",
        "Sosyal Güvenlik Kurumu",
        # Case-suffixed forms + unions/systems (real-contract cases, v0.5.1)
        "tüketici mahkemesine",
        "tüketici hakem heyetine",
        "Türkiye Noterler Birliği",
        "Adres Kayıt Sistemi",
        "Adres Kayıt Sisteminde",
    ])
    def test_institutions(self, name):
        assert is_institution(name) is True

    @pytest.mark.parametrize("name", [
        "Mavi Ada Teknoloji A.Ş.",
        "Ahmet Yılmaz",
        "İstanbul",
        "Garanti Bankası",
    ])
    def test_non_institutions(self, name):
        assert is_institution(name) is False


# ---------------------------------------------------------------------------
# Merge: gazetteer overrides NER's truncated boundary
# ---------------------------------------------------------------------------

class TestMergeWithGazetteer:

    def test_gazetteer_full_boundary_beats_ner_truncated(self):
        # NER drops the final period; gazetteer keeps it. Longest span wins.
        text = "Yıldız Tekstil A.Ş."
        ner = [Span(start=0, end=18, label="Company", source="ner",
                    confidence=0.9, text="Yıldız Tekstil A.Ş")]
        gaz = [Span(start=0, end=19, label="Company", source="gazetteer",
                    confidence=0.9, text="Yıldız Tekstil A.Ş.")]
        result = merge([], ner, gaz)
        assert len(result) == 1
        assert result[0].text == "Yıldız Tekstil A.Ş."
        assert result[0].source == "gazetteer"
