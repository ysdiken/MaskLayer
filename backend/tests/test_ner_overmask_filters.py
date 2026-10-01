"""
Unit tests for ner_engine.py over-masking filters:

1. Generic-tail trimming — the model over-extends Company spans into attached
   document titles ("Hava Bankası Tedarikçi Davranış İlkeleri"). Only the
   proper-name part must stay masked; the document-type words are context.
2. Public-platform blocklist — Twitter, Instagram & co. are public knowledge
   and must never be masked.

No ML model required — same stub pipeline approach as test_ner_chunking.py.
"""

import re

import pytest

from app.masking.ner_engine import (
    NerEngine,
    _is_blocklisted,
    _trim_generic_tail,
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
# Generic-tail trimming
# ---------------------------------------------------------------------------

class TestGenericTailTrimming:

    def test_document_title_trimmed_company_kept(self):
        # The reported false positive: company name + attached document title
        # tagged as one ORG span. Only the company name must stay masked.
        text = "Hava Bankası Tedarikçi Davranış İlkeleri tüm tedarikçiler için bağlayıcıdır."
        surfaces = detect_surfaces(
            text, {"Hava Bankası Tedarikçi Davranış İlkeleri": "ORG"}
        )
        assert surfaces == ["Hava Bankası"]

    def test_policy_suffix_trimmed(self):
        text = "Mavi Ada Teknoloji Gizlilik Politikası web sitesinde yayımlanmıştır."
        surfaces = detect_surfaces(
            text, {"Mavi Ada Teknoloji Gizlilik Politikası": "ORG"}
        )
        assert surfaces == ["Mavi Ada Teknoloji"]

    def test_case_suffixed_title_word_trimmed(self):
        # "İlkelerine" = ILKE + LERINE — root matching must handle suffix chains.
        text = "Hava Bankası Davranış İlkelerine uyulması zorunludur."
        surfaces = detect_surfaces(text, {"Hava Bankası Davranış İlkelerine": "ORG"})
        assert surfaces == ["Hava Bankası"]

    def test_company_yonergesi_trimmed_not_suppressed(self):
        # Leak-prevention: without trimming, the legal-reference filter would
        # suppress this WHOLE span (Yönerge keyword) and expose the bank name.
        text = "Hava Bankası Bilgi Güvenliği Yönergesi çalışanlara tebliğ edildi."
        surfaces = detect_surfaces(
            text, {"Hava Bankası Bilgi Güvenliği Yönergesi": "ORG"}
        )
        assert surfaces == ["Hava Bankası Bilgi Güvenliği"]

    def test_span_entirely_generic_dropped(self):
        text = "Tedarikçi Davranış İlkeleri ekte sunulmuştur."
        assert detect_surfaces(text, {"Tedarikçi Davranış İlkeleri": "ORG"}) == []

    def test_trim_cascades_into_blocklist(self):
        # "Banka Politikası" → trim "Politikası" → "Banka" → blocklisted.
        text = "Banka Politikası gereği işlem yapılamaz."
        assert detect_surfaces(text, {"Banka Politikası": "ORG"}) == []

    def test_legal_company_form_not_trimmed(self):
        # "A.Ş." is part of the legal name — trimming must stop there.
        text = "Mavi Ada Teknoloji A.Ş. yeni bir ofis açtı."
        surfaces = detect_surfaces(text, {"Mavi Ada Teknoloji A.Ş.": "ORG"})
        assert surfaces == ["Mavi Ada Teknoloji A.Ş."]

    def test_person_spans_never_trimmed(self):
        # Trimming applies to Company/Facility only.
        text = "Toplantıya Ahmet Kural katıldı."
        assert detect_surfaces(text, {"Ahmet Kural": "PER"}) == ["Ahmet Kural"]

    def test_trim_generic_tail_unit(self):
        text = "Hava Bankası Tedarikçi Davranış İlkeleri"
        end = _trim_generic_tail(text, 0, len(text))
        assert text[:end] == "Hava Bankası"


# ---------------------------------------------------------------------------
# Public platforms
# ---------------------------------------------------------------------------

class TestPublicPlatforms:

    def test_platforms_blocklisted_in_both_spellings(self):
        for name in ("Twitter", "Instagram", "İnstagram", "LinkedIn",
                      "YouTube", "WhatsApp", "TikTok", "Facebook"):
            assert _is_blocklisted(name), name

    def test_platform_span_suppressed(self):
        text = "Duyurular Twitter ve Instagram üzerinden paylaşılacaktır."
        assert detect_surfaces(text, {"Twitter": "ORG", "Instagram": "ORG"}) == []

    def test_platform_with_apostrophe_suffix_suppressed(self):
        text = "Kampanya Instagram'da yayınlandı."
        assert detect_surfaces(text, {"Instagram'da": "ORG"}) == []

    def test_real_company_next_to_platform_still_masked(self):
        text = "Mavi Ada Teknoloji, duyuruyu LinkedIn hesabından yaptı."
        surfaces = detect_surfaces(
            text, {"Mavi Ada Teknoloji": "ORG", "LinkedIn": "ORG"}
        )
        assert surfaces == ["Mavi Ada Teknoloji"]
