"""
Gazetteer engine — a high-precision, dictionary/pattern-based third detector.

Complements the regex engine (structured PII) and the NER model (semantic
entities) with deterministic lookups that are strong exactly where the model is
weak or unavailable:

  1. Company legal-form detection — "<Titlecase/CAPS words> A.Ş./Ltd. Şti./…".
     Fixes the NER trailing-period boundary artdefact ("Yıldız Tekstil A.Ş"
     → "Yıldız Tekstil A.Ş.") and adds recall for suffix-bearing companies even
     in ALL-CAPS text where the cased NER model degrades.
  2. Person lookup — a curated list of common Turkish first names; emits a span
     only for "<known first name> <Titlecase surname>" (two tokens), which keeps
     precision high (never fires on a lone ambiguous word).
  3. Location lookup — Turkish provinces + major districts. Skipped when the
     place is part of a public-institution name ("Ankara … Mahkemesi"), so the
     gazetteer never masks a court/municipality location.

Design: precision over recall. Every match is deterministic and reproducible;
spans carry source="gazetteer" so the merger and ablation can treat this as a
distinct detector. Public-institution spans are filtered downstream in the
pipeline (see is_institution), shared with the NER path.
"""

from __future__ import annotations

import re

from app.masking.ner_engine import is_institution
from app.models.schemas import Span


# ---------------------------------------------------------------------------
# Company legal-form detection
# ---------------------------------------------------------------------------

# A single name token: Turkish Titlecase ("Yıldız") or ALL-CAPS ("İNŞAAT").
_NAME_TOKEN = r"(?:[A-ZÇĞİÖŞÜ][a-zçğışöşü]+|[A-ZÇĞİÖŞÜ]{2,})"

# Legal-form suffixes — the trailing period is consumed greedily so the span
# boundary is correct ("A.Ş." not "A.Ş"). Covers the common Turkish forms.
_COMPANY_SUFFIX = (
    r"(?:"
    r"A\.?\s?Ş\.?"                          # A.Ş.  (Anonim Şirketi)
    r"|Ltd\.?\s?Şti\.?"                     # Ltd. Şti.
    r"|San\.?\s?ve\s?Tic\.?(?:\s?A\.?\s?Ş\.?)?"  # San. ve Tic. (A.Ş.)
    r"|T\.?A\.?\s?Ş\.?"                     # T.A.Ş.
    r"|A\.?\s?O\.?"                         # A.O.
    r"|Holding(?:\s+A\.?\s?Ş\.?)?"          # Holding / Holding A.Ş.
    r")"
)

# Optional leading single-initial abbreviation so "T. GARANTİ BANKASI A.Ş." is
# captured whole rather than dropping the "T." (Türkiye) prefix.
_INITIAL_PREFIX = r"(?:[A-ZÇĞİÖŞÜ]\.\s*)?"

# The trailing negative lookahead is critical: the suffix alternatives (esp.
# bare "AŞ" with the periods optional) must NOT match mid-word. Without it,
# "LÜTFEN AŞAĞIDAKİ" matched "LÜTFEN AŞ" as a company, leaking context into the
# masked span. Requiring a non-letter after the suffix forces it to be a
# complete legal-form token. (An apostrophe — "A.Ş.'nin" — still passes.)
_COMPANY_RE = re.compile(
    rf"\b({_INITIAL_PREFIX}{_NAME_TOKEN}(?:\s+(?:{_NAME_TOKEN}|ve))*\s+{_COMPANY_SUFFIX})"
    r"(?![A-Za-zÇĞİÖŞÜçğışöşü])"
)

COMPANY_CONFIDENCE = 0.90


# ---------------------------------------------------------------------------
# Person first-name lookup
# ---------------------------------------------------------------------------

_FIRST_NAMES: frozenset[str] = frozenset({
    # Male
    "Ahmet", "Mehmet", "Mustafa", "Ali", "Hüseyin", "Hasan", "İbrahim", "Ömer",
    "Yusuf", "Murat", "Osman", "Kemal", "Orhan", "Selim", "Levent", "Fatih",
    "Kenan", "Hakan", "Gökhan", "Serkan", "Tolga", "Onur", "Barış", "Cem",
    "Ozan", "Volkan", "Uğur", "Engin", "Sinan", "Arda", "Kerem", "Yiğit",
    "Berk", "Eren", "Kaan", "Mert", "Burak", "Emre", "Kadir", "Ramazan",
    "Tarık", "Selim", "Sefa", "Deniz", "Can", "Ege", "Doruk", "Bora", "Koray",
    "Selçuk", "Cenk", "Tuncay", "Erdem", "Görkem", "Anıl", "Kenan", "Polat",
    # Female
    "Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Sultan",
    "Burcu", "Gül", "Sema", "Selin", "Defne", "Nur", "Ece", "Pınar", "Yasemin",
    "Esra", "Merve", "Büşra", "Derya", "Sibel", "Gamze", "Ebru", "Aslı",
    "Çağla", "Dilara", "Melis", "İrem", "Naz", "Eylül", "Buse", "Tuğçe",
    "Özge", "Gizem", "Şeyma", "Beren", "Ceren", "Damla", "Ela", "Nehir",
    "Sıla", "Duru", "Azra", "Zehra", "Hande", "Bahar", "Gül", "Sevgi",
})

PERSON_CONFIDENCE = 0.80

# Every Titlecase word + position. We pair a known first name with the
# following Titlecase token in code (a single regex would greedily mis-pair
# "Toplantıya Ahmet" before reaching "Ahmet Yılmaz").
_TITLECASE_TOKEN = re.compile(r"[A-ZÇĞİÖŞÜ][a-zçğışöşü]+")
# Only a same-line run of spaces may separate the two name tokens.
_SPACES_ONLY = re.compile(r"[ \t]+")

# Words that are NOT surnames — company sector/legal terms and honorifics. Stops
# "Nur Holding" / "Pınar Emlak" (a first name + company keyword) being masked as
# a person, which leaks the keyword into the span. The company detector still
# catches the real org ("Nur Holding A.Ş.").
_NOT_SURNAME: frozenset[str] = frozenset({
    "Holding", "Emlak", "Sigorta", "Banka", "Bankası", "Teknoloji", "Tekstil",
    "İnşaat", "Otomotiv", "Bilişim", "Yazılım", "Lojistik", "Sanayi", "Ticaret",
    "Gıda", "Enerji", "Turizm", "Makine", "Mobilya", "Kimya", "Elektronik",
    "Grup", "Group", "Şirketi", "Limited", "Anonim", "Mağaza", "Market",
    "Bey", "Hanım", "Bay", "Bayan",  # honorifics, not surnames
})


# ---------------------------------------------------------------------------
# Location lookup — provinces + major districts
# ---------------------------------------------------------------------------

_PROVINCES: frozenset[str] = frozenset({
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya",
    "Artvin", "Aydın", "Balıkesir", "Bilecik", "Bingöl", "Bitlis", "Bolu",
    "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır",
    "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep",
    "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Isparta", "Mersin", "İstanbul",
    "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli", "Kırşehir", "Kocaeli",
    "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla",
    "Muş", "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt",
    "Sinop", "Sivas", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Şanlıurfa",
    "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman",
    "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova",
    "Karabük", "Kilis", "Osmaniye", "Düzce",
    # Major districts
    "Kadıköy", "Beşiktaş", "Şişli", "Üsküdar", "Bakırköy", "Beyoğlu", "Maltepe",
    "Pendik", "Çankaya", "Keçiören", "Bornova", "Konak", "Muratpaşa", "Nilüfer",
    "Osmangazi", "Selçuklu", "Çekmeköy", "Ataşehir", "Etimesgut",
})

LOCATION_CONFIDENCE = 0.85

_PLACE_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğışöşü]+)")

# Institution keyword anywhere within this many chars after a place → the place
# is part of an institution name ("Ankara 3. Asliye Ticaret Mahkemesi") → skip.
_INSTITUTION_LOOKAHEAD = 45
_INSTITUTION_WORD_RE = re.compile(
    r"\b(?:Mahkeme|Adliye|Valilik|Kaymakamlık|Belediye|Müdürlük|Başkanlık|"
    r"Bakanlık|Noter|Savcılık|Müsteşarlık|Kurum)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GazetteerEngine:
    """Deterministic dictionary/pattern detector. Stateless and synchronous."""

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        spans.extend(self._companies(text))
        spans.extend(self._persons(text))
        spans.extend(self._locations(text))
        return spans

    # ── Companies ─────────────────────────────────────────────────────────────
    def _companies(self, text: str) -> list[Span]:
        out: list[Span] = []
        for m in _COMPANY_RE.finditer(text):
            name = m.group(1).strip()
            if is_institution(name):
                continue   # "… A.Ş." would never be an institution, but be safe
            out.append(Span(
                start=m.start(1), end=m.start(1) + len(name),
                label="Company", source="gazetteer",
                confidence=COMPANY_CONFIDENCE, text=name, canonical_id=None,
            ))
        return out

    # ── Persons ───────────────────────────────────────────────────────────────
    def _persons(self, text: str) -> list[Span]:
        out: list[Span] = []
        tokens = list(_TITLECASE_TOKEN.finditer(text))
        for i, tok in enumerate(tokens):
            if tok.group(0) not in _FIRST_NAMES:
                continue
            if i + 1 >= len(tokens):
                continue
            nxt = tokens[i + 1]
            if nxt.group(0) in _NOT_SURNAME:       # "Holding"/"Emlak"/honorific → not a person
                continue
            gap = text[tok.end():nxt.start()]
            if not _SPACES_ONLY.fullmatch(gap):   # must be adjacent, same line
                continue
            out.append(Span(
                start=tok.start(), end=nxt.end(),
                label="Person", source="gazetteer",
                confidence=PERSON_CONFIDENCE,
                text=text[tok.start():nxt.end()], canonical_id=None,
            ))
        return out

    # ── Locations ─────────────────────────────────────────────────────────────
    def _locations(self, text: str) -> list[Span]:
        out: list[Span] = []
        for m in _PLACE_RE.finditer(text):
            place = m.group(1)
            if place not in _PROVINCES:
                continue
            # Skip places that are part of a public-institution name.
            window = text[m.end(): m.end() + _INSTITUTION_LOOKAHEAD]
            if _INSTITUTION_WORD_RE.search(window):
                continue
            out.append(Span(
                start=m.start(1), end=m.end(1),
                label="Location", source="gazetteer",
                confidence=LOCATION_CONFIDENCE, text=place, canonical_id=None,
            ))
        return out


engine = GazetteerEngine()
