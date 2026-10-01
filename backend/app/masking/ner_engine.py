"""
HuggingFace Transformers NER engine for Turkish named entity recognition.

Model: savasy/bert-base-turkish-ner-cased
  - Backbone: dbmdz/bert-base-turkish-cased (BERTurk)
  - Training: Turkish WikiANN dataset
  - Labels: B-PER / I-PER, B-LOC / I-LOC, B-ORG / I-ORG
  - Size: ~440 MB (downloaded once, cached in ~/.cache/huggingface)
  - F1: ~0.93 on WikiANN Turkish test set

Install:
    pip install transformers torch

Download (first run — model cached locally afterwards):
    The model downloads automatically on first detect() call.
    To pre-download offline: run `python -c "from app.masking.ner_engine import engine; engine.available"`

GPU usage:
    Automatically uses CUDA if available (your RTX 5070 Ti).
    VRAM usage: ~900 MB for bert-base. Well within your 16 GB budget.
    Inference: ~20-50 ms per document on GPU.

Graceful degradation:
    If transformers/torch are not installed, or model download fails,
    detect() returns [] and logs a warning. The backend continues
    with regex-only masking.

Turkish NLP rules:
    Apostrophe splitting: "Samsun'a" → entity ends before the apostrophe.
    The HuggingFace tokenizer splits on the apostrophe, so the entity
    span returned by the pipeline covers only the root word. We extract
    character offsets from the pipeline's start/end fields directly.

Long documents:
    BERT accepts at most 512 tokens (~1 page of Turkish text). detect()
    slides an overlapping token window over the document and runs the
    pipeline per window, so entities on page 2+ are detected too.
    See _compute_chunks() for the ownership rule that prevents duplicate
    or truncated entities at window edges.

Thesis note:
    This is the ML contribution. Entities this catches that regex misses:
    Person names (no fixed format), Company names, Location names.
    Compare regex-only F1 vs regex+NER F1 per entity type for ablation.
    The model's per-token confidence scores are averaged across sub-tokens
    to produce a single span confidence — used for the HITL routing threshold.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

from app.models.schemas import Span

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Model directories. The fine-tuned checkpoint (finetune/train.py — domain
# adapted so capitalised Turkish common/domain nouns aren't tagged ORG) is
# preferred when present; it eliminated the real-contract over-tagging without
# regressing synthetic (see finetune/README.md). Falls back to the baked
# WikiANN model, then the Hub.
_FT_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "bert-turkish-ner-ft"
_LOCAL_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "bert-turkish-ner"

# Override priority: MASKLAYER_NER_MODEL env var > fine-tuned > baked > Hub.
# The env var lets the eval harness A/B baseline vs fine-tuned without edits
# (set it to the baked dir to force the pre-fine-tune baseline).
def _default_model() -> str:
    if _FT_MODEL_DIR.exists():
        return str(_FT_MODEL_DIR)
    if _LOCAL_MODEL_DIR.exists():
        return str(_LOCAL_MODEL_DIR)
    return "savasy/bert-base-turkish-ner-cased"


MODEL_NAME: str = os.getenv("MASKLAYER_NER_MODEL") or _default_model()

# ── Long-document chunking ────────────────────────────────────────────────
# BERT's position embeddings cap input at 512 tokens; longer input is either
# truncated or crashes — entities past the first page would silently vanish.
# detect() therefore slides a window of _CHUNK_MAX_TOKENS over the document
# with _CHUNK_OVERLAP_TOKENS of context shared between consecutive windows.
# Each window exclusively "owns" the region outside the shared margins, so an
# entity near a window edge is reported once, by the window that sees it with
# full context on both sides. Margin (overlap/2 = 32 tokens) far exceeds any
# entity length, so the owning window always contains the entity intact.
_CHUNK_MAX_TOKENS = 384      # window size — headroom for [CLS]/[SEP] under 512
_CHUNK_OVERLAP_TOKENS = 64   # context shared between consecutive windows

# BIO label → MaskLayer taxonomy label
LABEL_MAP: dict[str, str] = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Company",
}

# Fixed confidence for spans where the pipeline returns a single score.
# The transformers pipeline returns per-token scores; we take the min
# across sub-tokens (conservative: weakest link in the span).
# Spans below 0.60 are skipped (below the HITL lower threshold).
MIN_CONFIDENCE = 0.70   # raised from 0.60 — reduces noisy ORG false-positives

# Per-label confidence floor (stricter for ORG which has most false-positives)
LABEL_MIN_CONFIDENCE: dict[str, float] = {
    "Person":   0.65,   # lowered from 0.70 — multi-word names like "Selin Arda Kılıç"
                        # often score 0.66–0.69 when surrounded by form/table context
    "Company":  0.78,   # ORG is noisiest on legal/bureaucratic text
    "Location": 0.70,
    "Title":    0.72,
    "Facility": 0.72,
}

# Turkish word-character pattern — ASCII + Turkish-specific letters
_WORD_CHAR = re.compile(r"[A-Za-z0-9ğüşıöçĞÜŞİÖÇ_]")

# Common Turkish legal/document abbreviations and generic terms that the NER
# model frequently misclassifies as ORG/LOC. Never mask these bare tokens.
# Stored in UPPERCASE — checked case-insensitively (see _is_blocklisted()).
_NER_BLOCKLIST: frozenset[str] = frozenset({
    # Abbreviations that appear verbatim in legal docs
    "TC", "TCKN", "TCN", "VKN", "IBAN", "KDV", "TL", "₺",
    "API", "URL", "IP", "ID", "CVC", "PIN",
    "A.Ş", "A.Ş.", "LTD", "LTD.", "VD", "MERSİS",
    # Address components
    "CAD", "CAD.", "SOK", "SOK.", "MAH", "MAH.", "BLV", "BLV.",
    "NO", "NO.", "KAT", "DAİRE", "İÇ KAPI",
    # Document / form words the model sometimes tags as ORG
    "SÖZLEŞME", "MADDE", "TARAF", "HİZMET", "MÜŞTERİ",
    "FATURA", "ÖDEME", "BANKA", "HESAP", "BELGE",
    "TEST", "MASKELEME", "KURGUSAL",
    "BİLGİ", "TÜRÜ", "KONU", "SENARYO",
    # Multi-word party role phrases
    "KİRAYA VEREN", "KIRAYA VEREN",
    "HİZMET ALAN", "HIZMET ALAN",
    "HİZMET SAĞLAYICI", "HIZMET SAGLAYICI",
    # Statute / regulator abbreviations — public legal references, not PII.
    # (3-letter ones like TTK/TBK/KHK are already dropped by the ≤3-char
    # Company/Location rule in _process_entities.)
    "KVKK", "FSEK", "İYUK", "IYUK", "GDPR", "TBMM", "BDDK", "MASAK",
    "RESMİ GAZETE", "RESMI GAZETE",
    # Banking/finance tax abbreviations and jargon the NER tags as ORG in real
    # contracts (capitalised common nouns). Generic across Turkish finance docs.
    "KKDF", "BSMV", "ÖTV", "OTV",
    "OLUR", "AKDİ", "AKDI", "TEMERRÜT", "TEMERRUT", "MUACCEL",
    "BİLEŞİK", "BILESIK", "MAHSUP", "ANAPARA", "MEBLAĞ", "MEBLAG",
    "FERİ", "FERI", "TEMİNAT", "TEMINAT", "REHİN", "REHIN",
    # Public social media / communication platforms — public knowledge,
    # masking them removes context without protecting anyone.
    # Both I/İ spellings stored: "İnstagram".upper() keeps İ, "Instagram"
    # uppercases to I (see the dotted-İ note on _BLOCKLIST_ROOTS).
    "TWITTER", "INSTAGRAM", "İNSTAGRAM", "FACEBOOK", "LINKEDIN", "LİNKEDİN",
    "YOUTUBE", "WHATSAPP", "TIKTOK", "TİKTOK", "TELEGRAM", "SNAPCHAT",
    "PINTEREST", "PİNTEREST", "REDDIT", "REDDİT", "DISCORD", "DİSCORD",
    "SKYPE", "ZOOM", "GOOGLE",
})

# Root forms for Turkish agglutination matching.
# "Şirket" (nominative) is caught by the exact blocklist above, but contracts also
# use case-suffixed forms WITHOUT apostrophe ("şirketi", "şirkete", "şirketin", …)
# because "şirket" is a common noun, not a proper name.
# Any span whose .upper() starts with one of these roots AND whose suffix length
# is ≤ 5 (covers all Turkish nominal case/possessive suffixes) is blocked.
_BLOCKLIST_ROOTS: tuple[str, ...] = (
    # Python .upper() maps "i"→"I", so "Şirket" → "ŞIRKET" (not "ŞİRKET").
    # Store both forms so matches work regardless of source encoding.
    "ŞIRKET",    # şirket / şirketi / şirkete / şirketin / şirketten / şirkette
    "BANKA",     # banka  / bankası / bankaya / bankadan / bankada / bankanın
    "KIRACI",    # kiracı / kiracıya / kiracıdan
    "ALICI",     # alıcı  / alıcıya  / alıcıdan
    "SATICI",    # satıcı / satıcıya / satıcıdan
    "ISVEREN",   # işveren + suffixes
    "ISCI",      # işçi   + suffixes
    "TASERON",   # taşeron + suffixes
    "YUKLENICI", # yüklenici + suffixes
    "BORCLU",    # borçlu + suffixes
    "ALACAKLI",  # alacaklı + suffixes
    "SIGORTACI", # sigortacı + suffixes
    "SIGORTALI", # sigortalı + suffixes
    "ARACI",     # aracı + suffixes
)


def _is_blocklisted(word: str) -> bool:
    """
    Return True if `word` is a standalone generic/placeholder term that should
    never be masked — whether it appears as a bare root or with a Turkish
    agglutination suffix attached without an apostrophe.

    Two-pass check:
      1. Exact match against _NER_BLOCKLIST (handles abbreviations and
         multi-word phrases).
      2. Root-prefix match against _BLOCKLIST_ROOTS: catches agglutinated
         forms like "Şirketi", "Şirkete", "Şirketin" (suffix len ≤ 5).
         Suffix cap prevents blocking real compound words (e.g. "Bankacılık").

    Does NOT match multi-word proper names that merely *contain* a root —
    e.g. "Garanti Bankası" has len("GARANTI BANKASI") - len("BANKA") = 10 > 5,
    so it passes through correctly.
    """
    upper = word.strip().upper()

    # Pass 1 — exact / abbreviation match
    if upper in _NER_BLOCKLIST or upper.rstrip(".") in _NER_BLOCKLIST:
        return True

    # Pass 2 — agglutination root match (single-word spans only)
    if " " not in upper:   # multi-word → can only be a proper name, not a suffix form
        for root in _BLOCKLIST_ROOTS:
            if upper.startswith(root):
                suffix_len = len(upper) - len(root)
                if 0 <= suffix_len <= 5:
                    return True

    return False


# ---------------------------------------------------------------------------
# Generic-tail trimming
# ---------------------------------------------------------------------------
# The model often over-extends a Company span into an attached document title:
# "Hava Bankası Tedarikçi Davranış İlkeleri" tagged as one ORG. The company
# name is PII; the document-type words are context that must stay readable.
# We trim generic title words off the RIGHT edge only (Turkish puts the
# document type after the name) until a non-generic word is reached.
# Trimming runs BEFORE legal-reference suppression so a company policy named
# "… Yönergesi" loses the title words instead of having the whole span —
# company name included — suppressed (which would leak the name unmasked).

def _tr_fold(s: str) -> str:
    """Uppercase with dotted İ folded to I — canonical form for root matching
    regardless of source spelling ("İlkeleri" → İLKELERİ, "ilkeleri" → ILKELERI)."""
    return s.upper().replace("İ", "I")


# Document-title word roots, in _tr_fold() form. Root-prefix matching with a
# suffix cap of 6 covers Turkish case/possessive/plural chains ("İlkelerine" =
# ILKE + LERINE). Deliberately absent: words that end real company names —
# YÖNETIM ("… Varlık Yönetimi"), GÜVENLIK ("… Güvenlik A.Ş."), HOLDING.
_TRIM_TAIL_ROOTS: tuple[str, ...] = (
    "ILKE",        # ilkeleri / ilkesi          (Davranış İlkeleri)
    "POLITIKA",    # politikası                 (Gizlilik Politikası)
    "PROSEDÜR",    # prosedürü
    "SÖZLEŞME",    # sözleşmesi
    "TALIMAT",     # talimatı
    "YÖNERGE",     # yönergesi                  (company-internal directives)
    "GENELGE",     # genelgesi
    "RAPOR",       # raporu / raporları
    "BELGE",       # belgesi
    "DAVRANIŞ",    # davranış(ı)                (Davranış Kuralları)
    "TEDARIKÇI",   # tedarikçi                  (Tedarikçi Davranış İlkeleri)
    "ŞARTNAME",    # şartnamesi
    "BEYANNAME",   # beyannamesi
    "TAAHHÜTNAME", # taahhütnamesi
    "KILAVUZ",     # kılavuzu
    "STANDART",    # standartları
    "KURAL",       # kuralları
    "ETIK",        # etik                       (Etik Kuralları)
    "GIZLILIK",    # gizlilik                   (Gizlilik Sözleşmesi)
    "ANLAŞMA",     # anlaşması
    "PROTOKOL",    # protokolü
    "BILDIRIM",    # bildirimi
)

_LAST_WORD_RE = re.compile(r"[A-Za-z0-9ğüşıöçĞÜŞİÖÇ]+$")


def _trim_generic_tail(text: str, start: int, end: int) -> int:
    """
    Return a new end offset for [start, end) with trailing generic
    document-title words removed. May trim the span to empty (end == start)
    when it consists entirely of generic words — the caller drops it then.
    """
    while end > start:
        m = _LAST_WORD_RE.search(text[start:end])
        if m is None:
            break
        last_word = _tr_fold(m.group(0))
        if not any(
            last_word.startswith(root) and len(last_word) - len(root) <= 6
            for root in _TRIM_TAIL_ROOTS
        ):
            break
        end = start + m.start()
        # Also drop the whitespace/punctuation that separated the trimmed word
        while end > start and not _is_word_char(text[end - 1]):
            end -= 1
    return end


# ---------------------------------------------------------------------------
# Legal-reference suppression
# ---------------------------------------------------------------------------
# WikiANN has no LAW label, so the model classifies statute/regulation names
# ("5846 sayılı Fikir ve Sanat Eserleri Kanunu") as ORG → {Company_N}.
# These are public citations, not PII — masking them destroys the legal
# context the masked text is supposed to preserve.

# Law-document keywords with Turkish consonant mutation handled at the root:
# Yönetmelik→Yönetmeliği (k→ğ), Tüzük→Tüzüğü, Tebliğ→Tebliği.
# [iİ] classes cover both "yönetmelik" and upper-cased "YÖNETMELİK" /
# Python-upper'd "YÖNETMELIK" (the dotted-İ quirk, see _BLOCKLIST_ROOTS).
_LAW_KEYWORD_PAT = (
    r"(?:Kanun|Yönetmel[iİ][kğ]|Tebl[iİ][ğg]|Genelge|Tüzü[kğ]|"
    r"Kararname|Yönerge|Mevzuat|Anayasa)"
)

# Keyword anywhere inside a span ("Türk Borçlar Kanunu'nun", "… Yönetmeliği")
_LAW_KEYWORD_IN_SPAN_RE = re.compile(
    rf"\b{_LAW_KEYWORD_PAT}[\wçğışöşüÇĞİÖŞÜ'’]*", re.IGNORECASE
)

# Keyword as the very next word after a span ("Fikir ve Sanat Eserleri" + "Kanunu")
_LAW_KEYWORD_FOLLOWS_RE = re.compile(
    rf"{_LAW_KEYWORD_PAT}[\wçğışöşüÇĞİÖŞÜ'’]*\b", re.IGNORECASE
)

# Full statute citation: "<number> sayılı <title…> Kanunu/Yönetmeliği/…"
_LAW_CITATION_RE = re.compile(
    rf"\b\d{{1,5}}\s+say[ıiİI]l[ıiİI]\s+[^\n.;:]{{0,100}}?{_LAW_KEYWORD_PAT}"
    rf"[\wçğışöşüÇĞİÖŞÜ'’]*",
    re.IGNORECASE,
)


def _legal_citation_zones(text: str) -> list[tuple[int, int]]:
    """Character ranges of '<no> sayılı … Kanunu'-style statute citations."""
    return [(m.start(), m.end()) for m in _LAW_CITATION_RE.finditer(text)]


def _is_legal_reference(
    text: str,
    start: int,
    end: int,
    span_text: str,
    label: str,
    citation_zones: list[tuple[int, int]],
    tail_trimmed: bool = False,
) -> bool:
    """
    True if the span is (part of) a statute/regulation reference, not PII.

    Rule A — the span itself contains a law keyword:
        "Fikir ve Sanat Eserleri Kanunu", "İş Sağlığı ve Güvenliği Yönetmeliği"
    Rule B — a Company span immediately followed by a law keyword:
        the model often tags only the title words and leaves the trailing
        "Kanunu" outside the span. Restricted to Company spans — a Person
        name directly followed by "Kanun…" must never be suppressed
        (suppression errs toward a PII leak; punctuation like the usual
        comma after a name already blocks this rule).
    Rule C — the span overlaps a "<no> sayılı … Kanunu" citation zone:
        catches fragments like the bare law number or inner title words.
    """
    # Rule A
    if _LAW_KEYWORD_IN_SPAN_RE.search(span_text):
        return True

    # Rule B — skipped when the generic-tail trimmer shortened this span:
    # the "following" word is then a document-title word we removed ourselves
    # ("Hava Bankası Bilgi Güvenliği" + trimmed "Yönergesi"), and suppressing
    # here would leak the company name the trim deliberately kept masked.
    if label == "Company" and not tail_trimmed:
        following = text[end: end + 40].lstrip()
        if _LAW_KEYWORD_FOLLOWS_RE.match(following):
            return True

    # Rule C
    return any(z_start < end and start < z_end for z_start, z_end in citation_zones)


# ---------------------------------------------------------------------------
# Public-institution suppression
# ---------------------------------------------------------------------------
# Courts, ministries, notaries, municipalities, directorates etc. are public
# institutions named in the document, not the PII of a data subject. The model
# (and the gazetteer) tag them as ORG; masking them removes legal/administrative
# context without protecting anyone. A span whose text ends in one of these
# institution-suffix words is suppressed.
# Turkish → ASCII fold so matching is immune to the dotted-İ / ğ / ş casing
# quirks: every accented letter maps to its ASCII base, all i-forms to I.
_ASCII_FOLD_TABLE = str.maketrans("çÇğĞıİöÖşŞüÜ", "cCgGiIoOsSuU")


def ascii_fold(s: str) -> str:
    """Uppercase ASCII fold: 'Noterliği' → 'NOTERLIGI', 'Valiliği' → 'VALILIGI'."""
    return s.translate(_ASCII_FOLD_TABLE).upper()


# Institution-suffix ROOTS (ascii_fold form). Matched by startswith() on the
# final word so Turkish case endings are covered: "Mahkemesi" / "Mahkemesine" /
# "Mahkemesinde" all share the MAHKEMES root.
_INSTITUTION_ROOTS: tuple[str, ...] = (
    "MAHKEMES", "MAHKEMELER", "ADLIYES",
    "VALILIG", "KAYMAKAMLIG", "BELEDIYES", "BUYUKSEHIR",
    "MUDURLUG", "BASKANLIG", "BAKANLIG", "MUSTESARLIG",
    "NOTERLIG", "SAVCILIG", "BASSAVCILIG", "KURUMU", "KURULU",
    "BIRLIG",            # Türkiye Noterler Birliği
    "HEYETI",            # tüketici hakem heyeti(ne)
)
# Substring phrases for multi-word public bodies / systems that don't end in a
# clean suffix word (case endings make endswith unreliable).
_INSTITUTION_PHRASES: tuple[str, ...] = (
    "KAYIT SISTEM",      # Adres Kayıt Sistemi(nde)
    "NOTERLER BIRLIG",   # Türkiye Noterler Birliği
    "HAKEM HEYET",       # tüketici hakem heyeti
    "TUKETICI MAHKEMES", # tüketici mahkemesi(ne)
)

# Final ASCII word of a folded span.
_INSTITUTION_LASTWORD = re.compile(r"[A-Z0-9]+$")


def is_institution(text: str) -> bool:
    """
    True if `text` is (ends in) a public-institution name — a court, ministry,
    municipality, notary, directorate, professional union, hakem heyeti, or a
    state system. Case-suffix tolerant ("mahkemesine", "Noterliğine").
    """
    folded = ascii_fold(text.strip())
    if any(phrase in folded for phrase in _INSTITUTION_PHRASES):
        return True
    m = _INSTITUTION_LASTWORD.search(folded)
    return bool(m and any(m.group(0).startswith(r) for r in _INSTITUTION_ROOTS))


class NerEngine:
    """
    Wraps a HuggingFace token-classification pipeline for Turkish NER.

    Model is loaded lazily on first call to detect() — importing this
    module never triggers a download or blocks startup.

    All inference runs synchronously. In async context, dispatch via:
        loop.run_in_executor(None, engine.detect, text)
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._pipeline = None
        self._load_attempted = False
        self._available = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                pipeline,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )

            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info("NER: CUDA available — running on GPU.")
            else:
                logger.info("NER: No CUDA — running on CPU (slower).")

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForTokenClassification.from_pretrained(self._model_name)

            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",  # merges B-/I- tokens into one span
                device=device,
            )
            self._available = True
            logger.info("NER model '%s' loaded successfully.", self._model_name)

        except ImportError as e:
            logger.warning(
                "NER dependencies not installed (%s). "
                "Run: pip install transformers torch  — continuing regex-only.",
                e,
            )
        except Exception as e:
            logger.warning(
                "NER model '%s' failed to load: %s — continuing regex-only.",
                self._model_name,
                e,
            )

    @property
    def available(self) -> bool:
        if not self._load_attempted:
            self._load()
        return self._available

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, text: str) -> list[Span]:
        """
        Run NER on text. Returns Span objects for PER, LOC, ORG entities.
        Long documents are processed in overlapping token windows (BERT's
        512-token limit); span offsets are always relative to the full text.
        Returns [] if the model is unavailable (graceful degradation).
        """
        if not self.available:
            return []

        assert self._pipeline is not None

        # Statute citation zones — computed once per document, used to
        # suppress law/regulation names misclassified as Company (Rule C).
        citation_zones = _legal_citation_zones(text)

        spans: list[Span] = []
        for win_start, win_end, core_start, core_end in self._window_bounds(text):
            raw: list[dict[str, Any]] = self._pipeline(text[win_start:win_end])
            spans.extend(
                self._process_entities(
                    raw, text, win_start, core_start, core_end, citation_zones
                )
            )
        return spans

    def _window_bounds(self, text: str) -> list[tuple[int, int, int, int]]:
        """
        Compute (win_start, win_end, core_start, core_end) char windows for
        chunked inference. Token-accurate when the fast tokenizer is available;
        falls back to conservative character windows otherwise.
        """
        assert self._pipeline is not None
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            try:
                return _compute_chunks(text, tokenizer)
            except Exception as e:
                logger.warning(
                    "NER: token-based chunking failed (%s) — "
                    "falling back to character windows.",
                    e,
                )
        return _char_window_chunks(text)

    def _process_entities(
        self,
        raw: list[dict[str, Any]],
        text: str,
        win_start: int,
        core_start: int,
        core_end: int,
        citation_zones: list[tuple[int, int]],
    ) -> list[Span]:
        """
        Convert raw pipeline entities from one window into Spans with offsets
        relative to the full document. Entities whose midpoint falls outside
        [core_start, core_end) belong to a neighbouring window and are skipped.
        """
        spans: list[Span] = []
        for ent in raw:
            # Strip the B-/I- prefix produced when aggregation_strategy != "simple"
            # With aggregation_strategy="simple" the entity_group is already clean (PER/LOC/ORG)
            raw_label: str = ent.get("entity_group") or ent.get("entity", "")
            raw_label = raw_label.lstrip("BI-")

            label = LABEL_MAP.get(raw_label)
            if label is None:
                continue

            score: float = float(ent["score"])
            if score < MIN_CONFIDENCE:
                continue

            word: str = ent["word"]
            start: int = int(ent["start"]) + win_start
            end: int = int(ent["end"]) + win_start

            # ── Window ownership ──────────────────────────────────────────
            # Overlapping windows both see entities in the shared margin;
            # only the window whose core region contains the entity midpoint
            # reports it, so each entity is emitted exactly once.
            midpoint = (start + end) // 2
            if not (core_start <= midpoint < core_end):
                continue

            # ── Apostrophe splitting (Turkish agglutination) ──────────────
            # e.g. "Samsun’a" → mask only "Samsun", leave "’a" as literal.
            # Both typographic (’) and straight (') apostrophes occur in
            # real documents — OCR output mostly emits the straight form.
            for apos in ("’", "'"):
                apos_pos = word.find(apos)
                if apos_pos > 0:
                    word = word[:apos_pos]
                    end = start + apos_pos
                    break

            # ── Word-boundary snapping ────────────────────────────────────
            # BERT’s BPE tokenizer sometimes returns spans whose start/end
            # offsets fall inside a word (e.g. "K|IRA|" → "IRA" as ORG,
            # leaving a bare "K" prefix). Snap the span outward to the full
            # surrounding word so we never split a token mid-character.
            start, end = _snap_word_boundaries(text, start, end)
            word = text[start:end]

            # ── Post-snap validation ──────────────────────────────────────
            stripped = word.strip()

            # Discard empty / whitespace-only spans
            if not stripped:
                continue

            # Discard single-character spans — never a meaningful entity
            if len(stripped) <= 1:
                continue

            # ── Generic-tail trimming ─────────────────────────────────────
            # "Hava Bankası Tedarikçi Davranış İlkeleri" → "Hava Bankası":
            # keep the proper name masked, leave the document title readable.
            # Must run before the blocklist/legal checks so the trimmed text
            # is what gets validated ("Banka Politikası" → "Banka" → blocked).
            tail_trimmed = False
            if label in ("Company", "Facility"):
                trimmed_end = _trim_generic_tail(text, start, end)
                if trimmed_end != end:
                    tail_trimmed = True
                    end = trimmed_end
                    word = text[start:end]
                    stripped = word.strip()
                    if len(stripped) <= 1:
                        continue   # span was entirely a document title

            # Discard blocklisted tokens — standalone generic/placeholder terms
            # (e.g. "Şirket", "Banka") that contracts use as defined aliases.
            # Multi-word proper names like "Garanti Bankası" are never equal to
            # the single-word entries in the blocklist, so they pass through.
            if _is_blocklisted(stripped):
                continue

            # Discard statute/regulation references ("… Kanunu", "… Yönetmeliği",
            # "<no> sayılı …") — public citations, not PII. See _is_legal_reference.
            if _is_legal_reference(
                text, start, end, stripped, label, citation_zones, tail_trimmed
            ):
                continue

            # Discard pure-numeric spans — regex handles those.
            # Also discard Location/Company spans that are mostly numeric
            # (e.g. "15" or "3B" being labeled as a location/org — table artefacts).
            digits_only = re.sub(r"[\s\-/.]", "", stripped)
            if digits_only.isdigit():
                continue
            if label in ("Location", "Company") and len(stripped) <= 3:
                continue

            # Apply per-label confidence floor
            label_floor = LABEL_MIN_CONFIDENCE.get(label, MIN_CONFIDENCE)
            if score < label_floor:
                continue

            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=label,
                    source="ner",
                    confidence=round(score, 4),
                    text=word,
                    canonical_id=None,
                )
            )

        return spans


# ---------------------------------------------------------------------------
# Long-document chunking
# ---------------------------------------------------------------------------

def _compute_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int = _CHUNK_MAX_TOKENS,
    overlap_tokens: int = _CHUNK_OVERLAP_TOKENS,
) -> list[tuple[int, int, int, int]]:
    """
    Split text into overlapping token windows for chunked NER inference.

    Returns a list of (win_start, win_end, core_start, core_end) character
    offsets. Windows overlap by `overlap_tokens`; the core regions tile the
    document exactly (each char position belongs to exactly one core), so an
    entity is owned by exactly one window — the one that sees it with at
    least overlap/2 tokens of context on each side.

    Tokenizes once over the full text with offset mapping (requires a fast
    tokenizer); window boundaries always fall on token boundaries, so
    re-tokenizing a window slice never exceeds max_tokens + special tokens.
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,   # suppress the >model_max_length warning — chunking handles it
    )
    # Drop zero-width mappings defensively (specials/padding map to (0, 0))
    offsets: list[tuple[int, int]] = [
        (o[0], o[1]) for o in enc["offset_mapping"] if o[1] > o[0]
    ]
    n = len(offsets)
    if n <= max_tokens:
        return [(0, len(text), 0, len(text))]

    margin = overlap_tokens // 2
    step = max_tokens - overlap_tokens
    chunks: list[tuple[int, int, int, int]] = []
    i = 0
    while True:
        j = min(i + max_tokens, n)
        win_start = 0 if i == 0 else offsets[i][0]
        win_end = len(text) if j == n else offsets[j - 1][1]
        # Core regions tile exactly: this window's core ends where the next
        # window's core starts (offsets[j - margin][0] == offsets[i' + margin][0]
        # for i' = i + step, since step + margin == max_tokens - margin).
        core_start = 0 if i == 0 else offsets[i + margin][0]
        core_end = len(text) if j == n else offsets[j - margin][0]
        chunks.append((win_start, win_end, core_start, core_end))
        if j == n:
            break
        i += step
    return chunks


def _char_window_chunks(
    text: str,
    max_chars: int = 900,
    overlap_chars: int = 180,
) -> list[tuple[int, int, int, int]]:
    """
    Fallback chunking when token offsets are unavailable: fixed character
    windows. 900 chars ≈ 250–350 BERTurk tokens for Turkish prose — safely
    under the 512-token limit. Same ownership semantics as _compute_chunks:
    the 90-char core margin exceeds any entity length, so the owning window
    always contains the entity intact even when a window edge cuts a word.
    """
    if len(text) <= max_chars:
        return [(0, len(text), 0, len(text))]

    margin = overlap_chars // 2
    step = max_chars - overlap_chars
    chunks: list[tuple[int, int, int, int]] = []
    i = 0
    while True:
        j = min(i + max_chars, len(text))
        core_start = 0 if i == 0 else i + margin
        core_end = len(text) if j == len(text) else j - margin
        chunks.append((i, j, core_start, core_end))
        if j == len(text):
            break
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Word-boundary helpers
# ---------------------------------------------------------------------------

def _is_word_char(ch: str) -> bool:
    """True if ch is a Turkish/ASCII word character (letter, digit, underscore)."""
    return bool(_WORD_CHAR.match(ch))


def _snap_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Snap a span to full word boundaries.

    If the character immediately before `start` is a word character, the span
    starts mid-word — extend left until a non-word char or the start of text.
    If the character at `end` is a word character, the span ends mid-word —
    extend right until a non-word char or the end of text.

    This corrects BERT subword-tokenizer offset artifacts like:
      original: "KİRA"   NER span: "İRA" (start=1) → snapped to "KİRA" (start=0)
      original: "TCKN"   NER span: "TCK" (end=-1)  → snapped to "TCKN" (end=len)
    """
    # Snap left
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    # Snap right
    while end < len(text) and _is_word_char(text[end]):
        end += 1
    return start, end


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

engine = NerEngine()
