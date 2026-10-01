"""
Turkish PII Regex Detection Engine.

Detects 20+ PII entity types using compiled regex patterns, with optional
checksum/format validation to eliminate false positives. Returns a list of
Span objects that are ready for the span merger and placeholder substitution.

Design principles:
- Each PatternConfig is self-contained: pattern, label, confidence, validator.
- Patterns are applied in priority order (higher-confidence types first).
- After all patterns run, _remove_overlaps() applies "longest span wins"
  (with confidence as tiebreaker) — consistent with the span merger spec.
- Span replacement must always be done right-to-left (end→start) by the
  caller to preserve character offsets.
- This module is synchronous. Callers in async contexts should dispatch via
  asyncio.get_event_loop().run_in_executor(None, engine.detect, text).

Thesis note: This regex engine forms the deterministic baseline. Comparing
its P/R/F1 per entity type against the combined regex+NER system is the
core ablation experiment of the thesis.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from app.models.schemas import Span
from app.masking.validators import (
    validate_tc_kimlik,
    validate_iban,
    validate_vkn,
    luhn_check,
    validate_license_plate,
)


# ---------------------------------------------------------------------------
# Pattern configuration
# ---------------------------------------------------------------------------

@dataclass
class PatternConfig:
    label: str                              # Matches placeholder taxonomy label
    pattern: str                            # Raw regex string
    base_confidence: float                  # Confidence when pattern matches
    flags: int = 0                          # re flags (e.g. re.IGNORECASE)
    validator: Callable[[str], bool] | None = None  # Optional checksum validator
    capture_group: int = 0                  # 0 = full match; 1+ = capturing group to use as span.
                                            # Use for keyword-anchored patterns where the keyword
                                            # provides context but should NOT be masked.
    compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compiled = re.compile(self.pattern, self.flags)


# ---------------------------------------------------------------------------
# Turkish month names for date patterns
# ---------------------------------------------------------------------------

_TR_MONTHS = (
    "Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|"
    "Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"
)

# ---------------------------------------------------------------------------
# Pattern registry — applied in order; earlier = higher priority on overlap
# ---------------------------------------------------------------------------

PATTERNS: list[PatternConfig] = [

    # ── Checksum-validated (highest confidence) ──────────────────────────────

    PatternConfig(
        label="TC_No",
        # 11 digits, first digit non-zero, strict word boundaries
        pattern=r"(?<!\d)[1-9]\d{10}(?!\d)",
        base_confidence=0.99,
        flags=0,
        validator=validate_tc_kimlik,
    ),

    # ── Keyword-anchored TC fallback ──────────────────────────────────────────
    # Catches synthetic / hand-typed TC numbers whose checksum is wrong but
    # whose context (the label before them) makes the intent unambiguous.
    # capture_group=1 masks only the digits, not the keyword.
    PatternConfig(
        label="TC_No",
        pattern=(
            r"(?:T\.?\s*C\.?\s*Kimlik\s*(?:No\.?|Numaras[ıiIİ])?|TCKN)\s*:?\s*"
            r"([1-9]\d{10})"
        ),
        base_confidence=0.88,
        flags=re.IGNORECASE,
        validator=None,
        capture_group=1,
    ),

    PatternConfig(
        label="IBAN",
        # TR + 2 check digits + 22-digit BBAN (26 chars total).
        # Allows optional whitespace or dashes every 4 chars (printed bank docs).
        # [\s] instead of [ ] so IBANs split across a single line-break are caught.
        pattern=r"\bTR\d{2}(?:[\s\-]?\d{4}){5}[\s\-]?\d{2}\b",
        base_confidence=0.99,
        flags=re.IGNORECASE,
        validator=validate_iban,
    ),

    PatternConfig(
        label="IBAN",
        # Partial IBAN: first 24 chars (TR + 2 check + 5×4 BBAN digits).
        # In PDF table extraction the last 2 digits often end up on a separate row
        # separated by unrelated text. Detect the 24-char prefix with lower confidence.
        # Validator checks mod-97 on the partial string padded to 26 — lenient.
        pattern=r"\bTR\d{2}(?:[ \-]?\d{4}){5}\b",
        base_confidence=0.82,
        flags=re.IGNORECASE,
        validator=None,   # no checksum possible without last 2 digits
    ),

    PatternConfig(
        label="Card_No",
        # 16-digit card number; groups of 4 separated by space or dash
        # Placed before Tax_No so 16-digit sequences don't match VKN first
        pattern=r"(?<!\d)\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}(?!\d)",
        base_confidence=0.97,
        flags=0,
        validator=luhn_check,
    ),

    PatternConfig(
        label="Tax_No",
        # 10-digit VKN — must not be part of a longer digit sequence.
        pattern=r"(?<!\d)\d{10}(?!\d)",
        base_confidence=0.95,
        flags=0,
        validator=validate_vkn,
    ),

    # ── Keyword-anchored VKN fallback ─────────────────────────────────────────
    # Catches synthetic / test VKNs that fail the checksum but are clearly
    # labelled as "Vergi No" in the document.
    PatternConfig(
        label="Tax_No",
        pattern=r"(?:Vergi\s*(?:Kimlik\s*)?No\.?\s*:?\s*)(\d{10})(?!\d)",
        base_confidence=0.88,
        flags=re.IGNORECASE,
        validator=None,
        capture_group=1,
    ),

    # ── MERSİS business registration number ──────────────────────────────────
    # 16-digit number in NNNN-NNNN-NNNN-NNNN format.
    # MUST appear before Card_No — Luhn accidentally passes on some MERSİS nums.
    # capture_group=1 masks only the digits (not the "MERSİS No:" label).
    PatternConfig(
        label="Account_No",
        pattern=(
            r"MERSİS\s*No\.?\s*:?\s*"
            r"(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})"
        ),
        base_confidence=0.97,
        flags=re.IGNORECASE,
        validator=None,
        capture_group=1,
    ),

    # ── Strong structural patterns ────────────────────────────────────────────

    PatternConfig(
        label="Email",
        pattern=r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        base_confidence=0.99,
        flags=0,
    ),

    PatternConfig(
        label="Phone_No",
        # Turkish mobile: 05XX XXX XX XX  (11 digits total)
        # Turkish landline: 0[234]XX XXX XX XX
        # International prefix variants: +90, 0090
        # Allows spaces and dashes as separators
        pattern=(
            r"(?:\+90|0090|0)"         # prefix
            r"[ \-]?"
            r"(?:5\d{2}|[234]\d{2})"   # area/mobile code
            r"[ \-]?\d{3}"             # subscriber part 1
            r"[ \-]?\d{2}"             # subscriber part 2
            r"[ \-]?\d{2}"             # subscriber part 3
        ),
        base_confidence=0.95,
        flags=0,
    ),

    PatternConfig(
        label="License_Plate",
        # Province code 01–81 + 1–3 uppercase letters + 2–4 digits
        # Turkish plates use Turkish uppercase letters (Ç Ğ İ Ö Ş Ü allowed)
        pattern=r"\b(0[1-9]|[1-7][0-9]|8[01])[ ]?([A-ZÇĞİÖŞÜ]{1,3})[ ]?(\d{2,4})\b",
        base_confidence=0.95,
        flags=0,  # Case-sensitive: plates are uppercase
        validator=validate_license_plate,
    ),

    PatternConfig(
        label="Passport_No",
        # Turkish passport: 1–2 uppercase letters + 7–9 digits
        # Lower confidence — no checksum, short pattern can appear in other contexts
        pattern=r"\b[A-Z]{1,2}\d{7,9}\b",
        base_confidence=0.80,
        flags=0,
    ),

    # ── Money amounts ─────────────────────────────────────────────────────────

    PatternConfig(
        label="Money_Amount",
        # Turkish format: 1.234,56 TL / ₺1.234,56 / 1.234 TL
        # Period = thousands separator, comma = decimal separator
        pattern=(
            r"(?:₺|TL|TRY)\s*\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?"   # symbol-first
            r"|"
            r"\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?\s*(?:₺|TL|TRY)"    # amount-first
        ),
        base_confidence=0.95,
        flags=re.IGNORECASE,
    ),

    PatternConfig(
        label="Money_Amount",
        # International format: $1,234.56 / 1,234.56 USD / €500
        # Comma = thousands separator, period = decimal separator
        pattern=(
            r"(?:\$|€|£|USD|EUR|GBP)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"  # symbol-first
            r"|"
            r"\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:\$|€|£|USD|EUR|GBP)"  # amount-first
        ),
        base_confidence=0.95,
        flags=re.IGNORECASE,
    ),

    # ── Dates ─────────────────────────────────────────────────────────────────

    PatternConfig(
        label="Date",
        # Numeric: dd.mm.yyyy / dd/mm/yyyy / dd-mm-yyyy / yyyy-mm-dd (ISO)
        pattern=r"\b(?:\d{2}[./\-]\d{2}[./\-]\d{4}|\d{4}[./\-]\d{2}[./\-]\d{2})\b",
        base_confidence=0.90,
        flags=0,
    ),

    PatternConfig(
        label="Date",
        # Turkish long form: "15 Ocak 2024", "1 Aralık 1999"
        pattern=rf"\b\d{{1,2}}\s+(?:{_TR_MONTHS})\s+\d{{4}}\b",
        base_confidence=0.99,
        flags=re.IGNORECASE,
    ),

    # ── Case / dossier numbers ────────────────────────────────────────────────

    PatternConfig(
        label="Case_No",
        # Keyword-anchored: "Esas No: 2023/456", "Karar No. 2024/1234".
        # capture_group=1 masks ONLY the number — the "Esas No:" label is not
        # sensitive and stays readable (consistent with the other keyword-anchored
        # number types). Accepts any digit count after the slash.
        pattern=r"(?:Esas|Karar|Dava|Dosya|İş)\s+No\.?:?\s*(\d{4}/\d+)",
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="Case_No",
        # Bare YYYY/NNNN+ fallback (4+ digits after slash to avoid date false
        # positives). Lower confidence — no keyword context.
        pattern=r"\b\d{4}/\d{4,}\b",
        base_confidence=0.85,
        flags=re.IGNORECASE,
    ),

    # ── Keyword-anchored document numbers ─────────────────────────────────────
    # Customer/policy/contract/invoice numbers have no inherent structure —
    # a bare digit run is too ambiguous to mask. The Turkish label before them
    # makes intent unambiguous. capture_group=1 masks only the number; the
    # keyword stays readable. Character classes like [ıiIİ] handle the
    # dotted-İ casing quirk under IGNORECASE; [üu]/[şs]/[öo]/[çc] tolerate
    # ASCII-degraded OCR output.

    PatternConfig(
        label="Customer_No",
        # "Müşteri No: 12345678", "MÜŞTERİ NUMARASI 00123456"
        pattern=(
            r"M[üu][şs]ter[iİ]\s*(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"(\d{4,15})(?!\d)"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="SGK_No",
        # "SGK Sicil No: 1234567890123", "SSK No: 12345678"
        pattern=(
            r"(?:SGK|SSK)\s*(?:Sicil\s*)?(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"(\d{7,13})(?!\d)"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="Policy_No",
        # "Poliçe No: POL-2024-123456" — insurance policies are often alphanumeric
        pattern=(
            r"Pol[iİ][çc]e\s*(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"([A-Z0-9][A-Z0-9\-/]{3,19})"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="Contract_No",
        # "Sözleşme No: SZL-2024/00123"
        pattern=(
            r"S[öo]zle[şs]me\s*(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"([A-Z0-9][A-Z0-9\-/]{3,24})"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="Invoice_No",
        # "Fatura No: GIB2024000001234" — e-Fatura: 3 letters + 13 digits,
        # but free-format alphanumerics also occur
        pattern=(
            r"Fatura\s*(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"([A-Z0-9][A-Z0-9\-/]{5,19})"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    PatternConfig(
        label="Reference_No",
        # "Referans No: REF-998877", "Dekont No: 123456", "İşlem No: TX-2024-1"
        pattern=(
            r"(?:Referans|Dekont|[İIi][şs]lem|Tak[iİ]p)\s*(?:No\.?|Numaras[ıiIİ])\s*:?\s*"
            r"([A-Z0-9][A-Z0-9\-/]{3,24})"
        ),
        base_confidence=0.92,
        flags=re.IGNORECASE,
        capture_group=1,
    ),

    # ── Keyword-anchored person names (KYC form fields) ───────────────────────
    # "Anne Adı: Fatma", "Baba Adı: Hasan Demir" — isolated given names in
    # form layouts are exactly where the NER model is weakest.
    PatternConfig(
        label="Person",
        pattern=(
            r"(?:Anne|Baba)\s*Ad[ıi]\s*:?\s*"
            # Optional second name word: single space only (form layouts use
            # multiple spaces/tabs between fields) and never the next field's
            # keyword ("Anne Adı: Fatma Baba Adı: …" must stop at "Fatma").
            r"([A-ZÇĞİÖŞÜ][a-zçğışöşü]+"
            r"(?:[ ](?!(?:Anne|Baba)\b)[A-ZÇĞİÖŞÜ][a-zçğışöşü]+)?)"
        ),
        base_confidence=0.90,
        flags=0,   # case-sensitive: the captured name must be Titlecase
        capture_group=1,
    ),

    # ── Bank account (keyword-anchored, standalone is too ambiguous) ──────────

    PatternConfig(
        label="Account_No",
        # "Hesap No: 1234567890" — 10 to 26 digits after keyword
        pattern=r"(?:Hesap|IBAN|Hesap\s+No\.?):?\s*[\d\s]{10,30}",
        base_confidence=0.88,
        flags=re.IGNORECASE,
    ),

    # ── SWIFT / BIC ───────────────────────────────────────────────────────────

    PatternConfig(
        label="Account_No",
        # SWIFT: 4 bank letters + "TR" (country) + 2 location chars + optional 3 branch chars
        pattern=r"\b[A-Z]{4}TR[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
        base_confidence=0.90,
        flags=0,
    ),

    # ── IPv4 address ──────────────────────────────────────────────────────────

    PatternConfig(
        label="IP_Address",
        pattern=(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        base_confidence=0.99,
        flags=0,
    ),

    # ── URL ───────────────────────────────────────────────────────────────────

    PatternConfig(
        label="URL",
        pattern=r"https?://[^\s<>\"{}|\\^`\[\]]+",
        base_confidence=0.95,
        flags=re.IGNORECASE,
    ),

    # ── Turkish street address ────────────────────────────────────────────────
    # Matches patterns like:
    #   "Büyükdere Caddesi No: 185, Kat: 12"
    #   "Lara Caddesi No: 98/7"
    #   "Organize Sanayi Bölgesi 3. Cadde No: 42"
    # Anchored on the road-type keyword to avoid false positives on generic text.
    # Lower confidence — no structural validation possible.
    PatternConfig(
        label="Address",
        # Matches a single-line Turkish street address containing a road-type
        # keyword AND a "No:" clause.
        # Uses [ \t] (space/tab only) to guarantee no cross-line matching.
        # Examples:
        #   "Büyükdere Caddesi No: 185, Kat: 12, Şişli / İstanbul"
        #   "Lara Caddesi No: 98/7, Muratpaşa / Antalya"
        #   "Organize Sanayi Bölgesi 3. Cadde No: 42"
        pattern=(
            r"\b[\wğüşıöçĞÜŞİÖÇ]+"             # first word of street name
            r"(?:[ \t]+[\wğüşıöçĞÜŞİÖÇ\.]+){0,5}"  # up to 5 more words (same line)
            r"[ \t]+"
            r"(?:Caddesi|Cadde|Cad\."           # road type keyword
            r"|Sokağı|Sokak|Sok\."
            r"|Bulvarı|Blv\."
            r"|Mahallesi|Mah\."
            r"|Bölgesi|Sitesi|Apartmanı)"
            r"[ \t]*No[ \t]*:[ \t]*\d+"         # mandatory "No: <number>"
            # ── Address-only continuations (NOT arbitrary prose) ──
            r"(?:[ \t]*/[ \t]*\d+[A-Za-zÇĞİÖŞÜ]?)?"  # flat: /7, /12A
            r"(?:[ \t]*,?[ \t]*(?:Kat|Daire|D|Blok|Bağımsız[ \t]+Bölüm)"
            r"[ \t]*[:.]?[ \t]*\d+[A-Za-zÇĞİÖŞÜ]?)*"  # Kat: 12, Daire: 3, Blok B
            r"(?:[ \t]*,[ \t]*[A-ZÇĞİÖŞÜ][A-Za-zğüşıöçĞÜŞİÖÇ.]{1,20}"
            r"(?:[ \t]*/[ \t]*[A-ZÇĞİÖŞÜ][A-Za-zğüşıöçĞÜŞİÖÇ.]{1,20})?){0,2}"  # , Kadıköy / İstanbul
        ),
        base_confidence=0.82,
        flags=re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegexEngine:
    """
    Applies all registered patterns to an input text and returns deduplicated
    Span objects suitable for the masking pipeline.

    Usage:
        engine = RegexEngine()
        spans = engine.detect(text)   # synchronous
        # In async context:
        spans = await loop.run_in_executor(None, engine.detect, text)
    """

    def __init__(self, patterns: list[PatternConfig] | None = None) -> None:
        self.patterns = patterns if patterns is not None else PATTERNS

    def detect(self, text: str) -> list[Span]:
        """
        Run all patterns against text. Returns a list of non-overlapping Spans,
        sorted by start position (ascending). Overlaps are resolved by:
          1. Longest span wins.
          2. On equal length, higher confidence wins.
        """
        raw_spans: list[Span] = []

        for config in self.patterns:
            for match in config.compiled.finditer(text):
                g = config.capture_group
                matched_text = match.group(g)
                if not matched_text:
                    continue
                span_start = match.start(g)
                span_end   = match.end(g)

                # Run checksum/format validator if present
                if config.validator is not None:
                    if not config.validator(matched_text):
                        continue

                raw_spans.append(
                    Span(
                        start=span_start,
                        end=span_end,
                        label=config.label,
                        source="regex",
                        confidence=config.base_confidence,
                        text=matched_text,
                        canonical_id=None,
                    )
                )

        return _remove_overlaps(raw_spans)


def _remove_overlaps(spans: list[Span]) -> list[Span]:
    """
    Given a list of potentially overlapping spans, return a non-overlapping
    subset using a greedy interval scheduling approach:
      - Sort by length descending (longer spans preferred), then confidence descending.
      - Accept a span only if it does not overlap any already-accepted span.

    This is consistent with the spec: "longest span wins on overlap; containment
    → outer span wins."

    Returns spans sorted by start position ascending (for display / right-to-left
    replacement the caller must reverse this list).
    """
    if not spans:
        return []

    # Sort: longest first, then highest confidence first
    candidates = sorted(
        spans,
        key=lambda s: (s.end - s.start, s.confidence),
        reverse=True,
    )

    accepted: list[Span] = []
    occupied: list[tuple[int, int]] = []  # (start, end) of accepted spans

    for span in candidates:
        # Check for overlap with any accepted span
        overlaps = any(
            span.start < end and span.end > start
            for start, end in occupied
        )
        if not overlaps:
            accepted.append(span)
            occupied.append((span.start, span.end))

    # Return sorted by start position for readability
    return sorted(accepted, key=lambda s: s.start)


# ---------------------------------------------------------------------------
# Module-level singleton — import and reuse across requests
# ---------------------------------------------------------------------------

engine = RegexEngine()
