"""
Span merger: combines regex and NER spans into a single non-overlapping list.

Merging rules (per spec):
  1. Regex wins on deterministic format types (TC, IBAN, VKN, Card, Phone,
     Plate, Email, Passport) — these are structurally validated and have
     near-zero false-positive rate after checksum. NER should never override
     a validated regex match.
  2. On overlap between two NER spans: longest span wins.
  3. On overlap between a non-deterministic regex span and an NER span:
     longest span wins; higher confidence breaks ties.
  4. Containment → outer span wins (consistent with regex engine behaviour).
  5. Coreference grouping: assign the same canonical_id to all spans whose
     normalised text is identical within the document. This ensures
     "Ahmet Yılmaz" always becomes {Person_1} throughout, not {Person_1},
     {Person_2}, etc.

Thesis note: the merge strategy is an ablation variable. You can measure:
  - regex-only F1  (disable NER, only step 1)
  - NER-only F1    (disable regex, only steps 2-3)
  - merged F1      (all steps)
  Comparing these three across entity types is the primary experimental
  contribution of the thesis.
"""

from collections import defaultdict

from app.models.schemas import Span


# ---------------------------------------------------------------------------
# Labels where a validated regex match always beats NER
# ---------------------------------------------------------------------------

REGEX_DOMINANT_LABELS: frozenset[str] = frozenset({
    "TC_No",
    "IBAN",
    "Tax_No",
    "Card_No",
    "Phone_No",
    "License_Plate",
    "Email",
    "Passport_No",
    "Money_Amount",
    "Account_No",
    "IP_Address",
    "URL",
    "Case_No",
    "Date",
    # Keyword-anchored document numbers — deterministic when the Turkish
    # label precedes them ("Müşteri No:", "Poliçe No:", …)
    "Customer_No",
    "SGK_No",
    "Policy_No",
    "Contract_No",
    "Invoice_No",
    "Reference_No",
})


def merge(
    regex_spans: list[Span],
    ner_spans: list[Span],
    gazetteer_spans: list[Span] | None = None,
) -> list[Span]:
    """
    Merge regex, NER, and gazetteer span lists into a single non-overlapping
    list, sorted by start offset ascending.

    Steps:
      1. Tag each span with its origin for priority decisions.
      2. Pool all spans and apply priority-aware overlap resolution.
      3. Assign canonical_ids for coreference grouping.

    The gazetteer is a deterministic third detector; on overlap it competes by
    length like NER (so its full "X A.Ş." boundary overrides NER's truncated
    "X A.Ş"), and outranks NER only on exact ties (deterministic > probabilistic).
    """
    all_spans = list(regex_spans) + list(ner_spans) + list(gazetteer_spans or [])
    deduplicated = _resolve_overlaps(all_spans)
    return _assign_canonical_ids(deduplicated)


def merge_with_manual(auto_spans: list[Span], manual_spans: list[Span]) -> list[Span]:
    """
    Combine auto-detected spans with user-confirmed manual spans.

    Manual spans always win over auto-detected spans on overlap — the user
    has explicitly chosen the label and text, so their intent is authoritative.
    Manual spans already carry canonical_ids from _find_all_occurrences, so
    we skip _assign_canonical_ids for them (it would overwrite the shared id).
    """
    if not manual_spans:
        return auto_spans

    # Remove any auto span that overlaps with a manual span
    manual_regions = [(s.start, s.end) for s in manual_spans]
    filtered_auto = [
        s for s in auto_spans
        if not any(s.start < me and s.end > ms for ms, me in manual_regions)
    ]

    combined = sorted(filtered_auto + manual_spans, key=lambda s: s.start)
    return combined


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """
    Greedy interval scheduling with priority-aware tie-breaking.

    Sort order (descending priority):
      1. Regex-dominant label wins unconditionally over NER for same region.
      2. Longest span wins (larger span = more context captured).
      3. Higher confidence breaks equal-length ties.
      4. Regex source beats NER source for equal length and confidence
         (deterministic > probabilistic).
    """
    if not spans:
        return []

    def sort_key(s: Span) -> tuple:
        is_manual   = s.source == "manual"
        is_dominant = s.label in REGEX_DOMINANT_LABELS and s.source == "regex"
        return (
            int(is_manual),          # user-confirmed always first (2 > 1 > 0)
            int(is_dominant),        # dominant regex second
            s.end - s.start,         # longer spans next
            s.confidence,            # higher confidence
            # deterministic sources (regex/gazetteer) beat NER on exact ties
            int(s.source in ("regex", "gazetteer")),
        )

    candidates = sorted(spans, key=sort_key, reverse=True)
    accepted: list[Span] = []
    occupied: list[tuple[int, int]] = []

    for span in candidates:
        overlaps = any(
            span.start < end and span.end > start
            for start, end in occupied
        )
        if not overlaps:
            accepted.append(span)
            occupied.append((span.start, span.end))

    return sorted(accepted, key=lambda s: s.start)


# ---------------------------------------------------------------------------
# Coreference grouping
# ---------------------------------------------------------------------------

def _assign_canonical_ids(spans: list[Span]) -> list[Span]:
    """
    Group spans that refer to the same real-world entity by assigning a
    shared canonical_id.

    Two spans are considered coreferent if they have the same label AND their
    normalised text matches (case-insensitive, whitespace-collapsed). This
    handles:
      - "Ahmet Yılmaz" appearing multiple times → all {Person_1}
      - "ahmet yılmaz" (different capitalisation) → same group
      - "TR33 0006..." and "TR330006..." (spacing variant) → same IBAN group

    canonical_id format: "{label}::{normalised_text}"
    e.g. "Person::ahmet yılmaz", "TC_No::10000000146"

    The placeholder generator (in pipeline.py) uses canonical_id so that the
    first occurrence sets the counter and all later occurrences reuse it.
    """
    # normalised_text → canonical_id
    seen: dict[str, str] = {}
    result: list[Span] = []

    for span in spans:
        key = f"{span.label}::{_normalise(span.text)}"
        if key not in seen:
            seen[key] = key   # canonical_id IS the normalised key (stable, human-readable)
        result.append(span.model_copy(update={"canonical_id": seen[key]}))

    return result


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip spaces — for coreference matching."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Utilities for ablation experiments (thesis)
# ---------------------------------------------------------------------------

def regex_only(regex_spans: list[Span]) -> list[Span]:
    """Return only regex spans (for ablation baseline)."""
    from app.masking.regex_engine import _remove_overlaps  # avoid circular import at module level
    deduped = _remove_overlaps(regex_spans)
    return _assign_canonical_ids(deduped)


def ner_only(ner_spans: list[Span]) -> list[Span]:
    """Return only NER spans (for ablation baseline)."""
    from app.masking.regex_engine import _remove_overlaps
    deduped = _remove_overlaps(ner_spans)
    return _assign_canonical_ids(deduped)
