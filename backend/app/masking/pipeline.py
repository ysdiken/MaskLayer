"""
MaskingPipeline: orchestrates the full PII detection and masking flow.

Flow:
  1. Regex engine   → deterministic spans (TC, IBAN, phone, etc.)
  2. NER engine     → semantic spans (Person, Company, Location, etc.)
       Both run concurrently in thread pool executors.
  3. Manual spans   → user-confirmed spans (find all occurrences of selected text).
       Always highest priority — user intent beats any detector.
  4. Span merger    → single non-overlapping list with coreference groups.
  5. Placeholder generator → typed, incrementing tokens per canonical entity.
  6. Right-to-left replacement → masked text string.
  7. Mapping dict   → { "{Person_1}": "Ahmet Yılmaz", ... } for Redis storage.

Thesis note: the `mode` parameter enables ablation experiments:
  - mode="regex"  → skip NER (regex baseline)
  - mode="ner"    → skip regex (NER baseline)
  - mode="full"   → both engines (production mode)
Manual spans are always included regardless of mode.
"""

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from app.masking.ner_engine import engine as ner_engine, _is_blocklisted, is_institution
from app.masking.gazetteer import engine as gazetteer_engine
from app.masking.regex_engine import engine as regex_engine
from app.masking import span_merger
from app.models.schemas import ManualSpanRequest, Span


# Detection-logic version — bump whenever regex/NER/filter behaviour changes.
# Stamped onto every training_examples row so the thesis can attribute each
# logged error to the pipeline that produced it and measure the effect of
# every change (FP rate per category, before vs after).
#   0.1.0 — baseline: regex engine + NER, span merger, coreference expansion
#   0.2.0 — NER chunking: overlapping token windows for >512-token documents
#   0.3.0 — legal-reference suppression, generic-tail trimming,
#           public-platform blocklist, straight-apostrophe splitting
#   0.4.0 — keyword-anchored number tags: Customer_No, SGK_No, Policy_No,
#           Contract_No, Invoice_No, Reference_No + Anne/Baba Adı person fields
#   0.4.1 — Case_No keyword form masks only the number (was over-capturing the
#           "Esas No:" label); eval-motivated fix.
#   0.5.0 — gazetteer detector (source="gazetteer"): company legal-form suffixes
#           (fixes A.Ş. boundary), Turkish name/place lists; public-institution
#           suppression (courts/ministries/notaries); Address over-capture fix.
#   0.5.1 — real-document fixes (motivated by a live bank contract): bare
#           parenthetical alias extraction ("Kredi"/"Biz"/"Siz" defined terms no
#           longer masked); institution suppression broadened (case endings +
#           Birliği/Sistemi/heyeti/tüketici mahkemesi).
#   0.6.0 — domain-adaptation FINE-TUNE of the NER model (finetune/): continued
#           training so capitalised Turkish common/domain nouns (Kredi, Platin,
#           Sözleşme) are no longer tagged ORG. Real-contract precision 0.38→0.57
#           (0.94 excl. bank info), F1 0.53→0.71; synthetic held at F1 0.997.
#   0.6.1 — gazetteer leak fixes: company suffix no longer matches mid-word
#           ("LÜTFEN AŞAĞIDAKİ"→no false "LÜTFEN AŞ"); leading initial kept
#           ("T. GARANTİ BANKASI A.Ş."); first-name + company-keyword pairs
#           ("Nur Holding") no longer tagged Person.
ENGINE_VERSION = "0.6.1"


# ---------------------------------------------------------------------------
# Contract alias extraction
# ---------------------------------------------------------------------------

# Matches the definition clause used in Turkish (and English) contracts:
#   "... Türkiye Hava Bankası A.Ş. (bundan böyle 'Şirket' olarak anılacaktır) ..."
# Captures the alias term inside quotes (or bare capitalised word after "böyle").
_ALIAS_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    # bundan böyle "X" olarak anıl…
    r'bundan\s+böyle\s+["""\'\'«»“”‘’]([^"""\'\'«»“”‘’\n]{1,60})["""\'\'«»“”‘’]\s+olarak\s+anıl',
    # (kısaca "X")
    r'kısaca\s+["""\'\'«»“”‘’]([^"""\'\'«»“”‘’\n]{1,60})["""\'\'«»“”‘’]',
    # "X" olarak anılacaktır  (no "bundan böyle" prefix)
    r'["""\'\'«»“”‘’]([^"""\'\'«»“”‘’\n]{1,60})["""\'\'«»“”‘’]\s+olarak\s+anıl',
    # English: hereinafter "X" / hereinafter referred to as "X"
    r'hereinafter(?:\s+referred\s+to\s+as)?\s+["""\'\'«»“”‘’]([^"""\'\'«»“”‘’\n]{1,60})["""\'\'«»“”‘’]',
    # bundan böyle X olarak (bare word, no quotes — must be a single Titlecase token)
    r'bundan\s+böyle\s+([A-ZÇĞİÖŞÜ][a-zçğışöşü]{1,30})\s+olarak\s+anıl',
    # Bare parenthetical definition: Taşıt Kredisi'nin ("Kredi"), … A.Ş ("Biz"),
    # müşterimiz ("Siz"). Only a quoted term with nothing but spaces between the
    # paren and the quote — so it never collides with the "(bundan böyle …)" or
    # "(kısaca …)" clauses above. These defined terms are generic, not PII.
    r'\(\s*["""\'\'«»“”‘’]([^"""\'\'«»“”‘’\n]{1,40})["""\'\'«»“”‘’]\s*\)',
]]


def _extract_contract_aliases(text: str) -> frozenset[str]:
    """
    Scan the document for alias-definition clauses and return the defined aliases
    as a frozenset of upper-cased strings (ready for blocklist comparison).

    Examples detected:
      "Türkiye Hava Bankası A.Ş. (bundan böyle 'Şirket' olarak anılacaktır)"
        → {"ŞİRKET"}
      "İstanbul Etkinlik Ltd. Şti. (kısaca 'Organizatör')"
        → {"ORGANİZATÖR"}
    """
    aliases: set[str] = set()
    for pat in _ALIAS_PATTERNS:
        for m in pat.finditer(text):
            alias = m.group(1).strip()
            if alias:
                aliases.add(alias.upper())
    return frozenset(aliases)


PipelineMode = Literal["full", "regex", "ner"]


@dataclass
class PipelineResult:
    spans: list[Span]
    masked_text: str
    mapping: dict[str, str]


class MaskingPipeline:
    def __init__(self) -> None:
        self._regex = regex_engine
        self._ner = ner_engine
        self._gazetteer = gazetteer_engine

    async def run(
        self,
        text: str,
        mode: PipelineMode = "full",
        manual_spans: list[ManualSpanRequest] | None = None,
        disabled_labels: frozenset[str] | set[str] | None = None,
    ) -> PipelineResult:
        loop = asyncio.get_event_loop()

        # ── 0. Contract alias extraction ───────────────────────────────────────
        # Detect terms defined as aliases in the document (e.g. "bundan böyle
        # 'Şirket' olarak anılacaktır"). These defined terms are generic
        # placeholders, not PII, and must survive unmasked.
        doc_aliases = _extract_contract_aliases(text)

        # ── 1. Detection (concurrent) ──────────────────────────────────────────
        # The gazetteer is a deterministic third detector; it runs only in
        # "full" mode so the regex/ner ablation arms stay pure single-detector.
        gazetteer_spans: list[Span] = []
        if mode == "full":
            regex_task = loop.run_in_executor(None, self._regex.detect, text)
            ner_task   = loop.run_in_executor(None, self._ner.detect, text)
            gaz_task   = loop.run_in_executor(None, self._gazetteer.detect, text)
            regex_spans, ner_spans, gazetteer_spans = await asyncio.gather(
                regex_task, ner_task, gaz_task
            )
        elif mode == "regex":
            regex_spans = await loop.run_in_executor(None, self._regex.detect, text)
            ner_spans = []
        else:  # "ner"
            regex_spans = []
            ner_spans = await loop.run_in_executor(None, self._ner.detect, text)

        # ── 1b. Filter NER + gazetteer spans ─────────────────────────────────
        # Remove document-defined aliases, static blocklist terms, and public
        # institutions (courts/ministries/notaries) — none of which are PII.
        if ner_spans:
            ner_spans = [s for s in ner_spans if not _is_auto_excluded(s, doc_aliases)]
        if gazetteer_spans:
            gazetteer_spans = [
                s for s in gazetteer_spans if not _is_auto_excluded(s, doc_aliases)
            ]

        # ── 2. Manual spans — find all occurrences in text ────────────────────
        manual: list[Span] = []
        for req in (manual_spans or []):
            manual.extend(_find_all_occurrences(text, req.text, req.label))

        # ── 2b. Coreference expansion ─────────────────────────────────────────
        # For any high-confidence NER entity, find ALL verbatim occurrences in
        # the text and inject spans for the ones the model missed. This ensures
        # "Mavi Ada Teknoloji A.Ş." is masked everywhere it appears, not just
        # where the NER happened to fire.
        if mode in ("full", "ner") and ner_spans:
            ner_spans = _expand_coreferences(text, ner_spans, doc_aliases)

        # ── 3. Merge: manual always beats auto-detected ────────────────────────
        if mode == "full":
            auto_merged = span_merger.merge(regex_spans, ner_spans, gazetteer_spans)
        elif mode == "regex":
            auto_merged = span_merger.regex_only(regex_spans)
        else:
            auto_merged = span_merger.ner_only(ner_spans)

        merged = span_merger.merge_with_manual(auto_merged, manual)

        # ── 3b. Fuzzy entity canonicalization ─────────────────────────────────
        # "Hava Bank ABCD", "Hava Bank", "HAVA BANK", "Hava ABCD" all refer to
        # the same entity → assign them one canonical_id so they get one
        # placeholder. Uses token overlap coefficient; threshold tunable below.
        # Also uses alias-definition clauses ("bundan böyle 'X'...") to link
        # full legal names with their in-document abbreviations.
        merged = _canonicalize_entities(merged, text=text)

        # ── 4. Output policy filter (post-detection) ───────────────────────────
        # `merged` is the COMPLETE detection — it is always returned in full and
        # is what the entities panel, audit counts, and the corrections/training
        # log see. The mask policy only decides which spans are substituted in
        # the OUTPUT text + Redis mapping. Disabling a label leaves its original
        # value in the output; it never affects detection or what we record.
        # Manual (user-confirmed) spans bypass the filter — explicit intent wins.
        disabled = disabled_labels or frozenset()
        spans_to_mask = (
            merged if not disabled
            else [s for s in merged if s.source == "manual" or s.label not in disabled]
        )

        # ── 5. Assign placeholders + build mapping ─────────────────────────────
        masked_text, mapping = _apply_masks(text, spans_to_mask)

        return PipelineResult(spans=merged, masked_text=masked_text, mapping=mapping)


# ---------------------------------------------------------------------------
# Coreference expansion
# ---------------------------------------------------------------------------

# Labels where we propagate high-confidence detections to all other occurrences.
# Location is excluded — too many false positives from expanding short place names.
_COREF_EXPAND_LABELS: frozenset[str] = frozenset({"Person", "Company", "Facility"})
_COREF_MIN_CONFIDENCE: float = 0.82   # only anchor on confident detections
_COREF_MIN_LEN: int = 4               # ignore short tokens (single-word abbreviations)


def _is_span_excluded(span_text: str, doc_aliases: frozenset[str]) -> bool:
    """True if span_text is either a static blocklist term or a doc-defined alias."""
    return _is_blocklisted(span_text) or span_text.strip().upper() in doc_aliases


# Labels for which a public-institution name (court, ministry, notary, …) is not
# PII and must not be masked.
_INSTITUTION_FILTER_LABELS: frozenset[str] = frozenset({"Company", "Location", "Facility"})


def _is_auto_excluded(span: Span, doc_aliases: frozenset[str]) -> bool:
    """
    Exclusion test for auto-detected (NER / gazetteer) spans:
    blocklist term, document alias, or a public-institution name.
    """
    if _is_span_excluded(span.text, doc_aliases):
        return True
    if span.label in _INSTITUTION_FILTER_LABELS and is_institution(span.text):
        return True
    return False


def _expand_coreferences(text: str, ner_spans: list[Span], doc_aliases: frozenset[str] = frozenset()) -> list[Span]:
    """
    For each high-confidence NER entity, find every verbatim occurrence of its
    text in the document and add a span for positions the model missed.

    Why: BERT NER runs on a sliding window of sub-tokens. The same entity name
    may appear in a table row, a footer, or a repeated clause where the model's
    confidence drops below threshold. Expansion anchors on the first clean
    detection and propagates it throughout the document.

    Confidence of injected spans is the anchor confidence × 0.95 (marked as
    slightly less certain) so they can be overridden by a regex match at the
    same position.
    """
    # Build a map: normalised_text → (label, best_confidence, canonical_text)
    anchors: dict[str, tuple[str, float, str]] = {}
    for span in ner_spans:
        if span.label not in _COREF_EXPAND_LABELS:
            continue
        if span.confidence < _COREF_MIN_CONFIDENCE:
            continue
        stripped = span.text.strip()
        if len(stripped) < _COREF_MIN_LEN:
            continue
        # Never use an alias or blocklisted term as a coreference anchor
        if _is_span_excluded(stripped, doc_aliases):
            continue
        key = stripped.lower()
        existing = anchors.get(key)
        if existing is None or span.confidence > existing[1]:
            anchors[key] = (span.label, span.confidence, stripped)

    if not anchors:
        return ner_spans

    # Record existing span positions so we don't double-add
    existing_positions: set[tuple[int, int]] = {(s.start, s.end) for s in ner_spans}

    extra: list[Span] = []
    for key, (label, conf, canonical_text) in anchors.items():
        start = 0
        while True:
            idx = text.find(canonical_text, start)
            if idx == -1:
                break
            end = idx + len(canonical_text)
            if (idx, end) not in existing_positions:
                extra.append(Span(
                    start=idx,
                    end=end,
                    label=label,
                    source="ner",
                    confidence=round(conf * 0.95, 4),
                    text=canonical_text,
                    canonical_id=None,   # merger assigns this
                ))
                existing_positions.add((idx, end))
            start = end

    return ner_spans + extra


# ---------------------------------------------------------------------------
# Manual span detection
# ---------------------------------------------------------------------------

def _find_all_occurrences(text: str, search: str, label: str) -> list[Span]:
    """
    Find every occurrence of `search` in `text` (case-sensitive).
    Returns a Span for each occurrence, all sharing the same canonical_id
    so they receive the same placeholder number throughout the document.
    """
    if not search:
        return []

    canonical = f"{label}::{search.lower().split()[0] if search else search}"
    spans: list[Span] = []
    start = 0

    while True:
        idx = text.find(search, start)
        if idx == -1:
            break
        spans.append(Span(
            start=idx,
            end=idx + len(search),
            label=label,
            source="manual",
            confidence=1.0,   # user-confirmed — maximum confidence
            text=search,
            canonical_id=canonical,
        ))
        start = idx + len(search)   # non-overlapping occurrences

    return spans


# ---------------------------------------------------------------------------
# Fuzzy entity canonicalization
# ---------------------------------------------------------------------------

# Labels where cross-span deduplication is applied.
_CANONICALIZE_LABELS: frozenset[str] = frozenset({"Company", "Person", "Facility"})

# Overlap-coefficient threshold: |A∩B| / min(|A|,|B|) ≥ this → same entity.
# 0.5 means "at least half of the shorter entity's tokens must appear in
# the longer one".  Works transitively via union-find clustering.
# Examples that now merge at 0.5:
#   "Hava Bankası" (2 tok) vs "Türkiye Hava Bankası A.Ş." (5 tok): 2/2 = 1.0 ✓
#   "Hava ABCD"   (2 tok) vs "Hava Bankası"                (2 tok): 1/2 = 0.5 ✓
_CANON_OVERLAP_THRESHOLD: float = 0.5

# Alias-definition clause — matches "(bundan böyle 'X' olarak anılacaktır)"
# Used to link a full entity name to its in-document abbreviation/alias.
_ALIAS_CLAUSE_RE: re.Pattern[str] = re.compile(
    r'\(bundan\s+böyle\s+'
    r'["""\'\'«»“”‘’]'
    r'([^“”‘’"\'«»\n]{1,80})'
    r'["""\'\'«»“”‘’]'
    r'\s+olarak\s+anıl',
    re.IGNORECASE,
)


def _entity_tokens(text: str) -> frozenset[str]:
    """Uppercase word tokens — punctuation and suffixes stripped at word level."""
    return frozenset(re.findall(r"[A-ZÇĞİÖŞÜa-zçğışöşü0-9]+", text.upper()))


def _overlap_coeff(a: frozenset[str], b: frozenset[str]) -> float:
    """Szymkiewicz–Simpson overlap coefficient: |A∩B| / min(|A|,|B|)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _canonicalize_entities(
    spans: list[Span],
    threshold: float = _CANON_OVERLAP_THRESHOLD,
    text: str = "",
) -> list[Span]:
    """
    Group spans that refer to the same real-world entity and assign them a
    shared canonical_id so they receive one placeholder throughout the doc.

    Two passes:

    Pass 1 — token overlap (per label):
      Union-Find clusters all auto-detected Company/Person/Facility spans
      whose Szymkiewicz–Simpson overlap coefficient ≥ threshold.
      Single-token entities only merge on exact token match.

    Pass 2 — alias definition linking (requires original text):
      Contracts write "Full Name (bundan böyle 'ShortName' olarak anılacaktır)".
      This clause links a full legal name to an in-document abbreviation that
      may share zero tokens with it (e.g. "Ankara Etkinlik..." ↔ "İKSV").
      We find the NER entity immediately preceding each such clause and link it
      with every span whose text matches the alias term.

    Manual spans keep whatever canonical_id was already set — user intent wins.
    """
    result = list(spans)

    # ── Pass 1: per-label token-overlap clustering ────────────────────────────
    by_label: dict[str, list[int]] = {}
    for idx, span in enumerate(result):
        if span.label in _CANONICALIZE_LABELS and span.source != "manual":
            by_label.setdefault(span.label, []).append(idx)

    for label, indices in by_label.items():
        n = len(indices)
        if n < 2:
            continue

        texts  = [result[i].text.strip() for i in indices]
        tokens = [_entity_tokens(t) for t in texts]

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                tok_i, tok_j = tokens[i], tokens[j]
                if len(tok_i) == 1 and len(tok_j) == 1:
                    if tok_i == tok_j:
                        union(i, j)
                    continue
                if _overlap_coeff(tok_i, tok_j) >= threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            if len(members) < 2:
                continue
            longest = max((texts[m] for m in members), key=len)
            cid = f"{label}::{longest.lower()}"
            for m in members:
                idx = indices[m]
                if result[idx].canonical_id is None:
                    result[idx] = result[idx].model_copy(update={"canonical_id": cid})

    # ── Pass 2: alias-definition linking ─────────────────────────────────────
    if text:
        _apply_alias_linking(result, text, threshold)

    return result


def _apply_alias_linking(spans: list[Span], text: str, threshold: float) -> None:
    """
    For each "bundan böyle 'X' olarak anılacaktır" clause in the document:
      1. Find the NER entity whose span ends closest before the clause (≤ 250 chars).
         This is the entity being defined (full legal name / long form).
      2. Split X on '/', ',' etc. to get individual alias terms.
      3. Find all other auto-detected spans of the same label whose text has
         token overlap ≥ threshold with any alias term.
      4. Merge them all into one cluster (same canonical_id).

    Modifies spans list in-place (replaces elements via model_copy).
    """
    for m in _ALIAS_CLAUSE_RE.finditer(text):
        clause_start = m.start()
        alias_raw = m.group(1).strip()

        # Split compound aliases: "İKSV /Şirket" → ["İKSV", "Şirket"]
        alias_parts: list[str] = [
            a.strip()
            for a in re.split(r'[/,]|\s+(?:ve|veya|ya da)\s+', alias_raw)
            if a.strip() and len(a.strip()) > 1
        ]

        # Find the NER span ending closest before this clause
        definer_idx: int | None = None
        definer_end = -1
        for i, span in enumerate(spans):
            if span.source == "manual" or span.label not in _CANONICALIZE_LABELS:
                continue
            if span.end <= clause_start and span.end > definer_end:
                if clause_start - span.end <= 250:
                    definer_end = span.end
                    definer_idx = i

        if definer_idx is None:
            continue

        definer_label = spans[definer_idx].label

        # Collect alias spans: auto spans of same label matching any alias term,
        # skipping generic blocklisted terms ("Şirket", "Banka", etc.)
        alias_indices: list[int] = []
        for alias in alias_parts:
            if _is_blocklisted(alias):
                continue          # "Şirket", "Banka" etc. stay unmasked
            alias_tokens = _entity_tokens(alias)
            if not alias_tokens:
                continue
            for i, span in enumerate(spans):
                if i == definer_idx:
                    continue
                if span.source == "manual" or span.label != definer_label:
                    continue
                span_tokens = _entity_tokens(span.text)
                if _overlap_coeff(span_tokens, alias_tokens) >= threshold:
                    alias_indices.append(i)

        if not alias_indices:
            continue

        # Merge definer + all alias occurrences → one canonical_id
        all_related = [definer_idx] + alias_indices
        longest = max((spans[i].text for i in all_related), key=len)
        cid = f"{definer_label}::{longest.lower()}"
        for i in all_related:
            if spans[i].canonical_id is None:
                spans[i] = spans[i].model_copy(update={"canonical_id": cid})


# ---------------------------------------------------------------------------
# Placeholder assignment (canonical_id-aware)
# ---------------------------------------------------------------------------

def _apply_masks(text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
    """
    Replace each span right-to-left with typed placeholders.
    Coreferent spans (same canonical_id) get the same placeholder number.
    Numbering is left-to-right (reading order).
    The mapping stores the LONGEST surface form seen for each placeholder
    so that demasking always restores the most complete entity name.
    """
    if not spans:
        return text, {}

    ordered = sorted(spans, key=lambda s: s.start)

    counters: defaultdict[str, int] = defaultdict(int)
    canonical_to_placeholder: dict[str, str] = {}
    placeholder_for_span: list[str] = []

    for span in ordered:
        cid = span.canonical_id or f"{span.label}::{span.text}"
        if cid not in canonical_to_placeholder:
            counters[span.label] += 1
            placeholder = f"{{{span.label}_{counters[span.label]}}}"
            canonical_to_placeholder[cid] = placeholder
        placeholder_for_span.append(canonical_to_placeholder[cid])

    # Store longest surface form per placeholder for best demasking fidelity
    mapping: dict[str, str] = {}
    for span, placeholder in zip(ordered, placeholder_for_span):
        existing = mapping.get(placeholder, "")
        if len(span.text) > len(existing):
            mapping[placeholder] = span.text

    for span, placeholder in zip(reversed(ordered), reversed(placeholder_for_span)):
        text = text[: span.start] + placeholder + text[span.end :]

    return text, mapping


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pipeline = MaskingPipeline()
