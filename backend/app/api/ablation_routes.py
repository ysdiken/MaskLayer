"""
POST /api/v1/ablation — run the same text through all three detection modes
and return a side-by-side breakdown for the thesis ablation dashboard.

Runs regex-only, NER-only, and full (regex+NER) concurrently, then computes:
  - per-mode span counts and timing
  - per-label counts per mode
  - exclusive / shared span sets (regex-only, NER-only, captured-by-both)
  - mean confidence per mode

This endpoint never stores a job or mapping — it's read-only analysis.
"""

import asyncio
import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.masking.pipeline import pipeline
from app.models.schemas import MaskRequest, Span

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class AblationRequest(BaseModel):
    text: str
    language: str = "tr"


class ModeResult(BaseModel):
    span_count: int
    processing_time_ms: float
    spans: list[Span]
    mean_confidence: float
    by_label: dict[str, int]           # label → count


class OverlapStats(BaseModel):
    regex_only_count:  int             # spans caught only by regex
    ner_only_count:    int             # spans caught only by NER
    both_count:        int             # spans caught by both (overlapping)
    regex_only_spans:  list[Span]
    ner_only_spans:    list[Span]
    both_spans:        list[Span]      # spans from 'full' that overlap with both


class AblationResponse(BaseModel):
    regex:   ModeResult
    ner:     ModeResult
    full:    ModeResult
    overlap: OverlapStats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mode_result(spans: list[Span], ms: float) -> ModeResult:
    by_label: dict[str, int] = {}
    for s in spans:
        by_label[s.label] = by_label.get(s.label, 0) + 1
    mean_conf = round(sum(s.confidence for s in spans) / len(spans), 4) if spans else 0.0
    return ModeResult(
        span_count=len(spans),
        processing_time_ms=round(ms, 2),
        spans=spans,
        mean_confidence=mean_conf,
        by_label=by_label,
    )


def _spans_overlap(a: Span, b: Span) -> bool:
    return a.start < b.end and a.end > b.start


def _compute_overlap(
    regex_spans: list[Span],
    ner_spans:   list[Span],
    full_spans:  list[Span],
) -> OverlapStats:
    """
    Classify full-mode spans as:
      - regex_only: present in regex but no NER span overlaps
      - ner_only:   present in NER but no regex span overlaps
      - both:       overlaps exist from both regex and NER detectors

    We use the full-mode span list as the ground truth positions and classify
    each span by what its source detector was.
    """
    # Regex-only: regex source spans in full that have no overlapping NER span
    regex_only: list[Span] = []
    ner_only:   list[Span] = []
    both_spans: list[Span] = []

    for span in full_spans:
        if span.source == "regex":
            # Does any NER span overlap this?
            has_ner = any(_spans_overlap(span, n) for n in ner_spans)
            if has_ner:
                both_spans.append(span)
            else:
                regex_only.append(span)
        elif span.source == "ner":
            has_regex = any(_spans_overlap(span, r) for r in regex_spans)
            if has_regex:
                both_spans.append(span)
            else:
                ner_only.append(span)
        # manual spans are excluded from overlap analysis

    return OverlapStats(
        regex_only_count=len(regex_only),
        ner_only_count=len(ner_only),
        both_count=len(both_spans),
        regex_only_spans=regex_only,
        ner_only_spans=ner_only,
        both_spans=both_spans,
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/v1/ablation", response_model=AblationResponse)
async def ablation(request: AblationRequest) -> AblationResponse:
    """
    Run regex-only, NER-only and full modes concurrently on the same text.
    Returns per-mode statistics and overlap analysis.
    """
    loop = asyncio.get_event_loop()

    async def run_mode(mode: str):
        t0 = time.perf_counter()
        result = await pipeline.run(request.text, mode=mode, manual_spans=[])  # type: ignore[arg-type]
        ms = (time.perf_counter() - t0) * 1000
        return result, ms

    (r_regex, ms_regex), (r_ner, ms_ner), (r_full, ms_full) = await asyncio.gather(
        run_mode("regex"),
        run_mode("ner"),
        run_mode("full"),
    )

    overlap = _compute_overlap(r_regex.spans, r_ner.spans, r_full.spans)

    return AblationResponse(
        regex=_mode_result(r_regex.spans, ms_regex),
        ner=_mode_result(r_ner.spans,   ms_ner),
        full=_mode_result(r_full.spans,  ms_full),
        overlap=overlap,
    )
