"""
Audit service — writes to jobs and audit_log tables.

All functions are async and designed to be fire-and-forget from route
handlers (use asyncio.create_task() to avoid blocking the response).
Failures are logged but never re-raised — the masking operation must
succeed even if the DB is temporarily unavailable.

KVKK compliance: every call to record_mask() / record_unmask() creates
an audit_log row that proves the event occurred, who triggered it, and when.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import hashlib

from sqlalchemy import insert, select, func, update

from app.db.schema import audit_log, corrections, jobs, training_examples
from app.db.session import get_engine
from app.masking.pipeline import ENGINE_VERSION, PipelineResult

logger = logging.getLogger(__name__)

# Match the Redis TTL default in MappingStore
DEFAULT_TTL_SECONDS = 86_400


# ---------------------------------------------------------------------------
# jobs table helpers
# ---------------------------------------------------------------------------

async def record_mask(
    job_id: str,
    pipeline_result: PipelineResult,
    *,
    filename: str | None = None,
    file_type: str | None = None,
    page_count: int | None = None,
    extraction_methods: set[str] | None = None,
    ocr_mean_confidence: float | None = None,
    mode: str = "full",
    processing_time_ms: float = 0.0,
    actor: str = "system",
    ip_address: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Insert a row into jobs and a 'mask' event into audit_log.
    Called after every successful POST /api/v1/mask or /api/v1/upload.
    """
    spans = pipeline_result.spans
    regex_count     = sum(1 for s in spans if s.source == "regex")
    ner_count       = sum(1 for s in spans if s.source == "ner")
    manual_count    = sum(1 for s in spans if s.source == "manual")
    gazetteer_count = sum(1 for s in spans if s.source == "gazetteer")

    # Span summary: placeholder text, not original PII
    mapping_inv = {v: k for k, v in pipeline_result.mapping.items()}
    spans_summary = [
        {
            "label":      s.label,
            "source":     s.source,
            "confidence": round(s.confidence, 4),
            "placeholder": mapping_inv.get(s.text, ""),
        }
        for s in spans
    ]

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    job_row = {
        "job_id":              uuid.UUID(job_id),
        "status":              "complete",
        "filename":            filename,
        "file_type":           file_type,
        "page_count":          page_count,
        "extraction_methods":  ",".join(sorted(extraction_methods)) if extraction_methods else None,
        "ocr_mean_confidence": ocr_mean_confidence,
        "mode":                mode,
        "span_count":          len(spans),
        "regex_spans":         regex_count,
        "ner_spans":           ner_count,
        "manual_spans":        manual_count,
        "gazetteer_spans":     gazetteer_count,
        "char_count":          len(pipeline_result.masked_text),
        "processing_time_ms":  processing_time_ms,
        "mapping_expires_at":  expires_at,
        "spans_summary":       spans_summary,
    }

    audit_row = {
        "event_type": "mask",
        "job_id":     uuid.UUID(job_id),
        "actor":      actor,
        "ip_address": ip_address,
        "outcome":    "success",
        "details": {
            "span_count":     len(spans),
            "mode":           mode,
            "filename":       filename,
            "has_ocr_pages":  bool(extraction_methods and "ocr" in extraction_methods),
        },
    }

    await _write(job_row, audit_row)


async def record_unmask(
    job_id: str,
    placeholder_count: int,
    *,
    actor: str = "system",
    ip_address: str | None = None,
) -> None:
    """
    Insert an 'unmask' event into audit_log.
    Called after every successful POST /api/v1/unmask.
    This is the KVKK Article 11 access record.
    """
    audit_row = {
        "event_type": "unmask",
        "job_id":     uuid.UUID(job_id),
        "actor":      actor,
        "ip_address": ip_address,
        "outcome":    "success",
        "details": {
            "placeholder_count": placeholder_count,
        },
    }
    await _write_audit(audit_row)


async def record_correction(
    job_id: str | None,
    event_type: str,          # false_positive | false_negative | relabel
    entity_label: str,
    entity_text: str,
    *,
    detector_source: str | None = None,
    corrected_label: str | None = None,
    confidence: float | None = None,
    context: str | None = None,
    ip_address: str | None = None,
) -> None:
    """
    Log one analyst correction.

    Writes two rows in one transaction:
      corrections        — hashed text (KVKK-safe analytics, threshold tuning)
      training_examples  — RAW text + engine_version (fine-tuning data and
                           per-version error tracking for the thesis; this
                           table never leaves the trust boundary)
    """
    text_hash = hashlib.sha256(entity_text.encode("utf-8")).hexdigest()
    row = {
        "job_id":          uuid.UUID(job_id) if job_id else None,
        "event_type":      event_type,
        "detector_source": detector_source,
        "entity_label":    entity_label,
        "corrected_label": corrected_label,
        "confidence":      confidence,
        "text_hash":       text_hash,
        "ip_address":      ip_address,
    }
    training_row = {
        "job_id":          uuid.UUID(job_id) if job_id else None,
        "event_type":      event_type,
        "detector_source": detector_source,
        "entity_label":    entity_label,
        "corrected_label": corrected_label,
        "confidence":      confidence,
        "entity_text":     entity_text,
        "context":         context,
        "engine_version":  ENGINE_VERSION,
    }
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(insert(corrections).values(**row))
            await conn.execute(insert(training_examples).values(**training_row))
        # Also write to audit_log for KVKK completeness
        await _write_audit({
            "event_type": "manual_correction",
            "job_id":     uuid.UUID(job_id) if job_id else None,
            "actor":      "system",
            "ip_address": ip_address,
            "outcome":    "success",
            "details": {
                "correction_type":  event_type,
                "entity_label":     entity_label,
                "corrected_label":  corrected_label,
                "detector_source":  detector_source,
            },
        })
    except Exception as e:
        logger.error("audit.record_correction failed: %s", e)


async def get_correction_stats() -> dict:
    """
    Aggregate correction counts for the active-learning dashboard.
    Returns per-label and per-source breakdowns, plus confidence percentiles
    and suggested threshold adjustments derived from recorded false-positives.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            # ── Count aggregates ──────────────────────────────────────────────
            count_rows = await conn.execute(
                select(
                    corrections.c.event_type,
                    corrections.c.entity_label,
                    corrections.c.detector_source,
                    func.count().label("n"),
                ).group_by(
                    corrections.c.event_type,
                    corrections.c.entity_label,
                    corrections.c.detector_source,
                )
            )
            records = count_rows.fetchall()

            # ── Confidence values for false-positives (for threshold tuning) ──
            fp_conf_rows = await conn.execute(
                select(
                    corrections.c.entity_label,
                    corrections.c.detector_source,
                    corrections.c.confidence,
                ).where(
                    corrections.c.event_type == "false_positive",
                    corrections.c.confidence.isnot(None),
                )
            )
            fp_confs: dict[str, list[float]] = {}
            for row in fp_conf_rows.fetchall():
                key = f"{row.entity_label}::{row.detector_source or 'any'}"
                fp_confs.setdefault(key, []).append(row.confidence)

            # ── Repeated entity hashes — systematic FP patterns ───────────────
            # Entities flagged as FP multiple times (same text_hash) are strong
            # candidates for the blocklist or a threshold increase.
            repeated_rows = await conn.execute(
                select(
                    corrections.c.text_hash,
                    corrections.c.entity_label,
                    corrections.c.detector_source,
                    func.count().label("n"),
                ).where(
                    corrections.c.event_type == "false_positive",
                    corrections.c.text_hash.isnot(None),
                ).group_by(
                    corrections.c.text_hash,
                    corrections.c.entity_label,
                    corrections.c.detector_source,
                ).having(func.count() >= 2)
                 .order_by(func.count().desc())
                 .limit(20)
            )
            repeated_fp = [
                {
                    "text_hash":       r.text_hash,
                    "entity_label":    r.entity_label,
                    "detector_source": r.detector_source,
                    "count":           r.n,
                }
                for r in repeated_rows.fetchall()
            ]

            # ── Recent corrections (last 20) ──────────────────────────────────
            recent_rows = await conn.execute(
                select(
                    corrections.c.occurred_at,
                    corrections.c.event_type,
                    corrections.c.entity_label,
                    corrections.c.corrected_label,
                    corrections.c.detector_source,
                    corrections.c.confidence,
                ).order_by(corrections.c.occurred_at.desc()).limit(20)
            )
            recent = [
                {
                    "occurred_at":      r.occurred_at.isoformat(),
                    "event_type":       r.event_type,
                    "entity_label":     r.entity_label,
                    "corrected_label":  r.corrected_label,
                    "detector_source":  r.detector_source,
                    "confidence":       r.confidence,
                }
                for r in recent_rows.fetchall()
            ]
    except Exception as e:
        logger.error("audit.get_correction_stats failed: %s", e)
        return {
            "total": 0,
            "by_label": {},
            "by_source": {},
            "recent": [],
            "threshold_hints": {},
            "repeated_fp": [],
        }

    # ── Build count aggregates ────────────────────────────────────────────────
    by_label: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    total = 0

    for event_type, label, source, n in records:
        total += n
        by_label.setdefault(label, {"false_positive": 0, "false_negative": 0, "relabel": 0})
        by_label[label][event_type] = by_label[label].get(event_type, 0) + n

        if source:
            by_source.setdefault(source, {"false_positive": 0, "false_negative": 0, "relabel": 0})
            by_source[source][event_type] = by_source[source].get(event_type, 0) + n

    # ── Threshold hints from FP confidence distribution ───────────────────────
    # For each label+source, compute the P50 and P90 FP confidence so the UI
    # can suggest "raising the threshold to X would eliminate Y% of FPs".
    threshold_hints: dict[str, dict] = {}
    for key, confs in fp_confs.items():
        if not confs:
            continue
        confs_sorted = sorted(confs)
        n = len(confs_sorted)
        p50_idx = max(0, int(n * 0.50) - 1)
        p90_idx = max(0, int(n * 0.90) - 1)
        threshold_hints[key] = {
            "count":       n,
            "min":         round(confs_sorted[0], 4),
            "p50":         round(confs_sorted[p50_idx], 4),
            "p90":         round(confs_sorted[p90_idx], 4),
            "max":         round(confs_sorted[-1], 4),
            # Raising the threshold above p90 would eliminate ~90% of these FPs
            "suggested_threshold": round(confs_sorted[p90_idx] + 0.01, 4),
        }

    return {
        "total":            total,
        "by_label":         by_label,
        "by_source":        by_source,
        "recent":           recent,
        "threshold_hints":  threshold_hints,
        "repeated_fp":      repeated_fp,
    }


async def get_training_examples(
    *,
    limit: int = 1000,
    event_type: str | None = None,
    entity_label: str | None = None,
    engine_version: str | None = None,
) -> dict:
    """
    Export logged training examples (raw flagged text) for fine-tuning and
    thesis analysis, newest first, plus a per-version error summary.

    The summary groups example counts by (engine_version, event_type,
    entity_label) — plotting false_positive counts per label across versions
    is the thesis's before/after evidence for each pipeline change.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            query = select(
                training_examples.c.example_id,
                training_examples.c.occurred_at,
                training_examples.c.job_id,
                training_examples.c.event_type,
                training_examples.c.detector_source,
                training_examples.c.entity_label,
                training_examples.c.corrected_label,
                training_examples.c.confidence,
                training_examples.c.entity_text,
                training_examples.c.context,
                training_examples.c.engine_version,
            )
            if event_type:
                query = query.where(training_examples.c.event_type == event_type)
            if entity_label:
                query = query.where(training_examples.c.entity_label == entity_label)
            if engine_version:
                query = query.where(training_examples.c.engine_version == engine_version)
            query = query.order_by(training_examples.c.occurred_at.desc()).limit(limit)

            rows = (await conn.execute(query)).fetchall()

            summary_rows = await conn.execute(
                select(
                    training_examples.c.engine_version,
                    training_examples.c.event_type,
                    training_examples.c.entity_label,
                    func.count().label("n"),
                ).group_by(
                    training_examples.c.engine_version,
                    training_examples.c.event_type,
                    training_examples.c.entity_label,
                ).order_by(training_examples.c.engine_version)
            )
            by_version: dict[str, dict[str, dict[str, int]]] = {}
            for version, ev_type, label, n in summary_rows.fetchall():
                by_version.setdefault(version, {}).setdefault(ev_type, {})[label] = n
    except Exception as e:
        logger.error("audit.get_training_examples failed: %s", e)
        return {"count": 0, "examples": [], "by_version": {},
                "current_engine_version": ENGINE_VERSION}

    examples = [
        {
            "example_id":      str(r.example_id),
            "occurred_at":     r.occurred_at.isoformat(),
            "job_id":          str(r.job_id) if r.job_id else None,
            "event_type":      r.event_type,
            "detector_source": r.detector_source,
            "entity_label":    r.entity_label,
            "corrected_label": r.corrected_label,
            "confidence":      r.confidence,
            "entity_text":     r.entity_text,
            "context":         r.context,
            "engine_version":  r.engine_version,
        }
        for r in rows
    ]
    return {
        "count":                  len(examples),
        "examples":               examples,
        "by_version":             by_version,
        "current_engine_version": ENGINE_VERSION,
    }


async def record_error(
    job_id: str | None,
    error_message: str,
    event_type: str = "error",
    actor: str = "system",
    ip_address: str | None = None,
) -> None:
    """Record a pipeline or extraction error without blocking the caller."""
    audit_row = {
        "event_type": event_type,
        "job_id":     uuid.UUID(job_id) if job_id else None,
        "actor":      actor,
        "ip_address": ip_address,
        "outcome":    "failure",
        "details":    {"error": error_message[:500]},
    }
    await _write_audit(audit_row)


# ---------------------------------------------------------------------------
# Low-level writers — swallow DB errors so they never surface to the user
# ---------------------------------------------------------------------------

async def _write(job_row: dict[str, Any], audit_row: dict[str, Any]) -> None:
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(insert(jobs).values(**job_row))
            await conn.execute(insert(audit_log).values(**audit_row))
    except Exception as e:
        logger.error("audit._write failed: %s", e)


async def _write_audit(audit_row: dict[str, Any]) -> None:
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(insert(audit_log).values(**audit_row))
    except Exception as e:
        logger.error("audit._write_audit failed: %s", e)
