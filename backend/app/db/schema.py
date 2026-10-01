"""
PostgreSQL schema for MaskLayer — KVKK/BDDK compliant audit trail.

Two tables:
  jobs       — one row per masking request. Tracks document lifecycle,
               extraction method, span counts, and processing time.
  audit_log  — append-only event log. Every state transition and access
               to PII (mask, unmask, manual correction, expiry) is recorded.
               This is your KVKK Article 12 evidence of "technical measures".

Design choices:
  - SQLAlchemy Core (no ORM) — explicit SQL, easier to inspect, no magic.
  - asyncpg driver (async) — consistent with the rest of the stack.
  - UUID primary keys — no sequential ID leakage across tenants.
  - Immutable audit_log — no UPDATE/DELETE. Rows are evidence.
  - JSONB columns for spans/mapping_summary — flexible for thesis iterations
    without schema migrations every time you add a span field.
  - Indexes on (created_at, job_id) — covers the analyst review query
    ("show me all jobs from the last 24 h") and the per-job audit lookup.

KVKK mapping:
  Article 12 — "appropriate technical measures" →
    audit_log.event_type IN ('mask','unmask','manual_correction','access','expiry')
  Article 7  — "storage limitation" →
    jobs.mapping_expires_at tracks Redis TTL; a scheduled task sets
    jobs.status = 'expired' and logs event_type = 'expiry' when TTL elapses.
  Article 11 — "right of access" →
    audit_log lets you reconstruct exactly who accessed which document when.

Thesis note:
  Show your committee the audit_log table in psql during the demo.
  Run:  SELECT event_type, COUNT(*) FROM audit_log GROUP BY event_type;
  This proves the system records every PII access event as required by KVKK.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

# ---------------------------------------------------------------------------
# Metadata registry — all tables registered here
# ---------------------------------------------------------------------------

metadata = MetaData()


# ---------------------------------------------------------------------------
# jobs table
# ---------------------------------------------------------------------------
#
# One row per POST /api/v1/mask or /api/v1/upload request.
#
# status lifecycle:
#   pending → processing → complete
#                        → failed
#   complete → expired   (after Redis TTL elapses)

jobs = Table(
    "jobs",
    metadata,

    # ── Identity ──────────────────────────────────────────────────────────────
    Column("job_id",         UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("created_at",     DateTime(timezone=True), nullable=False,
           default=lambda: datetime.now(timezone.utc)),
    Column("updated_at",     DateTime(timezone=True), nullable=False,
           default=lambda: datetime.now(timezone.utc),
           onupdate=lambda: datetime.now(timezone.utc)),

    # ── Status ────────────────────────────────────────────────────────────────
    # pending | processing | complete | failed | expired
    Column("status",         String(20), nullable=False, default="complete"),

    # ── Document metadata ─────────────────────────────────────────────────────
    Column("filename",       Text, nullable=True),       # original upload filename
    Column("file_type",      String(10), nullable=True), # pdf | docx | png | jpg | text
    Column("page_count",     Integer, nullable=True),

    # ── Extraction ────────────────────────────────────────────────────────────
    # comma-separated set of methods used, e.g. "native,ocr"
    Column("extraction_methods", Text, nullable=True),
    # mean Tesseract confidence across OCR pages (NULL if no OCR pages)
    Column("ocr_mean_confidence", Float, nullable=True),

    # ── Masking stats ─────────────────────────────────────────────────────────
    Column("mode",           String(10), nullable=False, default="full"),  # full|regex|ner
    Column("span_count",     Integer, nullable=True),     # total spans detected
    Column("regex_spans",    Integer, nullable=True),     # spans from regex engine
    Column("ner_spans",      Integer, nullable=True),     # spans from NER engine
    Column("manual_spans",   Integer, nullable=True),     # user-confirmed spans
    Column("gazetteer_spans", Integer, nullable=True),    # spans from gazetteer detector
    Column("char_count",     Integer, nullable=True),     # length of input text

    # ── Timing ────────────────────────────────────────────────────────────────
    Column("processing_time_ms", Float, nullable=True),

    # ── Redis mapping TTL ─────────────────────────────────────────────────────
    # When the Redis key expires — used by the expiry sweeper task
    Column("mapping_expires_at", DateTime(timezone=True), nullable=True),

    # ── Span summary (JSONB — avoids separate spans table for PoC) ────────────
    # List of { label, source, confidence, text } dicts — no original PII stored here.
    # text field is the *placeholder* (e.g. "{TC_No_1}"), not the original value.
    Column("spans_summary",  JSONB, nullable=True),

    # ── Error ─────────────────────────────────────────────────────────────────
    Column("error_message",  Text, nullable=True),

    # ── Indexes ───────────────────────────────────────────────────────────────
    Index("ix_jobs_created_at", "created_at"),
    Index("ix_jobs_status",     "status"),
)


# ---------------------------------------------------------------------------
# audit_log table
# ---------------------------------------------------------------------------
#
# Append-only event log. Every PII access event gets a row.
# Never UPDATE or DELETE rows — they are legal evidence under KVKK Article 12.
#
# event_type values:
#   mask              — document submitted and masked
#   unmask            — LLM response de-masked (PII accessed)
#   manual_correction — analyst added/removed/relabelled a span
#   access            — analyst viewed the original PII values (future: UI)
#   expiry            — Redis mapping TTL elapsed, PII no longer recoverable
#   error             — pipeline error (no PII accessed but worth recording)

audit_log = Table(
    "audit_log",
    metadata,

    # ── Identity ──────────────────────────────────────────────────────────────
    Column("log_id",      UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("occurred_at", DateTime(timezone=True), nullable=False,
           default=lambda: datetime.now(timezone.utc)),

    # ── Event ─────────────────────────────────────────────────────────────────
    Column("event_type",  String(30), nullable=False),  # see values above
    Column("job_id",      UUID(as_uuid=True), nullable=True),  # FK to jobs (not enforced for append speed)

    # ── Actor ─────────────────────────────────────────────────────────────────
    # Who triggered the event. "system" for automated events, user ID otherwise.
    Column("actor",       String(255), nullable=False, default="system"),
    # Client IP — KVKK Article 12 / access log
    Column("ip_address",  String(45), nullable=True),   # max 45 chars covers IPv6

    # ── Outcome ───────────────────────────────────────────────────────────────
    Column("outcome",     String(10), nullable=False, default="success"),  # success | failure

    # ── Details (JSONB) ───────────────────────────────────────────────────────
    # Flexible detail bag. Examples:
    #   mask:   { "span_count": 12, "mode": "full", "filename": "sozlesme.pdf" }
    #   unmask: { "placeholder_count": 5, "unknown_placeholders": 0 }
    #   expiry: { "mapping_expires_at": "2024-06-16T10:30:00Z" }
    # IMPORTANT: Never store original PII values in this column.
    Column("details",     JSONB, nullable=True),

    # ── Indexes ───────────────────────────────────────────────────────────────
    Index("ix_audit_occurred_at", "occurred_at"),
    Index("ix_audit_job_id",      "job_id"),
    Index("ix_audit_event_type",  "event_type"),
)


# ---------------------------------------------------------------------------
# corrections table  (active-learning signal)
# ---------------------------------------------------------------------------
#
# One row per analyst correction. Three event types:
#   false_positive  — analyst removed a span the detector produced
#   false_negative  — analyst added a span the detector missed
#   relabel         — analyst kept the span but changed its label
#
# KVKK note: entity text is never stored here — only the label, source,
# confidence, and a SHA-256 hash of the text (for deduplication analytics).
# The hash cannot be reversed to recover the original PII.

corrections = Table(
    "corrections",
    metadata,

    Column("correction_id",   UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("occurred_at",     DateTime(timezone=True), nullable=False,
           default=lambda: datetime.now(timezone.utc)),

    # ── Linkage ───────────────────────────────────────────────────────────────
    Column("job_id",          UUID(as_uuid=True), nullable=True),

    # ── Event classification ──────────────────────────────────────────────────
    # false_positive | false_negative | relabel
    Column("event_type",      String(20),  nullable=False),

    # Which detector produced the wrong span (null for false_negative — no detector fired)
    Column("detector_source", String(10),  nullable=True),   # regex | ner | null

    # Label the detector assigned (or the label the analyst chose for false_negatives)
    Column("entity_label",    String(50),  nullable=False),

    # For relabel events: what the analyst changed it to
    Column("corrected_label", String(50),  nullable=True),

    # Detector confidence at time of correction (null for false_negatives)
    Column("confidence",      Float,       nullable=True),

    # SHA-256 of the entity text — for deduplication; never reversible to PII
    Column("text_hash",       String(64),  nullable=True),

    Column("ip_address",      String(45),  nullable=True),

    # ── Indexes ───────────────────────────────────────────────────────────────
    Index("ix_corrections_occurred_at",  "occurred_at"),
    Index("ix_corrections_job_id",       "job_id"),
    Index("ix_corrections_label",        "entity_label"),
    Index("ix_corrections_event_type",   "event_type"),
)


# ---------------------------------------------------------------------------
# training_examples table  (fine-tuning + thesis evaluation store)
# ---------------------------------------------------------------------------
#
# Companion to `corrections`: same events, but stores the RAW flagged text.
# The corrections table hashes entity text (KVKK precaution), which makes it
# useless as fine-tuning data — you cannot recover "what the model got wrong"
# from a SHA-256. This table is the deliberate exception:
#
#   - It exists so analyst corrections can be exported as labelled training
#     examples (domain adaptation of the NER model) and so the thesis can
#     measure per-category error rates over time.
#   - engine_version stamps each row with the detection-logic version that
#     produced the error (see ENGINE_VERSION in app/masking/pipeline.py).
#     Comparing false-positive rates per category across versions is the
#     thesis's evidence that each pipeline change had measurable effect.
#   - Append-only, like audit_log: rows are experimental evidence, never
#     UPDATE or DELETE them during the study.
#
# KVKK note: this table holds potentially sensitive text in cleartext and
# must never leave the trust boundary. For production use, define a retention
# policy (purge after the fine-tune snapshot is taken); for the thesis PoC
# with fictional documents this is acceptable as-is.

training_examples = Table(
    "training_examples",
    metadata,

    Column("example_id",      UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("occurred_at",     DateTime(timezone=True), nullable=False,
           default=lambda: datetime.now(timezone.utc)),

    # ── Linkage ───────────────────────────────────────────────────────────────
    Column("job_id",          UUID(as_uuid=True), nullable=True),

    # ── Event classification (mirrors corrections) ────────────────────────────
    # false_positive | false_negative | relabel
    Column("event_type",      String(20),  nullable=False),
    Column("detector_source", String(10),  nullable=True),   # regex | ner | null
    Column("entity_label",    String(50),  nullable=False),
    Column("corrected_label", String(50),  nullable=True),
    Column("confidence",      Float,       nullable=True),

    # ── Training payload ──────────────────────────────────────────────────────
    # The raw flagged text — the whole point of this table.
    Column("entity_text",     Text,        nullable=False),
    # Optional surrounding text (sentence/window) — needed for NER fine-tuning,
    # where the model learns from context, not the entity string alone.
    Column("context",         Text,        nullable=True),

    # ── Versioning ────────────────────────────────────────────────────────────
    # Detection-logic version at the time of the error (ENGINE_VERSION).
    Column("engine_version",  String(40),  nullable=False),

    # ── Indexes ───────────────────────────────────────────────────────────────
    Index("ix_training_occurred_at",    "occurred_at"),
    Index("ix_training_label",          "entity_label"),
    Index("ix_training_event_type",     "event_type"),
    Index("ix_training_engine_version", "engine_version"),
)
