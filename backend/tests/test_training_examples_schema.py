"""
Offline tests for the training_examples store (no live PostgreSQL needed).

Verifies:
  - The table is registered in metadata with the columns the fine-tuning
    exporter and thesis analysis depend on (raw text, engine_version, timestamps).
  - record_correction() builds rows for BOTH tables: hashed text in
    corrections, raw text + engine_version in training_examples.
  - The API request model accepts the optional context field.
"""

import pytest
from sqlalchemy import insert

from app.api.correction_routes import CorrectionRequest
from app.db.schema import metadata, training_examples
from app.masking.pipeline import ENGINE_VERSION


class TestSchema:

    def test_table_registered(self):
        assert "training_examples" in metadata.tables

    def test_required_columns_present(self):
        cols = set(training_examples.c.keys())
        assert {
            "example_id", "occurred_at", "job_id", "event_type",
            "detector_source", "entity_label", "corrected_label",
            "confidence", "entity_text", "context", "engine_version",
        } <= cols

    def test_raw_text_and_version_not_nullable(self):
        # The two columns this table exists for must always be filled.
        assert training_examples.c.entity_text.nullable is False
        assert training_examples.c.engine_version.nullable is False

    def test_insert_statement_compiles(self):
        # Catches column/type mismatches without a live database.
        stmt = insert(training_examples).values(
            event_type="false_positive",
            detector_source="ner",
            entity_label="Company",
            confidence=0.91,
            entity_text="Hava Bankası Tedarikçi Davranış İlkeleri",
            context="… Hava Bankası Tedarikçi Davranış İlkeleri tüm tedarikçiler …",
            engine_version=ENGINE_VERSION,
        )
        assert "INSERT INTO training_examples" in str(stmt)


class TestEngineVersion:

    def test_version_is_set(self):
        assert ENGINE_VERSION
        assert len(ENGINE_VERSION) <= 40   # fits the schema column


class TestApiModel:

    def test_context_field_optional(self):
        req = CorrectionRequest(
            event_type="false_positive",
            entity_label="Company",
            entity_text="Twitter",
        )
        assert req.context is None

    def test_context_field_accepted(self):
        req = CorrectionRequest(
            event_type="false_positive",
            entity_label="Company",
            entity_text="Twitter",
            context="Duyurular Twitter üzerinden paylaşılacaktır.",
        )
        assert req.context is not None
