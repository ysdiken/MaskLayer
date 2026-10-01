-- MaskLayer — PostgreSQL bootstrap
-- Runs once automatically on first container start via docker-entrypoint-initdb.d.
--
-- The full schema (jobs, audit_log, spans, entity_stats) is created by
-- SQLAlchemy create_all() on backend startup — idempotent, always in sync
-- with app/db/schema.py which is the single source of truth.
--
-- This file only does things that must happen before the app connects:
--   1. Enable the pgcrypto extension (gen_random_uuid used in some queries).
--   2. Enable pg_stat_statements for query performance monitoring.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- KVKK note: no PII is ever stored in PostgreSQL.
-- Original values live in Redis with a TTL; only placeholders and metadata here.
