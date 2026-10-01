"""
Async SQLAlchemy engine + session factory for PostgreSQL.

Usage (in FastAPI lifespan):
    from app.db.session import init_db, get_engine, close_db

    async with lifespan(app):
        await init_db()   # creates tables if not exists
        yield
        await close_db()

Usage (in route handlers / services):
    async with get_engine().connect() as conn:
        await conn.execute(...)
        await conn.commit()

Design:
  - asyncpg driver — fully async, best PostgreSQL performance in Python.
  - create_all() on startup — idempotent. Safe for PoC; use Alembic for
    production migrations (the schema.py metadata is already Alembic-compatible).
  - Connection pool: min 2, max 10 — appropriate for a single-node thesis demo.
"""

import os
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from app.db.schema import metadata

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None


def _build_url() -> str:
    """
    Build the asyncpg database URL from environment variables.
    Falls back to sensible Docker Compose defaults.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        # Allow plain postgresql:// URLs — replace driver for asyncpg
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    host     = os.getenv("POSTGRES_HOST",     "localhost")
    port     = os.getenv("POSTGRES_PORT",     "5432")
    db       = os.getenv("POSTGRES_DB",       "masklayer")
    user     = os.getenv("POSTGRES_USER",     "masklayer")
    password = os.getenv("POSTGRES_PASSWORD", "masklayer")

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> AsyncEngine:
    """Return the module-level engine. Must call init_db() first."""
    if _engine is None:
        raise RuntimeError("Database engine not initialised. Call init_db() first.")
    return _engine


async def init_db() -> None:
    """
    Create the async engine and ensure all tables exist.
    Safe to call on every startup — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    global _engine

    url = _build_url()
    _engine = create_async_engine(
        url,
        pool_size=2,
        max_overflow=8,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        # Lightweight forward-migration for additive columns (create_all does not
        # ALTER existing tables). Idempotent — safe on every startup.
        await conn.execute(text(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS gazetteer_spans INTEGER"
        ))

    logger.info("PostgreSQL connected and schema verified.")


async def close_db() -> None:
    """Dispose the connection pool on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("PostgreSQL connection pool closed.")
