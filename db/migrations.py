"""
app/db/migrations.py — Alembic migration runner for the Supabase Postgres DB.

Runs `alembic upgrade head` against the same Postgres instance used by the
SQLAlchemy async engine in supabase_client.py.

Migrations live as Alembic revision scripts in alembic/versions/ and are
tracked in Postgres' `alembic_version` table.

Usage
-----
    from db.migrations import run_migrations, run_migrations_async, get_migration_status

    # From a plain script or CLI (no event loop):
    run_migrations()

    # From FastAPI startup or any async context:
    await run_migrations_async()

    # Read-only status check (async):
    status = await get_migration_status()

Configuration
-------------
Reads the DB connection string from SUPABASE_DB_URL, e.g.:

    SUPABASE_DB_URL=postgresql+asyncpg://postgres:<password>@db.<ref>.supabase.co:5432/postgres

This is the same env var used by supabase_client.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import create_async_engine

from core.logging import setup_logging

log = logging.getLogger(__name__)

PROJECT_ROOT           = Path(__file__).resolve().parent.parent.parent
ALEMBIC_INI            = PROJECT_ROOT / "alembic.ini"
ALEMBIC_SCRIPT_LOCATION = PROJECT_ROOT / "alembic"

DB_URL_ENV_VAR = "SUPABASE_DB_URL"


class MigrationsNotConfigured(RuntimeError):
    """Raised when SUPABASE_DB_URL is not set."""


def _require_db_url() -> str:
    db_url = os.environ.get(DB_URL_ENV_VAR, "")
    if not db_url:
        raise MigrationsNotConfigured(
            f"{DB_URL_ENV_VAR} is not set — cannot run or inspect migrations. "
            "Set it to your Supabase Postgres connection string "
            "(postgresql+asyncpg://...)."
        )
    return db_url


def _alembic_config(db_url: str | None = None) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_SCRIPT_LOCATION))
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def run_migrations() -> None:
    """
    Apply all pending Alembic revisions (`alembic upgrade head`).

    Synchronous and idempotent — safe to call on every app startup.
    Raises MigrationsNotConfigured (caller should catch it) if SUPABASE_DB_URL
    is not set.

    IMPORTANT — this is a blocking call. Do not call it from inside an already-
    running event loop (FastAPI startup, async request handlers, etc.).
    Use run_migrations_async() from async contexts instead.
    """
    db_url = _require_db_url()
    cfg = _alembic_config(db_url)
    log.info("[migrations] running `alembic upgrade head` ...")
    command.upgrade(cfg, "head")
    setup_logging()  # ← alembic's env.py just clobbered root's handlers via fileConfig(); reclaim it
    log.info("[migrations] database is up to date.")


async def run_migrations_async(timeout_seconds: float = 30.0) -> None:
    """
    Async-safe wrapper around run_migrations().

    Runs the blocking Alembic call in a worker thread via asyncio.to_thread,
    so the running event loop is not blocked and Alembic's own asyncio.run()
    call inside alembic/env.py has a fresh loop to work with.

    Wrapped in asyncio.wait_for() because asyncpg has no default connection
    timeout: a silently-dropped connection (port conflict, firewall, stale
    Postgres lock on alembic_version, etc.) will otherwise hang this call —
    and therefore the whole FastAPI startup — forever with no error or log
    output. If this fires, the underlying worker thread is NOT killed (Python
    threads can't be force-cancelled); it will keep running in the background
    until the OS-level connection attempt itself times out, but the app can
    at least continue starting up and surface a clear error now.
    """
    try:
        await asyncio.wait_for(asyncio.to_thread(run_migrations), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"[migrations] 'alembic upgrade head' did not complete within "
            f"{timeout_seconds}s — the DB connection is likely hanging "
            f"(check for a port conflict on the Postgres port, or a stale "
            f"lock on the alembic_version table from a previous interrupted run)."
        ) from exc


async def get_migration_status() -> dict:
    """
    Return the current vs. head revision and a list of pending migrations,
    without applying anything (used by GET /supabase/migrations/status).

    Returns
    -------
    {
        "current_revision": str | None,
        "head_revision":    str | None,
        "pending":          list[str],   # "<rev_id> <doc summary>"
        "up_to_date":       bool,
    }
    """
    db_url = _require_db_url()
    cfg = _alembic_config(db_url)
    script = ScriptDirectory.from_config(cfg)
    head_revision = script.get_current_head()

    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            current_revision = await conn.run_sync(
                lambda sync_conn: MigrationContext.configure(sync_conn).get_current_revision()
            )
    finally:
        await engine.dispose()

    pending = []
    if current_revision != head_revision:
        for rev in script.iterate_revisions(head_revision, current_revision):
            doc = (rev.doc or "").strip()
            pending.append(f"{rev.revision} {doc}".strip())
        pending.reverse()  # oldest-pending-first, matches application order

    return {
        "current_revision": current_revision,
        "head_revision":    head_revision,
        "pending":          pending,
        "up_to_date":       current_revision == head_revision,
    }