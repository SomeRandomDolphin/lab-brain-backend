"""
app/db/migrations.py — Alembic-based migration runner for the Supabase Postgres DB.

Replaces the old custom `exec_sql` RPC runner that used to live in
app/db/supabase_client.py (it read *.sql files from migrations/versions/
and executed them one by one via a Supabase Postgres function).

Migrations now live as proper Alembic revision scripts in alembic/versions/,
tracked in Postgres' own `alembic_version` table.

Usage
-----
    from app.db.migrations import run_migrations, get_migration_status

    run_migrations()                 # sync — runs `alembic upgrade head`
    status = await get_migration_status()   # async — read-only status check

Configuration
-------------
Reads the DB connection string from the SUPABASE_DB_URL env var, e.g.:

    SUPABASE_DB_URL=postgresql+asyncpg://postgres:<password>@db.<ref>.supabase.co:5432/postgres

This is the same Postgres instance backing your Supabase project — find it
under Project Settings → Database → Connection string in the Supabase
dashboard (use the "URI" value and swap the `postgresql://` prefix for
`postgresql+asyncpg://`).
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

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
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
    No-ops (raises MigrationsNotConfigured, caught by the caller) if
    SUPABASE_DB_URL isn't set, mirroring the old runner's graceful
    degradation when Supabase wasn't configured.

    IMPORTANT — this is a *blocking* call that internally runs
    `asyncio.run(...)` (see alembic/env.py's async template). Calling it
    directly from inside code that's already running on an event loop
    (e.g. FastAPI's `async def startup()`, or any `async def` request
    handler) raises "asyncio.run() cannot be called from a running event
    loop" — silently, because that error happens inside the underlying
    coroutine before anything awaits it, which surfaces as a
    `RuntimeWarning: coroutine '...' was never awaited` rather than an
    obvious traceback. From async code, use `run_migrations_async()`
    instead. This sync version is for plain scripts / CLI usage with no
    event loop already running.
    """
    db_url = _require_db_url()
    cfg = _alembic_config(db_url)
    log.info("[migrations] running `alembic upgrade head` ...")
    command.upgrade(cfg, "head")
    log.info("[migrations] database is up to date.")


async def run_migrations_async() -> None:
    """
    Async-safe entry point for use inside an already-running event loop
    (FastAPI startup, request handlers, etc).

    Runs the blocking `run_migrations()` in a worker thread via
    `asyncio.to_thread`. That thread has no event loop of its own, so the
    `asyncio.run(...)` inside alembic/env.py is free to create one there —
    avoiding the "cannot be called from a running event loop" failure you'd
    hit calling `run_migrations()` directly from async code.
    """
    await asyncio.to_thread(run_migrations)


async def get_migration_status() -> dict:
    """
    Report current vs. head revision and any pending migrations,
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
        "head_revision": head_revision,
        "pending": pending,
        "up_to_date": current_revision == head_revision,
    }