"""
alembic/env.py — async Alembic environment for the Supabase Postgres DB.

Connection URL resolution order:
  1. `sqlalchemy.url` already set on the Alembic Config object
     (app/db/migrations.py does this from SUPABASE_DB_URL before
     calling into Alembic — the normal code path).
  2. The SUPABASE_DB_URL env var directly (covers running the
     `alembic` CLI by hand).

When running via the CLI, load_env() is called at module load time to
populate os.environ from the project-root .env file before _resolve_db_url()
runs. Real env vars (Docker / Railway / shell exports) always take precedence.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Ensure the project root is on sys.path so app.core.env is importable
# when `alembic` is invoked from the CLI outside the normal app startup.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before anything reads os.environ (covers CLI usage;
# no-ops in production where real env vars are already set).
from app.core.env import load_env  # noqa: E402
load_env()

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _resolve_db_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    url = os.environ.get("SUPABASE_DB_URL", "")
    if not url:
        raise RuntimeError(
            "No DB URL configured. Set SUPABASE_DB_URL or pass sqlalchemy.url "
            "via the Alembic Config object."
        )
    return url


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (`alembic upgrade head --sql`)."""
    context.configure(
        url=_resolve_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        version_table_schema="public",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations using an async engine (the normal runtime path)."""
    connectable = create_async_engine(
        _resolve_db_url(),
        poolclass=pool.NullPool,
        # NOTE: no server_settings here — that's a Supabase cloud pooler feature
        # that rewrites the username to "postgres.<tenant-id>", which breaks
        # local Docker Supabase where the user is plain "postgres".
        connect_args={
            "command_timeout": 60,
        },
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())