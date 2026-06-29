"""
app/main.py — FastAPI application factory.

Creates and configures the FastAPI app:
  - CORS middleware
  - Static files
  - All API routers (v1)
  - Startup: configure LKC paths, run Alembic migrations
  - Root HTML redirect

Auth is now fully delegated to Supabase Auth (GoTrue).
The old SQLite auth.db initialisation has been removed.
Schema migrations are now run via Alembic (app/db/migrations.py) instead
of the old custom exec_sql-based SQL-file runner.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Load .env into os.environ first — before app.core.config (or anything else
# that reads os.environ at import time) gets imported below. Real environment
# variables (shell, Docker, your deploy platform) always take precedence;
# .env only fills in what isn't already set. See app/core/env.py.
from app.core.env import load_env
load_env()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import cfg
from app.core.logging import setup_logging
from app.api.v1.router import api_router

setup_logging()
log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lab Brain — Module 5",
        description=(
            "Multimodal Conversational Agent. "
            "LiveKit WebRTC media layer + SSE event stream + Supabase persistence."
        ),
        version="7.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # "*" is suitable for local dev. In production, replace with explicit
    # frontend origin(s) and set allow_credentials=False or use an allowlist.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files ──────────────────────────────────────────────────────────
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── API routers ───────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>Lab Brain API running. See <a href='/docs'>/docs</a>.</h2>")

    # ── Startup ───────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup() -> None:
        from pathlib import Path as _Path
        from app.db import lkc_graph

        # Configure LKC knowledge graph paths
        lkc_log = _Path(cfg.lkc.log_file)
        lkc_graph.configure(
            db_path=lkc_log.with_suffix(".db"),
            jsonl_path=lkc_log,
        )
        log.info(f"[startup] LKC graph configured at {cfg.lkc.db_file}")

        # Warm up the Supabase auth client so the first request doesn't pay
        # the cold-start cost and misconfiguration is caught at boot time.
        try:
            from app.db.supabase_auth import _get_admin, _get_anon
            _get_admin()
            _get_anon()
            log.info("[startup] Supabase Auth clients ready")
        except RuntimeError as exc:
            log.warning(f"[startup] Supabase Auth not configured: {exc}")

        # Run pending Alembic migrations on every startup (idempotent).
        # Replaces the old custom exec_sql-based SQL-file runner.
        # Uses run_migrations_async() (not run_migrations()) because we're
        # inside an async startup hook — see app/db/migrations.py for why.
        from app.db.migrations import run_migrations_async, MigrationsNotConfigured
        try:
            await run_migrations_async()
            log.info("[startup] Alembic migrations applied (or already up to date).")
        except MigrationsNotConfigured as exc:
            log.warning(f"[startup] Skipping migrations: {exc}")
        except Exception as exc:
            log.error(f"[startup] Migration run failed: {exc}")

    return app


app = create_app()