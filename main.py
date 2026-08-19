"""
main.py — Lab Brain FastAPI app + entry point.

Creates and configures the FastAPI app:
  - CORS middleware
  - Static files
  - All API routers
  - Startup: warm models, run Alembic migrations
  - Root HTML redirect

Also runnable directly (`python main.py`) to start uvicorn locally, using
SERVER_HOST / SERVER_PORT from the environment. In Docker, uvicorn is
invoked directly against this file's `app` object (`uvicorn main:app`) —
the `if __name__ == "__main__"` block below never runs in that path; see
the Dockerfile CMD.

Auth is fully delegated to Supabase Auth (GoTrue).
Schema migrations run via Alembic (app/db/migrations.py).
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env into os.environ first — before anything else that reads
# os.environ at import time gets imported below (pipeline/, services/,
# api/, db/ all read env vars directly at module import time now that
# core/config.py is gone). Real environment variables (shell, Docker, your
# deploy platform) always take precedence; the .env file only fills in
# what isn't already set. See core/env.py.
from core.env import load_env
load_env()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from core.logging import setup_logging

# setup_logging() must run BEFORE importing the api.* router modules below,
# not after. That import chain pulls in pipeline (asr.py's WhisperX/VAD
# loading, dialogue_service's diarization pipeline loading, etc.), which
# does real work and logs INFO-level progress messages ("Loading
# WhisperX...", "WhisperX ready.", ...) at MODULE IMPORT TIME — i.e. the
# moment those import statements run, not later. If that import happens
# before setup_logging() has attached a handler to the root logger, those
# INFO calls aren't delayed or buffered — the root logger has no handler
# yet and defaults to WARNING, so they're silently dropped and gone for
# good. This was invisible for a long time because asr.py's import used to
# fail (for unrelated reasons, since fixed) before ever reaching its
# "WhisperX ready." log line; now that it succeeds, the missing output
# became a real gap. Moving setup_logging() up here, before the api
# router imports, closes it.
setup_logging()
log = logging.getLogger(__name__)

# ── API routers ──────────────────────────────────────────────────────────────
# Flattened from api/v1/endpoints/ into api/ directly (api/v1/router.py and
# api/deps.py are gone — each endpoint module now owns its own auth
# dependencies; see the module docstrings in api/auth.py etc. for why).
# capture.py exports 4 routers, livekit.py exports 2 — same shape as the
# old api_router aggregator, just imported straight from their flattened
# module paths and registered on `app` directly below instead of going
# through an intermediate api_router object.
from api.auth import router as auth_router
from api.livekit import router as livekit_router, sse_router
from api.privacy import router as privacy_router
from api.lkc import router as lkc_router
from api.capture import (
    capture_router,
    agent_router,
    ner_router,
    retrieval_router,
)
from api.sessions import router as sessions_router
from api.supabase import router as supabase_router
from api.websockets import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db import lkc_graph  # noqa: F401

    # Force services.capture to import now, not on first pipeline run.
    # capture.py loads spaCy's NER model at module level (mirroring how
    # asr.py loads WhisperX), but unlike asr.py it isn't on the router's
    # import chain — nothing touches it until the first meeting's first
    # segment reaches process_segment(). That deferred import was causing
    # a ~60s synchronous spaCy load mid-meeting, during which incoming
    # audio piled up in the queue and got dropped. Importing it here runs
    # that load during startup instead, before any room connects.
    import services.capture as _capture  # noqa: F401
    log.info("[startup] services.capture imported (spaCy NER warm)")

    # Same trick for pipeline.asr: WhisperX (and its alignment model, when
    # WHISPER_LANGUAGE is pinned) loads at MODULE IMPORT TIME inside
    # asr.py's top-level try/except — see that file. But asr.py isn't on
    # the router import chain the way services.capture now is, so nothing
    # actually imports it until the first LiveKit room connects and
    # session_pipeline.py pulls it in. That's the ~5min gap in the docker
    # logs between "Application startup complete" (09:46:08) and "Loading
    # WhisperX..." (09:51:04) — the first real meeting's connect pays the
    # full WhisperX + wav2vec2 alignment-model cold-load cost synchronously,
    # the same class of problem the dialogue/vision Ollama warmups below
    # exist to avoid. Force the import here so it happens once, at boot.
    import pipeline.asr as _asr  # noqa: F401
    log.info("[startup] pipeline.asr imported (WhisperX warm)")

    # Surface Supabase misconfiguration at boot time rather than on the
    # first request. Replaces the old _get_admin()/_get_anon() warm-up
    # (removed — supabase_auth.py no longer exposes those; auth is now
    # fully delegated to Supabase Auth/GoTrue and this module only
    # covers Postgres via SQLAlchemy + Storage via supabase-py).
    from db.supabase_client import connectivity_status
    try:
        status = await connectivity_status()
        log.info(f"[startup] Supabase connectivity: {status}")
    except Exception as exc:
        log.error(f"[startup] Supabase connectivity check failed: {exc}")

    # Warm the LKC retrieval embedding model now so the first `mode=qa`
    # segment of a meeting doesn't pay a cold-start on the embedding
    # call. This used to be a synchronous ~420MB sentence-transformers
    # download+load, wrapped in run_in_executor to keep it off the
    # event loop (that's what fired mid-summon in a previous log —
    # modules.json, model.safetensors, etc. downloading while a live
    # segment was in flight).
    #
    # Since the qwen3-embedding switch, warmup() is a small async HTTP
    # call to Ollama's /api/embed — not a blocking local load — so it's
    # awaited directly here instead. Wrapping an async function in
    # run_in_executor(None, _warmup_retrieval) would schedule the
    # coroutine on a thread without ever awaiting it: it returns
    # immediately with an unawaited coroutine object, "[startup] LKC
    # retrieval embedding model warmed" logs regardless of whether the
    # embed call actually happened, and Ollama never actually receives
    # the warmup request — so it's important this isn't run_in_executor
    # any more, not just unnecessary.
    import asyncio
    from services.lkc_retrieval import warmup as _warmup_retrieval
    try:
        await _warmup_retrieval()
        log.info("[startup] LKC retrieval embedding model warmed")
    except Exception as exc:
        log.error(f"[startup] LKC retrieval warmup failed: {exc}")

    # Check the kg-agent literature service the same way — this is just
    # a GET /health (cheap, ~5s client timeout), not a model load, so it
    # doesn't need run_in_executor either. Unlike the warmups below,
    # failure here is expected/non-fatal in normal operation (e.g. off
    # the Tailscale network, citi-condor down) — the hybrid QA path in
    # session_pipeline.py already falls back to transcript-only when
    # kg-agent is unavailable, so this just gets that fact into the
    # startup log instead of surfacing silently on the first live query.
    from services.kg_agent_client import warmup as _warmup_kg_agent
    try:
        await _warmup_kg_agent()
    except Exception as exc:
        log.error(f"[startup] kg-agent warmup failed: {exc}")

    # Warm the local dialogue LLM (Ollama) the same way — otherwise the
    # first chat.completions.create() call of the process's life is
    # whichever user's first real summon happens to be, and that call
    # pays the full cost of Ollama loading the model off disk before it
    # can generate anything. Measured at 259s cold vs. ~2s warm in one
    # session's logs. Same run_in_executor reasoning as above: this is a
    # blocking network call and must not run directly on the event loop.
    from pipeline import dialogue_service
    try:
        await asyncio.get_event_loop().run_in_executor(None, dialogue_service.warmup)
    except Exception as exc:
        log.error(f"[startup] Dialogue LLM warmup failed: {exc}")

    # Warm the vision model (Ollama) too — separately from the dialogue
    # warmup above, since Ollama loads/evicts each model independently.
    # Without this, the first frame session_pipeline.py's _vision_worker
    # sends to vision.analyse_frame() pays the same cold-load cost the
    # dialogue warmup above exists to avoid — and worse, it can then
    # evict the already-warmed dialogue model to make room, so a QA
    # reply mid-meeting silently pays a second cold-load it should never
    # have had to (see the 147s "warm" QA reply this was chasing down).
    # Its own try/except so a vision warmup failure never blocks startup
    # or masks the dialogue warmup above having already succeeded.
    try:
        await asyncio.get_event_loop().run_in_executor(None, dialogue_service.warmup_vision)
    except Exception as exc:
        log.error(f"[startup] Vision LLM warmup failed: {exc}")

    # Run pending Alembic migrations on every startup (idempotent).
    # Replaces the old custom exec_sql-based SQL-file runner.
    # Uses run_migrations_async() (not run_migrations()) because we're
    # inside an async startup hook — see app/db/migrations.py for why.
    from db.migrations import run_migrations_async, MigrationsNotConfigured
    try:
        await run_migrations_async()
        log.info("[startup] Alembic migrations applied (or already up to date).")
    except MigrationsNotConfigured as exc:
        log.warning(f"[startup] Skipping migrations: {exc}")
    except Exception as exc:
        log.error(f"[startup] Migration run failed: {exc}")

    # Nothing to clean up on shutdown today; the yield is what makes this
    # a lifespan context manager rather than a one-shot startup function.
    yield


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
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # allow_origins=["*"] and allow_credentials=True cannot be used together —
    # browsers reject that combination. Use an explicit allowlist with credentials.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://100.122.56.39:5173",
            "http://100.122.56.39:3000",
            "https://citi-condor.tail360067.ts.net",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Debug: surface real 500 errors ────────────────────────────────────────
    _ALLOWED_ORIGINS = {"http://localhost:5173", "http://localhost:3000", "http://100.122.56.39:5173", "http://100.122.56.39:3000", "https://citi-condor.tail360067.ts.net"}

    @app.exception_handler(Exception)
    async def _debug_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        tb = traceback.format_exc()
        log.error(f"[500] {request.method} {request.url}\n{tb}")
        response = JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc), "traceback": tb},
        )
        # NOTE: responses built inside exception handlers don't reliably pick up
        # CORSMiddleware's headers (a known Starlette/FastAPI gotcha), which makes
        # genuine 500s show up in the browser as misleading "blocked by CORS
        # policy" errors instead of the real error. Attach the headers manually
        # so the frontend can actually see and report the real status/body.
        origin = request.headers.get("origin")
        if origin in _ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    # ── Static files ──────────────────────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── API routers ───────────────────────────────────────────────────────────
    # Auth first so /auth/... routes take priority (same ordering the old
    # api/v1/router.py aggregator used).
    app.include_router(auth_router)
    app.include_router(livekit_router)
    app.include_router(sse_router)
    app.include_router(privacy_router)
    app.include_router(lkc_router)
    app.include_router(capture_router)
    app.include_router(agent_router)
    app.include_router(ner_router)
    app.include_router(retrieval_router)
    app.include_router(sessions_router)
    app.include_router(supabase_router)
    app.include_router(ws_router)

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>Lab Brain API running. See <a href='/docs'>/docs</a>.</h2>")

    return app


app = create_app()


if __name__ == "__main__":
    # Local dev entrypoint (`python main.py`). In Docker, uvicorn is invoked
    # directly against this module (`uvicorn main:app`) — see the Dockerfile
    # CMD — so this block never runs in that path; host/port there come from
    # uvicorn's own --host/--port flags instead of SERVER_HOST/SERVER_PORT.
    import os
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVER_PORT", "8000")),
        reload=False,
        log_config=None,   # don't let uvicorn's dictConfig override setup_logging()
    )