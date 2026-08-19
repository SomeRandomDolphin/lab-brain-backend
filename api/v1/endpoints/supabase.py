"""
app/api/v1/endpoints/supabase.py — Supabase read endpoints + Alembic migration control.

GET    /supabase/status                               — connectivity check
GET    /supabase/sessions                             — list the caller's own sessions
GET    /supabase/sessions/{sid}/transcripts           — transcript rows (owner or participant)
GET    /supabase/sessions/{sid}/summary               — persisted summary (owner or participant)
GET    /supabase/sessions/{sid}/report                — report Storage URL (owner or participant)
GET    /supabase/sessions/{sid}/audio/{seg_idx}       — audio segment URL (owner or participant)

POST   /supabase/migrations/run                       — `alembic upgrade head` (admin only)
GET    /supabase/migrations/status                    — current vs. head revision, pending list (admin only)

Migrations are now powered by Alembic (see app/db/migrations.py) instead of
the old custom exec_sql SQL-file runner.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.deps import get_current_user, require_admin, require_session_access
from db import supabase_client
from db.migrations import get_migration_status, run_migrations_async

router = APIRouter(prefix="/supabase", tags=["supabase"])

import os
SUPABASE_STORE_AUDIO = os.environ.get("SUPABASE_STORE_AUDIO", "true").strip().lower() in ("1", "true", "yes", "on")


# ── Read endpoints ────────────────────────────────────────────────────────────

@router.get("/status")
async def supabase_status():
    return await supabase_client.connectivity_status()


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=50, le=200),
    current_user: dict = Depends(get_current_user),
):
    # Bug fix: get_sessions is `async def` in supabase_client.py — this was
    # previously called without await, handing FastAPI a coroutine object
    # instead of a result (a serialization error or garbage on every call).
    # Also now scoped to the caller's own sessions, replacing the previous
    # "every session in the database, to anyone who can reach the API"
    # behaviour.
    return {"sessions": await supabase_client.get_sessions(current_user["id"], limit=limit)}


@router.get("/sessions/{session_id}/transcripts")
async def get_transcripts(
    session_id: str = Depends(require_session_access),
    limit: int = Query(default=500, le=2000),
):
    # Bug fix: same missing-await issue as get_sessions above.
    rows = await supabase_client.get_transcripts(session_id, limit=limit)
    return {"session_id": session_id, "count": len(rows), "transcripts": rows}


@router.get("/sessions/{session_id}/summary")
async def get_summary(session_id: str = Depends(require_session_access)):
    # Bug fix: same missing-await issue as get_sessions above.
    row = await supabase_client.get_session_summary(session_id)
    if row is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No summary found. Run POST /summary/{session_id} first."},
        )
    return row


@router.get("/sessions/{session_id}/report")
async def get_report_url(session_id: str = Depends(require_session_access)):
    url = supabase_client.get_report_url(session_id)
    if url is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No report found. Run POST /summary/{session_id} first."},
        )
    return {"session_id": session_id, "report_url": url}


@router.get("/sessions/{session_id}/audio/{segment_index}")
async def get_audio_url(
    segment_index: int,
    session_id: str = Depends(require_session_access),
):
    if not SUPABASE_STORE_AUDIO:
        return JSONResponse(
            status_code=404,
            content={"error": "Audio storage disabled. Set SUPABASE_STORE_AUDIO=true in your .env."},
        )
    url = supabase_client.get_audio_segment_url(session_id, segment_index)
    if url is None:
        return JSONResponse(status_code=404, content={"error": "Segment not found."})
    return {"session_id": session_id, "segment_index": segment_index, "url": url}


# ── Migration endpoints (Alembic) — admin only ────────────────────────────────
# Confirmed decision: add a real admin-role concept rather than leaving these
# open. See api.deps.require_admin / db.supabase_auth for how
# isAdmin is set (Supabase user_metadata.role == "admin").

@router.post("/migrations/run", summary="Apply all pending Alembic migrations (admin only)")
async def apply_migrations(_admin: dict = Depends(require_admin)) -> dict:
    """
    Run `alembic upgrade head` against the Supabase database.
    Already-applied revisions are skipped automatically (idempotent).
    Returns the migration status after the upgrade completes.
    """
    try:
        # Async-safe: this handler is itself async, so the blocking Alembic
        # call must go through run_migrations_async() (runs in a worker
        # thread) rather than run_migrations() directly — see
        # app/db/migrations.py for why.
        await run_migrations_async()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    status = await get_migration_status()
    return {"ok": True, **status}


@router.get("/migrations/status", summary="Check pending Alembic migrations (admin only, dry-run)")
async def migration_status(_admin: dict = Depends(require_admin)) -> dict:
    """
    Return current and head revision without applying anything.

    Example response::

        {
            "current_revision": "0002",
            "head_revision":    "0003",
            "pending":          ["0003 add reporting views"],
            "up_to_date":       false
        }
    """
    try:
        return await get_migration_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc