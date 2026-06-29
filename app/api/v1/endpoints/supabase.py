"""
app/api/v1/endpoints/supabase.py — Supabase read endpoints + Alembic migration control.

GET    /supabase/status                               — connectivity check
GET    /supabase/sessions                             — list sessions from Supabase
GET    /supabase/sessions/{sid}/transcripts           — transcript rows
GET    /supabase/sessions/{sid}/summary               — persisted summary
GET    /supabase/sessions/{sid}/report                — report Storage URL
GET    /supabase/sessions/{sid}/audio/{seg_idx}       — audio segment URL

POST   /supabase/migrations/run                       — `alembic upgrade head`
GET    /supabase/migrations/status                    — current vs. head revision, pending list

Migrations are now powered by Alembic (see app/db/migrations.py) instead of
the old custom exec_sql SQL-file runner.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.config import cfg
from app.db import supabase_client
from app.db.migrations import get_migration_status, run_migrations_async

router = APIRouter(prefix="/supabase", tags=["supabase"])


# ── Read endpoints ────────────────────────────────────────────────────────────

@router.get("/status")
async def supabase_status():
    return supabase_client.connectivity_status()


@router.get("/sessions")
async def list_sessions(limit: int = Query(default=50, le=200)):
    return {"sessions": supabase_client.get_sessions(limit=limit)}


@router.get("/sessions/{session_id}/transcripts")
async def get_transcripts(
    session_id: str,
    limit: int = Query(default=500, le=2000),
):
    rows = supabase_client.get_transcripts(session_id, limit=limit)
    return {"session_id": session_id, "count": len(rows), "transcripts": rows}


@router.get("/sessions/{session_id}/summary")
async def get_summary(session_id: str):
    row = supabase_client.get_session_summary(session_id)
    if row is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No summary found. Run POST /summary/{session_id} first."},
        )
    return row


@router.get("/sessions/{session_id}/report")
async def get_report_url(session_id: str):
    url = supabase_client.get_report_url(session_id)
    if url is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No report found. Run POST /summary/{session_id} first."},
        )
    return {"session_id": session_id, "report_url": url}


@router.get("/sessions/{session_id}/audio/{segment_index}")
async def get_audio_url(session_id: str, segment_index: int):
    if not cfg.supabase.store_audio:
        return JSONResponse(
            status_code=404,
            content={"error": "Audio storage disabled. Set store_audio=true in config.json."},
        )
    url = supabase_client.get_audio_segment_url(session_id, segment_index)
    if url is None:
        return JSONResponse(status_code=404, content={"error": "Segment not found."})
    return {"session_id": session_id, "segment_index": segment_index, "url": url}


# ── Migration endpoints (Alembic) ─────────────────────────────────────────────

@router.post("/migrations/run", summary="Apply all pending Alembic migrations")
async def apply_migrations() -> dict:
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


@router.get("/migrations/status", summary="Check pending Alembic migrations (dry-run)")
async def migration_status() -> dict:
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