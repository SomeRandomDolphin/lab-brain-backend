"""
app/api/v1/endpoints/lkc.py — LKC graph read endpoints.

GET    /lkc                           — HTML viewer, recent records (admin only — full graph dump)
GET    /lkc/stats                     — graph statistics (any authenticated user — aggregate only)
GET    /lkc/sessions                  — list sessions the caller can access (owner + participant)
GET    /lkc/sessions/{sid}            — session records, filterable (owner or participant)
DELETE /lkc                           — wipe entire graph (admin only)
DELETE /lkc/sessions/{sid}            — wipe one session (owner only)

Confirmed decision: admin-gate the operator-only routes (full dump, full
wipe) rather than leaving them open. See app.api.deps.require_admin.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from app.api.deps import get_current_user, require_admin, require_session_access, require_session_owner
from app.db import lkc_graph, supabase_client

router = APIRouter(prefix="/lkc", tags=["lkc"])


@router.get("", response_class=HTMLResponse)
async def lkc_viewer(_admin: dict = Depends(require_admin)):
    records = await lkc_graph.read_lkc(limit=500)
    return HTMLResponse(
        f"<pre style='font-family:monospace;font-size:13px'>"
        f"{json.dumps(records, indent=2, ensure_ascii=False)}</pre>"
    )


@router.get("/stats")
async def lkc_stats(_current_user: dict = Depends(get_current_user)):
    # Aggregate counts only (no session ids, no content) — safe for any
    # logged-in user, doesn't need admin-gating like the full dump/wipe
    # routes do.
    return await lkc_graph.graph_stats()


@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    accessible_ids = [
        s["session_id"]
        for s in await supabase_client.get_sessions(current_user["id"], limit=10_000)
    ]
    return {"sessions": await lkc_graph.read_sessions(session_ids=accessible_ids)}


@router.get("/sessions/{session_id}")
async def get_session_records(
    session_id:  str = Depends(require_session_access),
    record_type: Optional[str]   = Query(default=None),
    since_unix:  Optional[float] = Query(default=None),
    limit:       int             = Query(default=200, le=2000),
):
    records = await lkc_graph.read_lkc(
        session_id=session_id,
        record_type=record_type,
        since_unix=since_unix,
        limit=limit,
    )
    return {"session_id": session_id, "count": len(records), "records": records}


@router.delete("")
async def clear_lkc(_admin: dict = Depends(require_admin)):
    count = await lkc_graph.clear_all()
    return {"cleared": True, "records_deleted": count}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str = Depends(require_session_owner)):
    count = await lkc_graph.clear_session(session_id)
    return {"session_id": session_id, "records_deleted": count}