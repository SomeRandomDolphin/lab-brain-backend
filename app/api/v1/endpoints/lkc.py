"""
app/api/v1/endpoints/lkc.py — LKC graph read endpoints.

GET    /lkc                           — HTML viewer (recent records)
GET    /lkc/stats                     — graph statistics
GET    /lkc/sessions                  — list all sessions
GET    /lkc/sessions/{sid}            — session records (filterable)
DELETE /lkc                           — wipe entire graph
DELETE /lkc/sessions/{sid}            — wipe one session
"""

import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.db import lkc_graph

router = APIRouter(prefix="/lkc", tags=["lkc"])


@router.get("", response_class=HTMLResponse)
async def lkc_viewer():
    records = await lkc_graph.read_lkc(limit=500)
    return HTMLResponse(
        f"<pre style='font-family:monospace;font-size:13px'>"
        f"{json.dumps(records, indent=2, ensure_ascii=False)}</pre>"
    )


@router.get("/stats")
async def lkc_stats():
    return await lkc_graph.graph_stats()


@router.get("/sessions")
async def list_sessions():
    return {"sessions": await lkc_graph.read_sessions()}


@router.get("/sessions/{session_id}")
async def get_session_records(
    session_id:  str,
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
async def clear_lkc():
    count = await lkc_graph.clear_all()
    return {"cleared": True, "records_deleted": count}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    count = await lkc_graph.clear_session(session_id)
    return {"session_id": session_id, "records_deleted": count}