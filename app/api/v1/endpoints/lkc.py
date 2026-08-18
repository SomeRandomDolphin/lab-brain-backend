"""
app/api/v1/endpoints/lkc.py — LKC graph read endpoints.

GET    /lkc                           — HTML viewer, recent records (admin only — full graph dump)
GET    /lkc/stats                     — graph statistics (any authenticated user — aggregate only)
GET    /lkc/sessions                  — list sessions the caller can access (owner + participant)
GET    /lkc/sessions/{sid}            — session records, filterable (owner or participant)
DELETE /lkc                           — wipe entire graph (admin only)
DELETE /lkc/sessions/{sid}            — wipe one session (owner only)
POST   /lkc/kg-query                  — ask the shared kg-agent literature KG directly (any authenticated user)

Confirmed decision: admin-gate the operator-only routes (full dump, full
wipe) rather than leaving them open. See app.api.deps.require_admin.

POST /lkc/kg-query is NOT a query over the session records above — it's a
separate corpus (a fixed 8-paper Neo4j knowledge graph on citi-condor, see
app/services/kg_agent_client.py) exposed here for the frontend to query
directly (e.g. a "search the literature" UI action), independent of the
live-session hybrid QA path in session_pipeline.py. Not session-scoped —
kg-agent's corpus doesn't vary by session — so no session ownership check,
just standard auth.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.api.deps import get_current_user, require_admin, require_session_access, require_session_owner
from app.core.config import cfg
from app.db import lkc_graph, supabase_client
from app.schemas.kg_agent import KgQueryRequest, KgQueryResponse
from app.services import kg_agent_client

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


@router.post("/kg-query", response_model=KgQueryResponse)
async def kg_query(
    body: KgQueryRequest,
    _current_user: dict = Depends(get_current_user),
):
    result = await kg_agent_client.query(body.query)
    if result is None:
        # Covers: kg-agent disabled, circuit-broken from a recent failure,
        # unreachable, Neo4j degraded, or the empty-answer sentinel — all
        # server-side conditions per the kg-agent doc, none of them the
        # caller's fault, so 503 rather than 4xx.
        raise HTTPException(
            status_code=503,
            detail="kg-agent is unavailable right now — check citi-condor/Tailscale, or try again shortly.",
        )
    grounded = result.in_corpus and result.faithfulness >= cfg.kg_agent.faithfulness_threshold
    return KgQueryResponse(
        answer=result.answer,
        grounded=grounded,
        faithfulness=result.faithfulness,
        overall_confidence=result.overall_confidence,
        temporal_validity_status=result.temporal_validity_status,
        documents_used=result.documents_used,
        disclaimer=result.disclaimer or None,
        strategy=result.strategy or None,
    )


@router.delete("")
async def clear_lkc(_admin: dict = Depends(require_admin)):
    count = await lkc_graph.clear_all()
    return {"cleared": True, "records_deleted": count}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str = Depends(require_session_owner)):
    count = await lkc_graph.clear_session(session_id)
    return {"session_id": session_id, "records_deleted": count}