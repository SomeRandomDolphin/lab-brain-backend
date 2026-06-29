"""
app/api/v1/endpoints/livekit.py — LiveKit room management endpoints.

POST   /livekit/room          — create room, return session_id + signed token
GET    /livekit/token         — re-issue token for guest join
GET    /livekit/room/{sid}    — room status
DELETE /livekit/room/{sid}    — end session, stop subscriber, push metrics to Supabase
GET    /events/{sid}          — SSE stream (replaces WebSocket endpoints)
"""

import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import cfg
from app.schemas import RoomCreateRequest, RoomCreateResponse
from app.pipeline import livekit_rooms
from app.db import supabase_client
from app.services import vision, eval_metrics
from app.services.capture import clear_summon
from app.pipeline.dialogue_service import clear_dialogue

log = logging.getLogger(__name__)

router = APIRouter(prefix="/livekit", tags=["livekit"])
sse_router = APIRouter(tags=["events"])


# ── Deferred import to avoid circular: pipeline module imports this router ────
def _get_pipeline():
    from app.pipeline.session_pipeline import livekit_pipeline
    return livekit_pipeline


@router.post("/room", response_model=RoomCreateResponse)
async def create_room(req: RoomCreateRequest = None):
    """Create a LiveKit room, start backend subscriber, persist session to Supabase."""
    if req is None:
        req = RoomCreateRequest()

    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "LiveKit SDK not installed. Run: pip install livekit livekit-api"},
        )

    session_id = str(uuid.uuid4())[:8]

    try:
        await livekit_rooms.create_room(session_id)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    token = livekit_rooms.create_token(session_id, identity=req.display_name)
    livekit_rooms.start_subscriber(session_id, _get_pipeline())

    supabase_client.upsert_session(
        session_id=session_id,
        host_identity=req.display_name,
        started_at=time.time(),
    )

    log.info(f"[livekit] room created: {session_id} host={req.display_name}")
    return RoomCreateResponse(session_id=session_id, token=token, lk_url=cfg.livekit.url)


@router.get("/token")
async def get_token(session_id: str, identity: str = "browser-user"):
    """Issue a JWT for a guest joining an existing room."""
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    room_info = await livekit_rooms.get_room(session_id)
    if room_info is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Room '{session_id}' does not exist."},
        )
    try:
        token = livekit_rooms.create_token(session_id, identity=identity)
        return {"session_id": session_id, "token": token, "lk_url": cfg.livekit.url}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/room/{session_id}")
async def room_status(session_id: str):
    """Return participant count and recording status for a room."""
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    info = await livekit_rooms.get_room(session_id)
    if info is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return info


@router.delete("/room/{session_id}")
async def delete_room(session_id: str):
    """End a session: stop subscriber, delete room, snapshot metrics to Supabase."""
    await livekit_rooms.stop_subscriber(session_id)
    deleted = await livekit_rooms.delete_room(session_id)

    ended_ts      = time.time()
    metrics_snap  = eval_metrics.get_metrics(session_id).summary()
    supabase_client.upsert_session(session_id=session_id, ended_at=ended_ts)
    supabase_client.upsert_eval_metrics(session_id, metrics_snap)

    # Clean up session state
    vision.clear_state(session_id)
    clear_dialogue(session_id)
    clear_summon(session_id)

    return {"session_id": session_id, "deleted": deleted}


@sse_router.get("/events/{session_id}")
async def sse_events(session_id: str):
    """
    Server-Sent Events stream. Replaces /ws/asr, /ws/vision, /ws/tts.
    Event types: session, transcript, agent_reply, perception, mode_change, speak, listening, error.
    """
    return StreamingResponse(
        livekit_rooms.sse_stream(session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
