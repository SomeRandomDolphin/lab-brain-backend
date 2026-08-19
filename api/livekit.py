"""
app/api/livekit.py — LiveKit room management endpoints.

POST   /livekit/room          — create room, return session_id + signed token (login required)
GET    /livekit/token         — join an existing room (login required — no more anonymous guests)
GET    /livekit/room/{sid}    — room status (owner or participant)
DELETE /livekit/room/{sid}    — end session, stop subscriber, push metrics to Supabase (owner only)
GET    /events/{sid}          — SSE stream (replaces WebSocket endpoints)

Confirmed decisions this file implements:
  - Creating a room requires login; the session is owned by that account.
  - Joining a room requires login too — no anonymous guests.

get_current_user() / require_session_access() / require_session_owner()
    Formerly shared from api/deps.py (now removed). Duplicated here rather
    than imported, since each endpoint file now owns its own copy of the
    auth dependencies it needs — see api/auth.py for the canonical version
    of get_current_user's docstring/behaviour notes (identical here).
"""

import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from schemas.livekit import RoomCreateRequest, RoomCreateResponse
from pipeline import livekit_rooms
from db import supabase_auth, supabase_client
from services import vision, eval_metrics
from services.capture import clear_summon
from pipeline.dialogue_service import clear_dialogue

log = logging.getLogger(__name__)

# The browser-facing LiveKit URL. Falls back to LIVEKIT_URL if
# LIVEKIT_PUBLIC_URL isn't set (mirrors the old core.config.py behaviour) —
# fine for same-host local dev, but a real deployment where the browser
# reaches this server over a LAN/Tailscale/public address needs
# LIVEKIT_PUBLIC_URL set explicitly.
import os
LIVEKIT_PUBLIC_URL = os.environ.get("LIVEKIT_PUBLIC_URL") or os.environ.get(
    "LIVEKIT_URL", "ws://host.docker.internal:7880"
)

router = APIRouter(prefix="/livekit", tags=["livekit"])
sse_router = APIRouter(tags=["events"])

_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
)

_404_SESSION = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Session not found.",
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(
        default=None,
        description="Access token fallback for clients that can't set headers (SSE/EventSource only).",
    ),
) -> dict:
    raw = credentials.credentials if credentials is not None else token
    if raw is None:
        raise _401
    user = supabase_auth.verify_session_token(raw)
    if user is None:
        raise _401
    return user


async def require_session_access(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> str:
    """Owner OR participant. Returns 404 (not 403) on mismatch."""
    owner, participants = await supabase_client.get_session_access(session_id)
    if owner is None:
        raise _404_SESSION
    if current_user["id"] != owner and current_user["id"] not in participants:
        raise _404_SESSION
    return session_id


async def require_session_owner(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> str:
    """Owner only — for destructive/management actions."""
    owner, _participants = await supabase_client.get_session_access(session_id)
    if owner is None or current_user["id"] != owner:
        raise _404_SESSION
    return session_id


# ── Deferred import to avoid circular: pipeline module imports this router ────
def _get_pipeline():
    from pipeline.session_pipeline import livekit_pipeline
    return livekit_pipeline


@router.post("/room", response_model=RoomCreateResponse)
async def create_room(
    req: RoomCreateRequest = None,
    current_user: dict = Depends(get_current_user),
):
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
        log.error(f"[livekit] create_room failed: {exc}", exc_info=True)
        return JSONResponse(status_code=503, content={"error": f"LiveKit error: {exc}"})

    try:
        # identity is the account's stable id — this is what participant.identity
        # will carry through to session_pipeline.py's audio/video drain, and
        # what app/services/privacy.py's consent registry is keyed on (see
        # POST /privacy/tos-consent). display_name is purely cosmetic and can
        # safely be whatever the client asked for; it must never be passed as
        # `identity` or two people could collide on the same LiveKit identity
        # (or on the same consent record) whenever they share a display name.
        token = livekit_rooms.create_token(
            session_id,
            identity=current_user["id"],
            display_name=req.display_name,
        )
    except Exception as exc:
        log.error(f"[livekit] create_token failed: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Token error: {exc}"})

    livekit_rooms.start_subscriber(session_id, _get_pipeline())

    try:
        # user_id is set once here, at creation, from the authenticated
        # requester — never trusted from the request body. host_identity
        # stays the cosmetic display name; it is no longer the access
        # boundary (that's sessions.user_id + session_participants now).
        await supabase_client.upsert_session(
            session_id=session_id,
            user_id=current_user["id"],
            host_identity=req.display_name,
            started_at=time.time(),
        )
    except Exception as exc:
        # Non-fatal: room is live, don't kill the session over a DB write
        log.error(f"[livekit] upsert_session failed (non-fatal): {exc}", exc_info=True)

    log.info(f"[livekit] room created: {session_id} host={req.display_name} owner={current_user['id']}")
    return RoomCreateResponse(session_id=session_id, token=token, lk_url=LIVEKIT_PUBLIC_URL)


@router.get("/token")
async def get_token(
    session_id: str = Query(...),
    identity: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    """
    Issue a JWT for an authenticated user joining an existing room.

    Anonymous guest join is gone (per confirmed decision) — that's enforced
    by get_current_user above. Access itself is NOT gated behind
    require_session_access here: that dependency only admits the owner or
    an existing session_participants row, but the only way to become a
    participant is to successfully call this endpoint — a first-time
    joiner could never pass it (the owner alone has an unconditional path
    via sessions.user_id, which is why this bug only ever showed up for
    non-owners). Any logged-in user who has the session_id (effectively a
    room code/invite link, matching this file's actual join model) is
    allowed to attempt to join; the room-existence check below is the real
    gate, exactly like every other join-by-code flow.

    `identity` (the query param) stays as an optional cosmetic display
    label for the LiveKit UI — it defaults to the current user's name
    rather than the old shared "browser-user" literal, since the
    participant is now a real authenticated account. It is NOT passed as
    the actual LiveKit identity below (that was the bug: a client could
    previously set ?identity=<anything> and that became their real LiveKit
    participant.identity, which is also what app/services/privacy.py's
    consent registry keys on — letting a client collide with, or even
    impersonate, another account's identity). The real identity is always
    server-derived from current_user["id"], never client-supplied.
    """
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    room_info = await livekit_rooms.get_room(session_id)
    if room_info is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Room '{session_id}' does not exist."},
        )

    display_identity = identity or current_user.get("name") or current_user["email"]

    try:
        token = livekit_rooms.create_token(
            session_id,
            identity=current_user["id"],
            display_name=display_identity,
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # Record participation (no-op if already recorded, or if this user is
    # the owner — an owner is already accessible via sessions.user_id).
    try:
        await supabase_client.add_session_participant(session_id, current_user["id"])
    except Exception as exc:
        log.error(f"[livekit] add_session_participant failed (non-fatal): {exc}", exc_info=True)

    return {"session_id": session_id, "token": token, "lk_url": LIVEKIT_PUBLIC_URL}


@router.get("/room/{session_id}")
async def room_status(session_id: str = Depends(require_session_access)):
    """Return participant count and recording status for a room. Owner or participant."""
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    info = await livekit_rooms.get_room(session_id)
    if info is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return info


@router.delete("/room/{session_id}")
async def delete_room(session_id: str = Depends(require_session_owner)):
    """End a session: stop subscriber, delete room, snapshot metrics to Supabase. Owner only."""
    # Stop subscriber and delete LiveKit room — always run these first.
    await livekit_rooms.stop_subscriber(session_id)
    deleted = await livekit_rooms.delete_room(session_id)

    ended_ts     = time.time()
    metrics_snap = eval_metrics.get_metrics(session_id).summary()
    db_errors    = []

    # DB writes are best-effort — a ProgrammingError or schema mismatch
    # must never cause a 500 that blocks the frontend from completing teardown.
    # No user_id passed here — this is a partial-update (ended_at only) and
    # must NOT touch the existing owner. See supabase_client.upsert_session.
    try:
        await supabase_client.upsert_session(session_id=session_id, ended_at=ended_ts)
    except Exception as exc:
        db_errors.append(f"upsert_session: {exc}")
        log.error(f"[livekit] delete_room upsert_session failed: {exc}", exc_info=True)

    try:
        await supabase_client.upsert_eval_metrics(session_id, metrics_snap)
    except Exception as exc:
        db_errors.append(f"upsert_eval_metrics: {exc}")
        log.error(f"[livekit] delete_room upsert_eval_metrics failed: {exc}", exc_info=True)

    # Clean up in-memory session state regardless of DB outcome.
    vision.clear_state(session_id)
    clear_dialogue(session_id)
    clear_summon(session_id)

    return {
        "session_id": session_id,
        "deleted":    deleted,
        **({"db_warnings": db_errors} if db_errors else {}),
    }


@sse_router.get("/events/{session_id}")
async def sse_events(session_id: str = Depends(require_session_access)):
    """
    Server-Sent Events stream. Replaces /ws/asr, /ws/vision, /ws/tts.
    Event types: session, transcript, agent_reply, perception, mode_change, speak, listening, error.
    """
    return StreamingResponse(
        livekit_rooms.sse_stream(session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )