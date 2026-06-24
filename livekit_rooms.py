"""
livekit_rooms.py — LiveKit integration layer (Module 5, Month 6)

Responsibilities
----------------
1. Token generation  — sign short-lived JWTs that the frontend uses to join rooms.
2. Room management   — create / inspect / close LiveKit rooms via the server API.
3. Audio subscriber  — subscribe to the browser's audio track, pipe decoded PCM
                       into the existing VadChunker → WhisperX pipeline.
4. Video subscriber  — subscribe to the browser's camera track, pipe JPEG frames
                       into the existing vision.analyse_frame() pipeline.
5. SSE broadcaster   — fan-out server-sent events (transcript, agent_reply,
                       perception, mode_change) to all connected browser clients
                       for a given session, replacing the three WebSocket endpoints.

All existing pipeline internals (VadChunker, WhisperX, capture, dialogue,
lkc_graph, lkc_retrieval, vision) are unchanged — only the ingress/egress
transport layer is replaced.

Dependencies
------------
    pip install livekit livekit-api
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

log = logging.getLogger(__name__)

# ── LiveKit SDK imports (graceful degradation if not installed) ───────────────
LIVEKIT_AVAILABLE = False
try:
    from livekit import rtc as lk_rtc                    # room / track objects
    from livekit.api import AccessToken, VideoGrants      # token signing
    from livekit.api import LiveKitAPI                    # room management REST
    LIVEKIT_AVAILABLE = True
    log.info("[livekit] SDK loaded.")
except ImportError:
    log.warning(
        "[livekit] 'livekit' and/or 'livekit-api' packages not installed. "
        "Run: pip install livekit livekit-api\n"
        "LiveKit endpoints will return 503 until the packages are available."
    )

from config import cfg


# ── SSE event bus ─────────────────────────────────────────────────────────────
# Maps session_id → list of asyncio.Queue instances (one per SSE subscriber).
# When server code calls broadcast(session_id, event_dict), every connected
# browser client receives the event on its SSE stream.

_sse_subscribers: dict[str, list[asyncio.Queue]] = {}


def _get_queues(session_id: str) -> list[asyncio.Queue]:
    return _sse_subscribers.setdefault(session_id, [])


def broadcast(session_id: str, event: dict) -> None:
    """
    Push an event dict to every SSE subscriber for the session.
    Safe to call from any coroutine or thread (uses put_nowait).
    """
    for q in _get_queues(session_id):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # slow client — drop rather than block


async def sse_stream(session_id: str) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted strings.
    Register a new queue, yield events, unregister on disconnect.

    Usage in FastAPI:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(sse_stream(sid), media_type="text/event-stream")
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    queues = _get_queues(session_id)
    queues.append(q)
    try:
        # Send a hello event so the browser knows the SSE connection is live
        yield f"event: connected\ndata: {json.dumps({'session_id': session_id})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=25.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Keep-alive comment so reverse proxies don't close idle connections
                yield ": keepalive\n\n"
    finally:
        try:
            queues.remove(q)
        except ValueError:
            pass


# ── Token generation ──────────────────────────────────────────────────────────

def create_token(session_id: str, identity: str, ttl_seconds: int = 3600) -> str:
    """
    Return a signed LiveKit JWT for the given participant identity and room.

    The token grants:
      - canPublish    (browser publishes mic + camera)
      - canSubscribe  (backend subscribes to the same tracks)
      - roomJoin      for the room named after session_id
    """
    if not LIVEKIT_AVAILABLE:
        raise RuntimeError("livekit-api package not installed")

    grants = VideoGrants(
        room_join=True,
        room=session_id,
        can_publish=True,
        can_subscribe=True,
    )
    token = (
        AccessToken(cfg.livekit.api_key, cfg.livekit.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=ttl_seconds))
        .to_jwt()
    )
    return token


# ── Room management ───────────────────────────────────────────────────────────

async def create_room(session_id: str) -> dict:
    """
    Create a LiveKit room with name == session_id.
    Returns room metadata dict.
    """
    if not LIVEKIT_AVAILABLE:
        raise RuntimeError("livekit-api package not installed")

    async with LiveKitAPI(
        url=cfg.livekit.url,
        api_key=cfg.livekit.api_key,
        api_secret=cfg.livekit.api_secret,
    ) as api:
        room = await api.room.create_room(
            lk_rtc.CreateRoomRequest(name=session_id, empty_timeout=300)
        )
    return {"name": room.name, "sid": room.sid}


async def get_room(session_id: str) -> Optional[dict]:
    """
    Return room metadata or None if the room doesn't exist.
    participant_count reflects only human participants (excludes lab-brain-server).
    """
    if not LIVEKIT_AVAILABLE:
        return None
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            rooms = await api.room.list_rooms(lk_rtc.ListRoomsRequest(names=[session_id]))
            if not rooms.rooms:
                return None
            r = rooms.rooms[0]
            # Subtract 1 for the backend server participant so the frontend
            # shows how many *human* participants are in the room.
            human_count = max(0, r.num_participants - 1)
            return {
                "name":              r.name,
                "sid":               r.sid,
                "num_participants":  r.num_participants,
                "participant_count": human_count,   # human-only count for UI
                "active_recording":  r.active_recording,
                "exists":            True,
            }
    except Exception as exc:
        log.warning(f"[livekit] get_room({session_id}) failed: {exc}")
    return None


async def delete_room(session_id: str) -> bool:
    """Delete a LiveKit room and trigger subscriber task shutdown."""
    if not LIVEKIT_AVAILABLE:
        return False
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            await api.room.delete_room(lk_rtc.DeleteRoomRequest(name=session_id))
        log.info(f"[livekit] room {session_id} deleted")
        return True
    except Exception as exc:
        log.warning(f"[livekit] delete_room({session_id}) failed: {exc}")
        return False


# ── Active subscriber tasks ───────────────────────────────────────────────────
# Maps session_id → asyncio.Task (the combined audio+video subscriber loop)
_subscriber_tasks: dict[str, asyncio.Task] = {}


def start_subscriber(session_id: str, pipeline_fn) -> None:
    """
    Launch a background asyncio task that connects to the LiveKit room
    as a server-side participant and pipes audio/video into pipeline_fn.

    pipeline_fn signature:
        async def pipeline_fn(session_id: str, audio_frame_queue: asyncio.Queue,
                              video_frame_queue: asyncio.Queue) -> None

    server.py passes its own coroutine that drives VadChunker + WhisperX
    (audio) and vision.analyse_frame (video).
    """
    if session_id in _subscriber_tasks:
        log.warning(f"[livekit] subscriber for {session_id} already running")
        return

    task = asyncio.create_task(
        _subscriber_loop(session_id, pipeline_fn),
        name=f"lk-sub-{session_id}",
    )
    _subscriber_tasks[session_id] = task
    task.add_done_callback(lambda t: _subscriber_tasks.pop(session_id, None))
    log.info(f"[livekit] subscriber task started for {session_id}")


async def stop_subscriber(session_id: str) -> None:
    """Cancel the subscriber task for a session (called on room DELETE)."""
    task = _subscriber_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    log.info(f"[livekit] subscriber task stopped for {session_id}")


async def _subscriber_loop(session_id: str, pipeline_fn) -> None:
    """
    Internal: connect to the LiveKit room as 'lab-brain-server', subscribe to
    the first audio + video tracks published by any participant, and relay
    frames into separate asyncio queues consumed by pipeline_fn.
    """
    if not LIVEKIT_AVAILABLE:
        log.error("[livekit] Cannot start subscriber — SDK not installed")
        return

    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    video_q: asyncio.Queue = asyncio.Queue(maxsize=32)

    server_token = create_token(
        session_id,
        identity="lab-brain-server",
        ttl_seconds=7200,
    )

    SERVER_IDENTITY = "lab-brain-server"

    room = lk_rtc.Room()

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        # Ignore tracks published by the backend itself (shouldn't happen,
        # but guard anyway) and any future server-side participants.
        if participant.identity == SERVER_IDENTITY:
            return
        log.info(
            f"[livekit:{session_id}] track subscribed: "
            f"kind={track.kind} participant={participant.identity}"
        )
        if track.kind == lk_rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(_drain_audio(track, audio_q))
        elif track.kind == lk_rtc.TrackKind.KIND_VIDEO:
            asyncio.ensure_future(_drain_video(track, video_q))

    @room.on("disconnected")
    def on_disconnect(reason=None):
        log.info(f"[livekit:{session_id}] room disconnected: {reason}")

    try:
        await room.connect(cfg.livekit.url, server_token)
        log.info(f"[livekit:{session_id}] server participant connected to room")

        # Run the caller's pipeline coroutine; it consumes audio_q and video_q
        await pipeline_fn(session_id, audio_q, video_q)

    except asyncio.CancelledError:
        log.info(f"[livekit:{session_id}] subscriber cancelled")
    except Exception as exc:
        log.error(f"[livekit:{session_id}] subscriber error: {exc}", exc_info=True)
    finally:
        await room.disconnect()


async def _drain_audio(track: "lk_rtc.RemoteAudioTrack", q: asyncio.Queue) -> None:
    """
    Read AudioFrames from a LiveKit audio track and push raw PCM float32
    bytes into the queue.  LiveKit delivers 10ms frames at 48kHz stereo;
    server.py resamples to 16kHz mono before passing to VadChunker.
    """
    audio_stream = lk_rtc.AudioStream(track)
    async for event in audio_stream:
        frame: lk_rtc.AudioFrame = event.frame
        try:
            # frame.data is bytes of int16 PCM at frame.sample_rate, frame.num_channels
            q.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # drop if pipeline is lagging


async def _drain_video(track: "lk_rtc.RemoteVideoTrack", q: asyncio.Queue) -> None:
    """
    Read VideoFrames from a LiveKit video track and push JPEG bytes into
    the queue.  server.py throttles to VISION_FRAME_INTERVAL before calling
    vision.analyse_frame().
    """
    video_stream = lk_rtc.VideoStream(track, format=lk_rtc.VideoBufferType.JPEG)
    async for event in video_stream:
        frame: lk_rtc.VideoFrame = event.frame
        try:
            q.put_nowait(bytes(frame.data))
        except asyncio.QueueFull:
            pass  # throttle — vision pipeline is slower than video FPS
