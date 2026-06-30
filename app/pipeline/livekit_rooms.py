"""
app/pipeline/livekit_rooms.py — LiveKit integration layer.

Responsibilities:
  1. Token generation  — sign JWTs for browser participants.
  2. Room management   — create / inspect / close rooms via LiveKit API.
  3. Audio subscriber  — pipe PCM from LiveKit into VadChunker → WhisperX.
  4. Video subscriber  — pipe JPEG frames into vision.analyse_frame().
  5. SSE broadcaster   — fan-out events to all connected browser clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timedelta
from typing import AsyncIterator, Optional

from app.core.config import cfg

log = logging.getLogger(__name__)

LIVEKIT_AVAILABLE = False
try:
    from livekit import rtc as lk_rtc
    from livekit.api import (
        AccessToken,
        VideoGrants,
        LiveKitAPI,
        CreateRoomRequest,
        ListRoomsRequest,
        DeleteRoomRequest,
    )
    LIVEKIT_AVAILABLE = True
    log.info("[livekit] SDK loaded.")
except ImportError:
    log.warning(
        "[livekit] 'livekit' and/or 'livekit-api' not installed. "
        "Run: pip install livekit livekit-api"
    )


# ── SSE event bus ─────────────────────────────────────────────────────────────
_sse_subscribers: dict[str, list[asyncio.Queue]] = {}


def _get_queues(session_id: str) -> list[asyncio.Queue]:
    return _sse_subscribers.setdefault(session_id, [])


def broadcast(session_id: str, event: dict) -> None:
    for q in _get_queues(session_id):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


async def sse_stream(session_id: str) -> AsyncIterator[str]:
    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    queues = _get_queues(session_id)
    queues.append(q)
    try:
        yield f"event: connected\ndata: {json.dumps({'session_id': session_id})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=25.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    finally:
        try:
            queues.remove(q)
        except ValueError:
            pass


# ── Token generation ──────────────────────────────────────────────────────────

def create_token(session_id: str, identity: str, ttl_seconds: int = 3600) -> str:
    if not LIVEKIT_AVAILABLE:
        raise RuntimeError("livekit-api package not installed")
    grants = VideoGrants(
        room_join=True,
        room=session_id,
        can_publish=True,
        can_subscribe=True,
    )
    return (
        AccessToken(cfg.livekit.api_key, cfg.livekit.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=ttl_seconds))
        .to_jwt()
    )


# ── Room management ───────────────────────────────────────────────────────────

async def create_room(session_id: str) -> dict:
    if not LIVEKIT_AVAILABLE:
        raise RuntimeError("livekit-api package not installed")
    async with LiveKitAPI(
        url=cfg.livekit.url,
        api_key=cfg.livekit.api_key,
        api_secret=cfg.livekit.api_secret,
    ) as api:
        room = await api.room.create_room(
            CreateRoomRequest(name=session_id, empty_timeout=300)
        )
    return {"name": room.name, "sid": room.sid}


async def get_room(session_id: str) -> Optional[dict]:
    if not LIVEKIT_AVAILABLE:
        return None
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            rooms = await api.room.list_rooms(ListRoomsRequest(names=[session_id]))
            if not rooms.rooms:
                return None
            r          = rooms.rooms[0]
            human_count = max(0, r.num_participants - 1)
            return {
                "name":              r.name,
                "sid":               r.sid,
                "num_participants":  r.num_participants,
                "participant_count": human_count,
                "active_recording":  r.active_recording,
                "exists":            True,
            }
    except Exception as exc:
        log.warning(f"[livekit] get_room({session_id}) failed: {exc}")
    return None


async def delete_room(session_id: str) -> bool:
    if not LIVEKIT_AVAILABLE:
        return False
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            await api.room.delete_room(DeleteRoomRequest(name=session_id))
        log.info(f"[livekit] room {session_id} deleted")
        return True
    except Exception as exc:
        log.warning(f"[livekit] delete_room({session_id}) failed: {exc}")
        return False


# ── Active subscriber tasks ───────────────────────────────────────────────────
_subscriber_tasks: dict[str, asyncio.Task] = {}


def start_subscriber(session_id: str, pipeline_fn) -> None:
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
    task = _subscriber_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    log.info(f"[livekit] subscriber task stopped for {session_id}")


async def _subscriber_loop(session_id: str, pipeline_fn) -> None:
    if not LIVEKIT_AVAILABLE:
        log.error("[livekit] Cannot start subscriber — SDK not installed")
        return

    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    video_q: asyncio.Queue = asyncio.Queue(maxsize=32)

    server_token   = create_token(session_id, identity="lab-brain-server", ttl_seconds=7200)
    SERVER_IDENTITY = "lab-brain-server"
    room = lk_rtc.Room()

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
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
        log.info(f"[livekit:{session_id}] server participant connected")
        await pipeline_fn(session_id, audio_q, video_q)
    except asyncio.CancelledError:
        log.info(f"[livekit:{session_id}] subscriber cancelled")
    except Exception as exc:
        log.error(f"[livekit:{session_id}] subscriber error: {exc}", exc_info=True)
    finally:
        await room.disconnect()


async def _drain_audio(track, q: asyncio.Queue) -> None:
    audio_stream = lk_rtc.AudioStream(track)
    async for event in audio_stream:
        try:
            q.put_nowait(event.frame)
        except asyncio.QueueFull:
            pass


async def _drain_video(track, q: asyncio.Queue) -> None:
    video_stream = lk_rtc.VideoStream(track, format=lk_rtc.VideoBufferType.JPEG)
    async for event in video_stream:
        try:
            q.put_nowait(bytes(event.frame.data))
        except asyncio.QueueFull:
            pass