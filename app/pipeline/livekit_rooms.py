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
import io
from datetime import timedelta
from typing import AsyncIterator, Optional
from PIL import Image

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

def create_token(session_id: str, identity: str, ttl_seconds: int = 3600, display_name: Optional[str] = None) -> str:
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
        .with_name(display_name or identity)
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
            await api.room.delete_room(DeleteRoomRequest(room=session_id))
        log.info(f"[livekit] room {session_id} deleted")
        return True
    except Exception as exc:
        log.warning(f"[livekit] delete_room({session_id}) failed: {exc}")
        return False


# ── Active subscriber tasks ───────────────────────────────────────────────────
_subscriber_tasks: dict[str, asyncio.Task] = {}

# Strong references to per-session drain tasks so the GC cannot silently
# collect them while they are still awaiting frames from the LiveKit streams.
_drain_tasks: dict[str, list[asyncio.Task]] = {}

# The real display name of the (non-server) participant currently in the
# room, keyed by session_id. Populated from LiveKit's own participant object
# — which carries the name set via create_token()'s display_name — rather
# than anything vision.py can see, since the VLM only ever produces generic
# "Person A"/"Person B" labels with no notion of who that actually is.
# Session_pipeline reads this to substitute the real name in place of the
# generic label for whoever is recognized as the account holder.
_known_identities: dict[str, str] = {}


def get_known_identity(session_id: str) -> Optional[str]:
    return _known_identities.get(session_id)


def start_subscriber(session_id: str, pipeline_fn) -> None:
    if session_id in _subscriber_tasks:
        log.warning(f"[livekit] subscriber for {session_id} already running")
        return
    task = asyncio.create_task(
        _subscriber_loop(session_id, pipeline_fn),
        name=f"lk-sub-{session_id}",
    )
    _subscriber_tasks[session_id] = task

    def _on_subscriber_done(t: asyncio.Task) -> None:
        _subscriber_tasks.pop(session_id, None)
        if t.cancelled():
            log.warning(f"[livekit:{session_id}] subscriber task was CANCELLED")
            return
        exc = t.exception()
        if exc:
            import traceback
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log.error(
                f"[livekit:{session_id}] *** SUBSCRIBER TASK CRASHED ***\n"
                f"  type : {type(exc).__name__}\n"
                f"  value: {exc!r}\n"
                f"  traceback:\n{tb}"
            )
        else:
            log.warning(
                f"[livekit:{session_id}] subscriber task finished WITHOUT exception "
                f"(pipeline_fn returned or an except clause swallowed the error)"
            )

    task.add_done_callback(_on_subscriber_done)
    log.info(f"[livekit] subscriber task started for {session_id}")


async def stop_subscriber(session_id: str) -> None:
    task = _subscriber_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Cancel any drain tasks that are still alive and drop the strong refs.
    for drain in _drain_tasks.pop(session_id, []):
        if not drain.done():
            drain.cancel()

    _known_identities.pop(session_id, None)

    log.info(f"[livekit] subscriber task stopped for {session_id}")


async def _subscriber_loop(session_id: str, pipeline_fn) -> None:
    if not LIVEKIT_AVAILABLE:
        log.error("[livekit] Cannot start subscriber — SDK not installed")
        return

    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    video_q: asyncio.Queue = asyncio.Queue(maxsize=32)

    try:
        server_token = create_token(
            session_id, identity="lab-brain-server", ttl_seconds=7200,
            display_name="Lab Brain Agent",
        )
    except Exception as exc:
        log.error(f"[livekit:{session_id}] create_token (server identity) failed: {exc}", exc_info=True)
        return

    SERVER_IDENTITY = "lab-brain-server"
    room = lk_rtc.Room()

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if participant.identity == SERVER_IDENTITY:
            return
        # participant.name is whatever create_token()'s display_name set —
        # i.e. the real account name once useSession.ts/livekit.py pass it
        # through, falling back to the raw identity for older/guest tokens
        # that never got a real display_name.
        _known_identities[session_id] = participant.name or participant.identity
        log.info(
            f"[livekit:{session_id}] track subscribed: kind={track.kind} "
            f"participant={participant.identity} muted={publication.muted}"
        )

        def _on_drain_done(t: asyncio.Task, kind=track.kind) -> None:  # noqa: B023
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                log.error(
                    f"[livekit:{session_id}] drain task ({kind}) crashed: {exc!r}",
                    exc_info=exc,
                )

        if track.kind == lk_rtc.TrackKind.KIND_AUDIO:
            t = asyncio.create_task(_drain_audio(track, audio_q))
        elif track.kind == lk_rtc.TrackKind.KIND_VIDEO:
            t = asyncio.create_task(_drain_video(track, video_q))
        else:
            return

        # Keep a strong reference so the GC cannot silently cancel the task
        # while it is awaiting frames. Cleaned up in stop_subscriber().
        _drain_tasks.setdefault(session_id, []).append(t)

        def _on_drain_done_and_remove(
            finished: asyncio.Task,
            kind=track.kind,
            session=session_id,
        ) -> None:
            _on_drain_done(finished, kind)
            try:
                _drain_tasks.get(session, []).remove(finished)
            except ValueError:
                pass

        t.add_done_callback(_on_drain_done_and_remove)

    @room.on("track_muted")
    def on_track_muted(participant, publication):
        log.warning(f"[livekit:{session_id}] track MUTED: {participant.identity} / {publication.kind}")

    @room.on("track_unmuted")
    def on_track_unmuted(participant, publication):
        log.warning(f"[livekit:{session_id}] track UNMUTED: {participant.identity} / {publication.kind}")

    @room.on("disconnected")
    def on_disconnect(reason=None):
        # This fires for BOTH the normal, expected case (our own
        # room.disconnect() call in the `finally` block below, once a
        # session ends cleanly) and genuine failures — but it was logging
        # every single one at ERROR, so every ordinary end-of-meeting showed
        # up indistinguishable from an actual problem. reason 0/1
        # (UNKNOWN/CLIENT_INITIATED in LiveKit's protocol) covers the
        # expected paths; anything else is more likely a real disconnect
        # (network drop, server-side issue, etc.) and stays loud.
        reason_val = getattr(reason, "value", reason)
        if reason_val in (0, 1):
            log.info(f"[livekit:{session_id}] room disconnected (reason={reason!r})")
        else:
            log.error(f"[livekit:{session_id}] *** ROOM DISCONNECTED *** reason={reason!r}")

    @room.on("connection_state_changed")
    def on_conn_state(state):
        log.warning(f"[livekit:{session_id}] connection_state_changed -> {state!r}")

    @room.on("reconnecting")
    def on_reconnecting():
        log.warning(f"[livekit:{session_id}] reconnecting...")

    @room.on("participant_disconnected")
    def on_participant_left(participant):
        log.warning(f"[livekit:{session_id}] participant left: {participant.identity!r}")

    _MAX_CONNECT_ATTEMPTS = 3
    _CONNECT_BACKOFF_BASE = 2.0  # seconds; attempt n waits backoff_base ** (n-1)

    try:
        for attempt in range(1, _MAX_CONNECT_ATTEMPTS + 1):
            try:
                log.info(
                    f"[livekit:{session_id}] room.connect attempt {attempt}/{_MAX_CONNECT_ATTEMPTS} "
                    f"→ {cfg.livekit.url!r}"
                )
                await room.connect(cfg.livekit.url, server_token)
                log.info(
                    f"[livekit:{session_id}] *** CONNECTED *** "
                    f"sid={room.local_participant.sid}"
                )
                break  # success
            except asyncio.CancelledError:
                raise
            except Exception as connect_exc:
                if attempt == _MAX_CONNECT_ATTEMPTS:
                    log.error(
                        f"[livekit:{session_id}] room.connect failed after "
                        f"{_MAX_CONNECT_ATTEMPTS} attempts: {connect_exc!r}"
                    )
                    raise
                wait = _CONNECT_BACKOFF_BASE ** (attempt - 1)
                log.warning(
                    f"[livekit:{session_id}] room.connect attempt {attempt} failed "
                    f"({connect_exc!r}), retrying in {wait:.0f}s …"
                )
                await asyncio.sleep(wait)

        log.info(f"[livekit:{session_id}] calling pipeline_fn={pipeline_fn!r} ...")
        await pipeline_fn(session_id, audio_q, video_q)
        log.info(f"[livekit:{session_id}] pipeline_fn returned normally (unexpected)")
    except asyncio.CancelledError:
        log.info(f"[livekit:{session_id}] subscriber CancelledError (clean shutdown)")
        raise
    except Exception as exc:
        log.error(
            f"[livekit:{session_id}] *** SUBSCRIBER CRASHED *** "
            f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
    finally:
        log.info(f"[livekit:{session_id}] finally: calling room.disconnect()")
        await room.disconnect()


async def _drain_audio(track, q: asyncio.Queue) -> None:
    audio_stream = lk_rtc.AudioStream(track)
    dropped = 0
    async for event in audio_stream:
        try:
            q.put_nowait(event.frame)
        except asyncio.QueueFull:
            dropped += 1
            if dropped == 1 or dropped % 50 == 0:
                log.warning(
                    f"[livekit] audio_q full — dropped {dropped} frame(s) so far "
                    f"(consumer is falling behind real time)"
                )


def _encode_frame(rgba) -> bytes:
    img = Image.frombuffer(
        "RGBA", (rgba.width, rgba.height), rgba.data, "raw", "RGBA", 0, 1
    )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=70)
    return buf.getvalue()


async def _drain_video(track, q: asyncio.Queue) -> None:
    video_stream = lk_rtc.VideoStream(track, format=lk_rtc.VideoBufferType.I420)
    dropped = 0
    loop = asyncio.get_event_loop()
    async for event in video_stream:
        frame = event.frame
        rgba = frame.convert(lk_rtc.VideoBufferType.RGBA)
        # PIL conversion + JPEG encode is CPU-bound. Doing it inline used to
        # block this loop from pulling the next frame off the LiveKit stream,
        # which is why the SDK's own internal buffer overflowed within a
        # couple of seconds of subscribing — before any ASR/diarization work
        # even started. Offloading it lets frames keep draining in real time.
        jpeg_bytes = await loop.run_in_executor(None, _encode_frame, rgba)
        try:
            q.put_nowait(jpeg_bytes)
        except asyncio.QueueFull:
            dropped += 1
            if dropped == 1 or dropped % 50 == 0:
                log.warning(
                    f"[livekit] video_q full — dropped {dropped} frame(s) so far "
                    f"(consumer is falling behind real time)"
                )