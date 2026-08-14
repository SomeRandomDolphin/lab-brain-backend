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
        RoomCompositeEgressRequest,
        EncodedFileOutput,
        EncodedFileType,
        EncodingOptions,
        ListEgressRequest,
        EgressStatus,
        StopEgressRequest,
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


# ── Egress (room-composite recording → local disk → Supabase Storage) ────────
#
# This uses LiveKit's server-side Egress feature rather than hand-rolling a
# JPEG-frame/audio-segment uploader: Egress records the actual room composite
# (matching what a participant would hear). NOTE: start_egress() below sets
# audio_only=True — this is audio-only now (video encoding was cut for file
# size), so "composite" here means "mixed audio from all participants", not
# audio+video. Written as .mp3 (audio only, no video track) rather than
# .mp4 — an MP4 with only an AAC audio track is valid but is a video-shaped
# container, which can render as a blank/broken player if anything ever
# plays it back via <video> instead of <audio>. MP3 sidesteps that and
# gets a native audio player in virtually every browser without relying on
# the frontend picking the right tag.
#
# NOT going through egress's built-in S3 upload (S3Upload) anymore: self-
# hosted Supabase Storage's S3-compatible endpoint has a longstanding,
# unresolved upstream bug where every S3-protocol request — regardless of
# client (aws-sdk-go-v2, boto3, aws-cli), regardless of proxy/network path —
# fails signature verification with SignatureDoesNotMatch, even with
# byte-verified-correct credentials. See:
#   https://github.com/supabase/storage/issues/572
#   https://github.com/supabase/storage/issues/646
# No client-side config change works around it. Instead: egress writes the
# finished recording to local disk on a Docker volume SHARED between the
# egress container and this app's container (RECORDINGS_MOUNT below — must
# match the mount point in both start_services.sh's start_egress() and
# docker-compose.yml's backend service), and stop_egress() picks the file up
# from there once egress reports it complete, then uploads it via the
# JWT-authenticated Supabase Storage API — the same path report.md/
# summary.md already use successfully (see app/db/supabase_client.py).
#
# NOTE ON CONFIG: this still expects the following to exist on `cfg`:
#   cfg.livekit.egress_enabled     (bool)
#
# The cfg.supabase.s3_* fields added for the old S3 upload path are no
# longer read here — they can stay in config.py/​.env harmlessly (unused)
# or be removed once you're confident nothing else references them.

# Must match the volume mount point used for the egress container in
# start_services.sh's start_egress() AND the `backend` service in
# docker-compose.yml — a mismatch here means this app looks for finished
# recordings somewhere egress never wrote them.
RECORDINGS_MOUNT = "/recordings"

# How long to wait, after StopEgressRequest is accepted, for egress to
# actually finish flushing the file to disk and report EGRESS_COMPLETE.
# StopEgressRequest returns as soon as the stop is ACCEPTED, not once the
# file is fully written — uploading before that risks a half-written file.
# 30s is generous for a local-disk flush (there's no network upload for
# egress itself to wait on anymore, unlike the old S3 path).
_EGRESS_POLL_INTERVAL = 1.0
_EGRESS_POLL_TIMEOUT = 30.0

async def start_egress(session_id: str) -> Optional[str]:
    """Start a room-composite (single mixed file) recording, written to
    local disk on the shared RECORDINGS_MOUNT volume. Returns the egress_id
    (used to stop it later) or None if egress is disabled/unavailable.
    """
    if not LIVEKIT_AVAILABLE or not getattr(cfg.livekit, "egress_enabled", False):
        return None

    relative_path = f"{session_id}/{session_id}-{int(time.time())}.mp3"
    file_output = EncodedFileOutput(
        file_type=EncodedFileType.MP3,
        filepath=f"{RECORDINGS_MOUNT}/{relative_path}",
    )
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            info = await api.egress.start_room_composite_egress(
                RoomCompositeEgressRequest(
                    room_name=session_id,
                    # Audio-only — drops video encoding entirely, by far
                    # the biggest lever for file size (video was still the
                    # dominant cost even at 960x540/800kbps). video_only
                    # must stay False; it's the opposite toggle and the two
                    # are mutually exclusive.
                    audio_only=True,
                    video_only=False,
                    # width/height/framerate/video_bitrate no longer apply
                    # with audio_only=True — only audio_bitrate matters now.
                    advanced=EncodingOptions(
                        audio_bitrate=64_000,
                    ),
                    file_outputs=[file_output],
                )
            )
        _egress_ids[session_id] = info.egress_id
        _egress_paths[info.egress_id] = f"{RECORDINGS_MOUNT}/{relative_path}"
        log.info(f"[livekit:{session_id}] egress started: egress_id={info.egress_id}")
        return info.egress_id
    except Exception as exc:
        log.error(f"[livekit:{session_id}] start_egress failed: {exc!r}", exc_info=True)
        return None


async def _wait_for_egress_complete(api: "LiveKitAPI", egress_id: str):
    """Poll list_egress until *egress_id* reaches a terminal state, or
    _EGRESS_POLL_TIMEOUT elapses (returns None in that case)."""
    elapsed = 0.0
    while elapsed < _EGRESS_POLL_TIMEOUT:
        result = await api.egress.list_egress(ListEgressRequest(egress_id=egress_id))
        if result.items:
            info = result.items[0]
            if info.status in (
                EgressStatus.EGRESS_COMPLETE,
                EgressStatus.EGRESS_FAILED,
                EgressStatus.EGRESS_ABORTED,
            ):
                return info
        await asyncio.sleep(_EGRESS_POLL_INTERVAL)
        elapsed += _EGRESS_POLL_INTERVAL
    return None


async def stop_egress(session_id: str) -> None:
    egress_id = _egress_ids.pop(session_id, None)
    if not egress_id:
        return
    local_path = _egress_paths.pop(egress_id, None)

    info = None
    try:
        async with LiveKitAPI(
            url=cfg.livekit.url,
            api_key=cfg.livekit.api_key,
            api_secret=cfg.livekit.api_secret,
        ) as api:
            await api.egress.stop_egress(StopEgressRequest(egress_id=egress_id))
            log.info(f"[livekit:{session_id}] egress stopped: egress_id={egress_id}")
            info = await _wait_for_egress_complete(api, egress_id)
    except Exception as exc:
        log.warning(f"[livekit:{session_id}] stop_egress failed: {exc!r}")
        return

    if info is None:
        log.warning(
            f"[livekit:{session_id}] egress {egress_id} did not reach a terminal "
            f"state within {_EGRESS_POLL_TIMEOUT}s — skipping upload"
        )
        return

    if info.status != EgressStatus.EGRESS_COMPLETE:
        log.warning(
            f"[livekit:{session_id}] egress {egress_id} ended with "
            f"status={info.status!r} error={info.error!r} — skipping upload"
        )
        return

    # Prefer the path LiveKit itself reports over the one we requested —
    # they should match, but file_results is the source of truth if they
    # ever diverge (e.g. a filename-collision suffix LiveKit adds).
    reported_path = info.file_results[0].filename if info.file_results else None
    path_to_upload = reported_path or local_path
    if not path_to_upload:
        log.error(
            f"[livekit:{session_id}] egress {egress_id} completed but no file "
            "path is known — cannot upload recording"
        )
        return

    from app.db.supabase_client import upload_recording
    url = await upload_recording(session_id, path_to_upload)
    if url:
        log.info(f"[livekit:{session_id}] recording uploaded: {url}")
    else:
        log.error(f"[livekit:{session_id}] recording upload failed for {path_to_upload}")


# ── Active subscriber tasks ───────────────────────────────────────────────────
_subscriber_tasks: dict[str, asyncio.Task] = {}

# Strong references to per-session drain tasks so the GC cannot silently
# collect them while they are still awaiting frames from the LiveKit streams.
_drain_tasks: dict[str, list[asyncio.Task]] = {}

# The real display name of each (non-server) participant currently in the
# room, keyed by (session_id -> participant_identity -> display_name).
# Populated from LiveKit's own participant object — which carries the name
# set via create_token()'s display_name — rather than anything vision.py can
# see, since the VLM only ever produces generic "Person A"/"Person B" labels
# with no notion of who that actually is.
# Session_pipeline reads this to substitute the real name in place of the
# generic label for whoever is recognized as the account holder.
#
# NOTE: this used to be dict[session_id, str] — a single slot per session.
# In a multi-participant room, every participant's track-subscribed event
# overwrote that one slot, so whichever participant's track happened to
# subscribe last silently clobbered everyone else's name. Keying by
# (session_id, participant_identity) keeps each participant's identity
# independent.
_known_identities: dict[str, dict[str, str]] = {}


# Active LiveKit Egress recording IDs, keyed by session_id, so the recording
# can be stopped explicitly when the session ends rather than relying on
# room deletion to implicitly kill it (which risks losing the tail of the
# recording depending on how LiveKit's server handles that).
_egress_ids: dict[str, str] = {}

# Local filesystem path (on the shared RECORDINGS_MOUNT volume) each active
# egress_id is writing to — stop_egress() needs this as a fallback if
# EgressInfo.file_results is ever empty/missing on completion.
_egress_paths: dict[str, str] = {}


def get_known_identity(session_id: str, participant_identity: Optional[str] = None) -> Optional[str]:
    identities = _known_identities.get(session_id, {})
    if participant_identity is not None:
        return identities.get(participant_identity)
    # Back-compat / single-participant convenience: if exactly one identity
    # is known for this session, return it; otherwise there's no single
    # answer, so don't guess.
    if len(identities) == 1:
        return next(iter(identities.values()))
    return None


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
        _known_identities.setdefault(session_id, {})[participant.identity] = (
            participant.name or participant.identity
        )
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

        # Every track is tagged with participant.identity when it's pushed
        # onto the shared queue. Previously this pushed the bare frame/jpeg
        # with no indication of which participant it came from — with one
        # remote participant that's harmless, but with several, frames from
        # different people's tracks landed in the same queue indistinguishable
        # from one another, which is what let them get merged into a single
        # VadChunker/vision slot downstream as if they were one continuous
        # feed. session_pipeline.py now keys per-participant state off this.
        if track.kind == lk_rtc.TrackKind.KIND_AUDIO:
            t = asyncio.create_task(_drain_audio(participant.identity, track, audio_q))
        elif track.kind == lk_rtc.TrackKind.KIND_VIDEO:
            t = asyncio.create_task(_drain_video(participant.identity, track, video_q))
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
        # Was WARNING — fired on every transition including the completely
        # normal 0->1 (connected), so a routine connect logged identically
        # to something actually worth noticing. on_disconnect and
        # on_reconnecting above already flag the transitions that matter;
        # this one is just informational.
        log.info(f"[livekit:{session_id}] connection_state_changed -> {state!r}")

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
                await start_egress(session_id)
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
        log.info(f"[livekit:{session_id}] finally: stopping egress (if any) and disconnecting")
        await stop_egress(session_id)
        await room.disconnect()


async def _drain_audio(participant_identity: str, track, q: asyncio.Queue) -> None:
    audio_stream = lk_rtc.AudioStream(track)
    dropped = 0
    async for event in audio_stream:
        try:
            q.put_nowait((participant_identity, event.frame))
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


async def _drain_video(participant_identity: str, track, q: asyncio.Queue) -> None:
    video_stream = lk_rtc.VideoStream(track, format=lk_rtc.VideoBufferType.I420)
    dropped = 0
    frame_counter = 0
    loop = asyncio.get_event_loop()
    async for event in video_stream:
        # Apply VISION_FRAME_INTERVAL here, BEFORE the RGBA convert + JPEG
        # encode — this used to happen downstream in session_pipeline.py's
        # _process_video, after every single incoming frame (typically
        # 15-30fps from the browser) had already paid the CPU cost of
        # _encode_frame via the shared default executor. That executor is
        # also used by WhisperX ASR and diarization, so encoding frames that
        # were just going to be discarded 4 times out of 5 (frame_interval=5)
        # was pure wasted contention against the audio pipeline. Only frames
        # that will actually be used ever get encoded now.
        frame_counter += 1
        if frame_counter % cfg.vision.frame_interval != 0:
            continue

        frame = event.frame
        rgba = frame.convert(lk_rtc.VideoBufferType.RGBA)
        # PIL conversion + JPEG encode is CPU-bound. Doing it inline used to
        # block this loop from pulling the next frame off the LiveKit stream,
        # which is why the SDK's own internal buffer overflowed within a
        # couple of seconds of subscribing — before any ASR/diarization work
        # even started. Offloading it lets frames keep draining in real time.
        jpeg_bytes = await loop.run_in_executor(None, _encode_frame, rgba)
        try:
            q.put_nowait((participant_identity, jpeg_bytes))
        except asyncio.QueueFull:
            dropped += 1
            if dropped == 1 or dropped % 50 == 0:
                log.warning(
                    f"[livekit] video_q full — dropped {dropped} frame(s) so far "
                    f"(consumer is falling behind real time)"
                )