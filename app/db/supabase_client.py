"""
app/db/supabase_client.py — Supabase persistence layer (SQLAlchemy async ORM).

All runtime data access goes through SQLAlchemy with asyncpg, talking directly
to the Postgres instance that backs your Supabase project. The supabase-py
client is retained only for Storage bucket operations (audio segments and
report exports), which have no SQLAlchemy equivalent.

What goes where
---------------
Postgres tables (SQLAlchemy ORM — see models.py):
  sessions             — one row per LiveKit room / Lab Brain session (owned by user_id)
  session_participants — authenticated users who joined a session they don't own
  transcripts          — one row per WhisperX segment (text + tags + word timestamps)
  agent_replies        — one row per LLM reply
  vision_frames        — one row per analysed camera frame
  session_summaries    — one row per end-of-session LLM summary
  eval_metrics         — one row per session metric snapshot
  consent_registry     — speaker consent records, scoped per session

Supabase Storage buckets (supabase-py):
  audio-segments    — raw float32 PCM blobs
  report-exports    — generated markdown summary exports
  recordings        — LiveKit Egress room-composite recordings (uploaded via
                       upload_recording() — see its docstring for why this
                       goes through the JWT Storage API, not the S3 protocol)

Required environment variables
-------------------------------
  SUPABASE_DB_URL      postgresql+asyncpg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres
  SUPABASE_URL         https://<project>.supabase.co          (Storage only)
  SUPABASE_SERVICE_KEY service_role or anon key                (Storage only)

Schema migrations
-----------------
Handled by Alembic — see app/db/migrations.py. This module is pure runtime
read/write only.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, or_, text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import (
    AgentReply,
    ConsentRegistry,
    EvalMetrics,
    Session as SessionModel,
    SessionParticipant,
    SessionSummary,
    Transcript,
    VisionFrame,
)

log = logging.getLogger(__name__)

# ── Engine (lazy, singleton) ───────────────────────────────────────────────────

_engine = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _get_engine():
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    db_url = os.environ.get("SUPABASE_DB_URL", "")
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. "
            "Set it to your Supabase Postgres connection string "
            "(postgresql+asyncpg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres)."
        )

    _engine = create_async_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    log.info("[supabase] async engine initialised")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    _get_engine()
    return _session_factory


# ── Supabase Storage client (for buckets only) ────────────────────────────────

_storage_client = None


def _get_storage_client():
    global _storage_client
    if _storage_client is not None:
        return _storage_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        log.warning("[supabase] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — Storage disabled.")
        return None

    try:
        from supabase import create_client
        _storage_client = create_client(url, key)
        return _storage_client
    except Exception as exc:
        log.warning(f"[supabase] Storage client init failed: {exc}")
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _from_unix(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _safe_storage_upload(
    bucket: str, path: str, data: bytes, content_type: str
) -> Optional[str]:
    client = _get_storage_client()
    if client is None:
        return None
    try:
        client.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return client.storage.from_(bucket).get_public_url(path)
    except Exception as exc:
        log.warning(f"[supabase] storage upload {bucket}/{path} failed: {exc}")
        return None


def _fire_sync(fn, *args) -> None:
    """
    Run a sync function (Storage uploads) in a thread-pool so it never
    blocks the async event loop.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, fn, *args)
    except RuntimeError:
        fn(*args)


async def _ensure_session_row(session_id: str, db: AsyncSession, user_id: Optional[str] = None) -> None:
    """
    Guarantee a sessions row exists for *session_id* without overwriting any
    real session data that may already be there.

    Called by upsert_session_summary and upsert_eval_metrics, which can be
    invoked by the frontend before POST /livekit/room has run its DB write
    (the room itself is already created by then, since these are teardown /
    mid-session calls, but this stays a defensive fallback).

    As of migration 0007, sessions.user_id is NOT NULL, so this can no
    longer insert a bare fallback row without an owner. In every real call
    path here, `user_id` is available from the caller (the authenticated
    request that led here) — pass it through. If it is ever missing, the
    INSERT is skipped and the caller's own FK-dependent write will fail
    loudly with a clear FK error rather than silently creating an unowned
    session.
    """
    if user_id is None:
        return
    await db.execute(
        sa_text("""
            INSERT INTO sessions (session_id, user_id, host_identity, started_at, metadata, updated_at)
            VALUES (:sid, :uid, 'browser-user', NOW(), '{}', NOW())
            ON CONFLICT (session_id) DO NOTHING
        """),
        {"sid": session_id, "uid": user_id},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public write API
# ═══════════════════════════════════════════════════════════════════════════════

async def upsert_session(
    session_id: str,
    user_id: Optional[str] = None,
    host_identity: str = "browser-user",
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    livekit_room_sid: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    user_id is required on session *creation* (POST /livekit/room) and must
    be omitted on every later partial-update call (e.g. DELETE /livekit/room/{sid}
    setting ended_at) — passing user_id=None here means "leave whatever
    user_id already exists on the row untouched", not "set it to NULL".
    This is why user_id is excluded from `values`/the ON CONFLICT SET clause
    unless it was explicitly given: the ON CONFLICT DO UPDATE previously
    used a single `values` dict for both INSERT and UPDATE, which meant a
    teardown call with no user_id would have overwritten the owner to NULL
    (and NULL is no longer even legal post-migration-0007).
    """
    values = {
        "session_id":      session_id,
        "host_identity":   host_identity,
        "started_at":      _from_unix(started_at or time.time()),
        "livekit_room_sid": livekit_room_sid,
        "metadata_":       metadata or {},
        "updated_at":      _utcnow(),
    }
    if ended_at is not None:
        values["ended_at"] = _from_unix(ended_at)
    if user_id is not None:
        values["user_id"] = user_id

    update_values = {k: v for k, v in values.items() if k != "session_id"}

    if user_id is None:
        # Update-only path (e.g. teardown): the row must already exist.
        # Insert would fail anyway (user_id NOT NULL), so go straight to UPDATE.
        from sqlalchemy import update as sa_update
        stmt = (
            sa_update(SessionModel)
            .where(SessionModel.session_id == session_id)
            .values(**update_values)
        )
    else:
        stmt = (
            pg_insert(SessionModel)
            .values(**values)
            .on_conflict_do_update(index_elements=["session_id"], set_=update_values)
        )

    async with get_session_factory()() as db:
        await db.execute(stmt)
        await db.commit()


async def add_session_participant(session_id: str, user_id: str) -> None:
    """
    Record that `user_id` has joined `session_id` as a non-owner participant.
    Called on every successful GET /livekit/token. ON CONFLICT DO NOTHING —
    re-joining an already-recorded session is a no-op, not an error.
    """
    stmt = (
        pg_insert(SessionParticipant)
        .values(session_id=session_id, user_id=user_id)
        .on_conflict_do_nothing(index_elements=["session_id", "user_id"])
    )
    async with get_session_factory()() as db:
        await db.execute(stmt)
        await db.commit()


async def get_session_access(session_id: str) -> tuple[Optional[str], set[str]]:
    """
    Return (owner_user_id, {participant_user_ids}) for a session, or
    (None, set()) if the session doesn't exist. Used by
    app.api.deps.require_session_access / require_session_owner so the
    ownership check lives in one place instead of being duplicated per route.
    """
    async with get_session_factory()() as db:
        owner_row = await db.execute(
            select(SessionModel.user_id).where(SessionModel.session_id == session_id)
        )
        owner = owner_row.scalar_one_or_none()
        if owner is None:
            return None, set()

        participants_row = await db.execute(
            select(SessionParticipant.user_id).where(
                SessionParticipant.session_id == session_id
            )
        )
        participants = set(participants_row.scalars().all())

    return owner, participants


async def insert_transcript(
    session_id: str,
    speaker: str,
    text: str,
    language: str,
    mode: str,
    timestamp_unix: float,
    timestamp_iso: str,
    tags: dict,
    word_timestamps: list,
    asr_latency_ms: int = 0,
    e2e_latency_ms: int = 0,
    segment_index: int = 0,
) -> None:
    row = Transcript(
        session_id=session_id,
        segment_index=segment_index,
        speaker=speaker,
        text=text,
        language=language,
        mode=mode,
        timestamp_iso=timestamp_iso,
        timestamp_unix=round(timestamp_unix, 3),
        tags=tags,
        word_timestamps=word_timestamps,
        asr_latency_ms=asr_latency_ms,
        e2e_latency_ms=e2e_latency_ms,
        created_at=_utcnow(),
    )
    async with get_session_factory()() as db:
        db.add(row)
        await db.commit()


async def insert_agent_reply(
    session_id: str,
    text: str,
    mode: str,
    timestamp_unix: float,
    grounded: bool = False,
    lkc_context: str = "",
) -> None:
    row = AgentReply(
        session_id=session_id,
        text=text,
        mode=mode,
        timestamp_iso=_from_unix(timestamp_unix).isoformat(),
        timestamp_unix=round(timestamp_unix, 3),
        grounded=grounded,
        lkc_context=lkc_context,
        created_at=_utcnow(),
    )
    async with get_session_factory()() as db:
        db.add(row)
        await db.commit()


async def insert_vision_frame(
    session_id: str,
    timestamp_unix: float,
    scene_summary: str,
    present_speakers: list,
    engagement_cues: dict,
    environment_state: dict,
    latency_ms: int = 0,
) -> None:
    row = VisionFrame(
        session_id=session_id,
        timestamp_iso=_from_unix(timestamp_unix).isoformat(),
        timestamp_unix=round(timestamp_unix, 3),
        scene_summary=scene_summary,
        present_speakers=present_speakers,
        engagement_cues=engagement_cues,
        environment_state=environment_state,
        latency_ms=latency_ms,
        created_at=_utcnow(),
    )
    async with get_session_factory()() as db:
        db.add(row)
        await db.commit()


async def upsert_session_summary(
    session_id: str, summary_md: str, tags: dict, user_id: Optional[str] = None
) -> None:
    values = {
        "session_id": session_id,
        "summary_md": summary_md,
        "tags":       tags,
        "updated_at": _utcnow(),
    }
    # Use __table__ (the raw SA Table object) instead of the ORM class so that
    # SQLAlchemy does NOT auto-append "RETURNING session_summaries.id".  The ORM
    # model declares `id` as its primary key, but the actual table's PK is
    # session_id — a RETURNING on a non-existent column raises
    # asyncpg.UndefinedColumnError.  Table-level inserts never add RETURNING.
    stmt = (
        pg_insert(SessionSummary.__table__)
        .values(**values, created_at=_utcnow())
        .on_conflict_do_update(index_elements=["session_id"], set_=values)
    )
    async with get_session_factory()() as db:
        # Ensure the parent sessions row exists before inserting the FK-dependent
        # summary. As of migration 0007 this requires a user_id — pass the
        # current requester's id through from the endpoint.
        await _ensure_session_row(session_id, db, user_id=user_id)
        await db.execute(stmt)
        await db.commit()

    # Mirror to Storage bucket
    _fire_sync(
        _safe_storage_upload,
        "report-exports",
        f"{session_id}/summary.md",
        summary_md.encode("utf-8"),
        "text/markdown",
    )


async def upsert_eval_metrics(
    session_id: str, metrics_dict: dict, user_id: Optional[str] = None
) -> None:
    now = _utcnow()
    values = {
        "session_id":  session_id,
        "snapshot":    metrics_dict,
        "snapshot_at": now,
        "updated_at":  now,
    }
    # Same __table__ pattern as upsert_session_summary — see comment there.
    stmt = (
        pg_insert(EvalMetrics.__table__)
        .values(**values)
        .on_conflict_do_update(index_elements=["session_id"], set_=values)
    )
    async with get_session_factory()() as db:
        # Same FK-safety pattern as upsert_session_summary — ensure the parent
        # sessions row exists before writing eval_metrics.
        await _ensure_session_row(session_id, db, user_id=user_id)
        await db.execute(stmt)
        await db.commit()


async def upsert_consent(
    session_id: str,
    speaker_label: str,
    consented: bool,
    real_name: Optional[str] = None,
) -> None:
    """
    session_id is now required (migration 0009) — consent is scoped per
    session, since speaker_label alone ("Person A") is not a stable identity
    across unrelated sessions.
    """
    values = {
        "session_id":    session_id,
        "speaker_label": speaker_label,
        "consented":     consented,
        "real_name":     real_name,
        "updated_at":    _utcnow(),
    }
    # Same __table__ pattern as upsert_session_summary — see comment there.
    stmt = (
        pg_insert(ConsentRegistry.__table__)
        .values(**values)
        .on_conflict_do_update(index_elements=["session_id", "speaker_label"], set_=values)
    )
    async with get_session_factory()() as db:
        await db.execute(stmt)
        await db.commit()


async def upload_audio_segment(
    session_id: str,
    segment_index: int,
    pcm_float32,  # np.ndarray
) -> Optional[str]:
    import numpy as np

    raw_bytes = pcm_float32.astype(np.float32).tobytes()
    path = f"{session_id}/{segment_index:05d}.f32"

    _fire_sync(
        _safe_storage_upload,
        "audio-segments",
        path,
        raw_bytes,
        "application/octet-stream",
    )

    client = _get_storage_client()
    if client is None:
        return None
    try:
        return client.storage.from_("audio-segments").get_public_url(path)
    except Exception:
        return None


async def upload_recording(session_id: str, local_path: str) -> Optional[str]:
    """
    Read a finished LiveKit Egress recording off the shared /recordings
    volume (see livekit_rooms.py's RECORDINGS_MOUNT) and push it to the
    'recordings' Supabase Storage bucket via the JWT-authenticated Storage
    API — NOT the S3 protocol. Self-hosted Supabase Storage's S3-compatible
    endpoint has a longstanding, unresolved SignatureDoesNotMatch bug
    (https://github.com/supabase/storage/issues/572) that affects every S3
    client regardless of config, so egress now writes locally and this is
    the upload step instead.

    NOTE: storage-api's FILE_SIZE_LIMIT env var (set in
    supabase-project/docker-compose.yml) caps a single upload — default
    52428800 bytes (50MB). Unlike the S3 protocol's multipart upload, this
    single-request JWT upload does NOT chunk large files, so a recording
    longer than fits under that limit will fail outright. Raise
    FILE_SIZE_LIMIT there if meetings routinely exceed 50MB.

    Unlike upload_audio_segment/export_report (fire-and-forget via
    _fire_sync), this awaits the upload and returns its result — the caller
    (stop_egress) logs success/failure per recording, so silently firing
    and forgetting isn't appropriate here.
    """
    if not os.path.exists(local_path):
        log.error(f"[supabase] recording file not found at {local_path}")
        return None

    def _read_and_upload() -> Optional[str]:
        with open(local_path, "rb") as f:
            data = f.read()
        # Derived from the actual file extension rather than hardcoded —
        # this used to be a hardcoded "video/mp4" left over from when
        # egress recorded audio+video. It silently went stale once
        # livekit_rooms.py's start_egress() switched to audio_only MP3
        # output: the upload itself still succeeded, but Storage served
        # every recording back tagged as video/mp4, which is wrong for an
        # MP3 file and breaks browser playback/preview. guess_type() keeps
        # this correct automatically if the output format ever changes
        # again (e.g. back to OGG) without needing another manual edit here.
        content_type, _ = mimetypes.guess_type(local_path)
        content_type = content_type or "application/octet-stream"
        return _safe_storage_upload(
            "recordings",
            f"{session_id}/{os.path.basename(local_path)}",
            data,
            content_type,
        )

    loop = asyncio.get_event_loop()
    url = await loop.run_in_executor(None, _read_and_upload)

    # Best-effort local cleanup — the shared volume is just a handoff point
    # between egress and this upload step, not meant as permanent storage.
    try:
        os.remove(local_path)
    except OSError as exc:
        log.warning(f"[supabase] could not remove local recording {local_path}: {exc}")

    return url


async def export_report(
    session_id: str,
    summary_md: str,
    tags: dict,
    transcript_rows: list[dict],
) -> Optional[str]:
    lines = [
        f"# Lab Brain Session Report — {session_id}\n",
        "## Summary\n",
        summary_md,
        "\n## Action Items\n",
        *[f"- {a}" for a in tags.get("action_items", [])],
        "\n## Decisions\n",
        *[f"- {d}" for d in tags.get("decisions", [])],
        "\n## Transcript\n",
        *[
            f"**[{r.get('timestamp_iso', '')[:19]}] {r.get('speaker', '?')}:** {r.get('text', '')}\n"
            for r in transcript_rows
        ],
    ]
    report_md = "\n".join(lines)
    path = f"{session_id}/report.md"

    _fire_sync(
        _safe_storage_upload,
        "report-exports",
        path,
        report_md.encode("utf-8"),
        "text/markdown",
    )

    client = _get_storage_client()
    if client is None:
        return None
    try:
        return client.storage.from_("report-exports").get_public_url(path)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public read API
# ═══════════════════════════════════════════════════════════════════════════════

async def get_session_owner(session_id: str) -> Optional[str]:
    """
    Return the user_id who owns *session_id* (the account that created it —
    see Session.user_id, NOT NULL as of migration 0007), or None if the
    session row doesn't exist yet.

    Used by the LKC pipeline to scope cross-session retrieval to "this
    session owner's past meetings" (see lkc_retrieval.LKCRetriever's
    user-scoped index and session_pipeline.py's livekit_pipeline) rather
    than either a single session or every session in the table.
    """
    stmt = select(SessionModel.user_id).where(SessionModel.session_id == session_id)
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


async def get_sessions(user_id: str, limit: int = 50) -> list[dict]:
    """
    Return sessions the given user owns OR has participated in — replacing
    the previous "every session in the database, to anyone" behaviour.
    """
    accessible_ids = (
        select(SessionParticipant.session_id)
        .where(SessionParticipant.user_id == user_id)
        .scalar_subquery()
    )
    stmt = (
        select(SessionModel)
        .where(
            or_(
                SessionModel.user_id == user_id,
                SessionModel.session_id.in_(accessible_ids),
            )
        )
        .order_by(SessionModel.started_at.desc())
        .limit(limit)
    )
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        rows = result.scalars().all()
    return [
        {
            "session_id":       r.session_id,
            "host_identity":    r.host_identity,
            "started_at":       r.started_at.isoformat() if r.started_at else None,
            "ended_at":         r.ended_at.isoformat() if r.ended_at else None,
            "livekit_room_sid": r.livekit_room_sid,
            "metadata":         r.metadata_,
        }
        for r in rows
    ]


async def get_transcripts(session_id: str, limit: int = 2000) -> list[dict]:
    stmt = (
        select(Transcript)
        .where(Transcript.session_id == session_id)
        .order_by(Transcript.timestamp_unix)
        .limit(limit)
    )
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        rows = result.scalars().all()
    return [
        {
            "session_id":      r.session_id,
            "segment_index":   r.segment_index,
            "speaker":         r.speaker,
            "text":            r.text,
            "language":        r.language,
            "mode":            r.mode,
            "timestamp_iso":   r.timestamp_iso,
            "timestamp_unix":  r.timestamp_unix,
            "tags":            r.tags,
            "word_timestamps": r.word_timestamps,
            "asr_latency_ms":  r.asr_latency_ms,
            "e2e_latency_ms":  r.e2e_latency_ms,
        }
        for r in rows
    ]


async def get_session_summary(session_id: str) -> Optional[dict]:
    stmt = select(SessionSummary).where(SessionSummary.session_id == session_id)
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
    if row is None:
        return None
    return {
        "session_id": row.session_id,
        "summary_md": row.summary_md,
        "tags":       row.tags,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def get_report_url(session_id: str) -> Optional[str]:
    client = _get_storage_client()
    if client is None:
        return None
    try:
        return client.storage.from_("report-exports").get_public_url(
            f"{session_id}/report.md"
        )
    except Exception:
        return None


def get_audio_segment_url(session_id: str, segment_index: int) -> Optional[str]:
    client = _get_storage_client()
    if client is None:
        return None
    try:
        return client.storage.from_("audio-segments").get_public_url(
            f"{session_id}/{segment_index:05d}.f32"
        )
    except Exception:
        return None


async def connectivity_status() -> dict:
    db_url = os.environ.get("SUPABASE_DB_URL", "")
    reachable = False

    if db_url:
        try:
            async with get_session_factory()() as db:
                await db.execute(select(func.now()))
            reachable = True
        except Exception as exc:
            log.debug(f"[supabase] DB ping failed: {exc}")

    client = _get_storage_client()

    return {
        "db_url_configured":     bool(db_url),
        "storage_url_configured": bool(os.environ.get("SUPABASE_URL")),
        "storage_key_configured": bool(os.environ.get("SUPABASE_SERVICE_KEY")),
        "db_reachable":          reachable,
        "storage_available":     client is not None,
    }