"""
app/db/supabase_client.py — Supabase persistence layer (SQLAlchemy async ORM).

All runtime data access goes through SQLAlchemy with asyncpg, talking directly
to the Postgres instance that backs your Supabase project. The supabase-py
client is retained only for Storage bucket operations (audio segments and
report exports), which have no SQLAlchemy equivalent.

What goes where
---------------
Postgres tables (SQLAlchemy ORM — see models.py):
  sessions          — one row per LiveKit room / Lab Brain session
  transcripts       — one row per WhisperX segment (text + tags + word timestamps)
  agent_replies     — one row per LLM reply
  vision_frames     — one row per analysed camera frame
  session_summaries — one row per end-of-session LLM summary
  eval_metrics      — one row per session metric snapshot
  consent_registry  — speaker consent records

Supabase Storage buckets (supabase-py):
  audio-segments    — raw float32 PCM blobs
  report-exports    — generated markdown summary exports

Required environment variables
-------------------------------
  SUPABASE_DB_URL   postgresql+asyncpg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres
  SUPABASE_URL      https://<project>.supabase.co          (Storage only)
  SUPABASE_KEY      service_role or anon key               (Storage only)

Schema migrations
-----------------
Handled by Alembic — see app/db/migrations.py. This module is pure runtime
read/write only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, delete, text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import (
    AgentReply,
    ConsentRegistry,
    EvalMetrics,
    Session as SessionModel,
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


async def _ensure_session_row(session_id: str, db: AsyncSession) -> None:
    """
    Guarantee a sessions row exists for *session_id* without overwriting any
    real session data that may already be there.

    Called by upsert_session_summary and upsert_eval_metrics, which can be
    invoked by the frontend before POST /sessions has been called (the client
    generates IDs locally and may send summary/metrics for a session that was
    never explicitly persisted).  A bare INSERT … ON CONFLICT DO NOTHING is the
    lightest-weight way to satisfy the FK constraint.

    We use raw SQL rather than pg_insert(SessionModel.__table__) to sidestep
    the SQLAlchemy metadata/metadata_ column naming quirk entirely.
    """
    await db.execute(
        sa_text("""
            INSERT INTO sessions (session_id, host_identity, started_at, metadata, updated_at)
            VALUES (:sid, 'browser-user', NOW(), '{}', NOW())
            ON CONFLICT (session_id) DO NOTHING
        """),
        {"sid": session_id},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public write API
# ═══════════════════════════════════════════════════════════════════════════════

async def upsert_session(
    session_id: str,
    host_identity: str = "browser-user",
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    livekit_room_sid: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
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

    stmt = (
        pg_insert(SessionModel)
        .values(**values)
        .on_conflict_do_update(index_elements=["session_id"], set_=values)
    )
    async with get_session_factory()() as db:
        await db.execute(stmt)
        await db.commit()


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


async def upsert_session_summary(session_id: str, summary_md: str, tags: dict) -> None:
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
        # summary.  The client generates session IDs locally and may call this
        # endpoint before the session has been persisted — ON CONFLICT DO NOTHING
        # is a no-op when the row is already there.
        await _ensure_session_row(session_id, db)
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


async def upsert_eval_metrics(session_id: str, metrics_dict: dict) -> None:
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
        await _ensure_session_row(session_id, db)
        await db.execute(stmt)
        await db.commit()


async def upsert_consent(
    speaker_label: str,
    consented: bool,
    real_name: Optional[str] = None,
) -> None:
    values = {
        "speaker_label": speaker_label,
        "consented":     consented,
        "real_name":     real_name,
        "updated_at":    _utcnow(),
    }
    # Same __table__ pattern as upsert_session_summary — see comment there.
    stmt = (
        pg_insert(ConsentRegistry.__table__)
        .values(**values)
        .on_conflict_do_update(index_elements=["speaker_label"], set_=values)
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

async def get_sessions(limit: int = 50) -> list[dict]:
    stmt = select(SessionModel).order_by(SessionModel.started_at.desc()).limit(limit)
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
        "storage_key_configured": bool(os.environ.get("SUPABASE_KEY")),
        "db_reachable":          reachable,
        "storage_available":     client is not None,
    }