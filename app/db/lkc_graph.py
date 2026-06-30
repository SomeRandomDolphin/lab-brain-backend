"""
app/db/lkc_graph.py — Supabase-backed LKC event graph (SQLAlchemy async ORM).

Replaces the old SQLite + JSONL implementation. All records are persisted
to the `lkc_records` Postgres table in your Supabase project via the shared
async engine in supabase_client.py.

The public API surface (write_to_lkc, read_lkc, read_sessions, …) is
unchanged so existing callers need no modification.

Schema: see models.LkcRecord
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Integer, func, select, delete

from .models import LkcRecord
from .supabase_client import get_session_factory

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _from_unix(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Write ──────────────────────────────────────────────────────────────────────

async def write_to_lkc(record: dict) -> None:
    """Persist a record to the lkc_records table in Supabase."""
    now = time.time()
    record.setdefault("timestamp_unix", now)
    record.setdefault("timestamp_iso", _from_unix(record["timestamp_unix"]))

    row = LkcRecord(
        session_id     = record.get("session_id", "unknown"),
        record_type    = record.get("type", "unknown"),
        timestamp_unix = record["timestamp_unix"],
        timestamp_iso  = record["timestamp_iso"],
        speaker        = record.get("speaker"),
        text           = record.get("text") or record.get("summary"),
        mode           = record.get("mode"),
        language       = record.get("language"),
        payload        = record,
    )

    async with get_session_factory()() as db:
        db.add(row)
        await db.commit()
        log.debug(f"[lkc_graph] wrote {row.record_type} for session {row.session_id}")


# ── Read ───────────────────────────────────────────────────────────────────────

async def read_lkc(
    session_id: Optional[str] = None,
    record_type: Optional[str] = None,
    since_unix: Optional[float] = None,
    limit: int = 2000,
) -> list[dict]:
    stmt = select(LkcRecord).order_by(LkcRecord.timestamp_unix.asc()).limit(limit)

    if session_id:
        stmt = stmt.where(LkcRecord.session_id == session_id)
    if record_type:
        stmt = stmt.where(LkcRecord.record_type == record_type)
    if since_unix is not None:
        stmt = stmt.where(LkcRecord.timestamp_unix > since_unix)

    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        rows = result.scalars().all()

    return [r.payload for r in rows]


async def read_sessions() -> list[dict]:
    stmt = (
        select(
            LkcRecord.session_id,
            func.min(LkcRecord.timestamp_unix).label("started"),
            func.max(LkcRecord.timestamp_unix).label("ended"),
            func.count().label("total"),
            func.sum(
                (LkcRecord.record_type == "transcript").cast(Integer)
            ).label("transcripts"),
            func.sum(
                (LkcRecord.record_type == "vision").cast(Integer)
            ).label("vision_frames"),
            func.sum(
                (LkcRecord.record_type == "agent_reply").cast(Integer)
            ).label("agent_replies"),
            func.sum(
                (LkcRecord.record_type == "session_summary").cast(Integer)
            ).label("summaries"),
        )
        .group_by(LkcRecord.session_id)
        .order_by(func.min(LkcRecord.timestamp_unix).desc())
    )

    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        rows = result.all()

    return [
        {
            "session_id":    r.session_id,
            "started_iso":   _from_unix(r.started),
            "ended_iso":     _from_unix(r.ended),
            "total_records": r.total,
            "transcripts":   r.transcripts or 0,
            "vision_frames": r.vision_frames or 0,
            "agent_replies": r.agent_replies or 0,
            "summaries":     r.summaries or 0,
        }
        for r in rows
    ]


async def session_text_corpus(session_id: str) -> list[dict]:
    stmt = (
        select(LkcRecord.timestamp_iso, LkcRecord.speaker, LkcRecord.text)
        .where(LkcRecord.session_id == session_id)
        .where(LkcRecord.record_type == "transcript")
        .where(LkcRecord.text.isnot(None))
        .where(LkcRecord.text != "")
        .order_by(LkcRecord.timestamp_unix.asc())
    )

    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        rows = result.all()

    return [
        {"timestamp_iso": r.timestamp_iso, "speaker": r.speaker, "text": r.text}
        for r in rows
    ]


async def clear_session(session_id: str) -> int:
    stmt = delete(LkcRecord).where(LkcRecord.session_id == session_id)
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        await db.commit()
    log.info(f"[lkc_graph] cleared {result.rowcount} records for session {session_id}")
    return result.rowcount


async def clear_all() -> int:
    stmt = delete(LkcRecord)
    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        await db.commit()
    log.warning(f"[lkc_graph] cleared ALL {result.rowcount} lkc records")
    return result.rowcount


async def graph_stats() -> dict:
    stmt = select(
        func.count().label("total"),
        func.count(func.distinct(LkcRecord.session_id)).label("sessions"),
        func.sum(
            (LkcRecord.record_type == "transcript").cast(Integer)
        ).label("transcripts"),
        func.sum(
            (LkcRecord.record_type == "vision").cast(Integer)
        ).label("vision"),
        func.sum(
            (LkcRecord.record_type == "agent_reply").cast(Integer)
        ).label("agent_replies"),
        func.sum(
            (LkcRecord.record_type == "session_summary").cast(Integer)
        ).label("summaries"),
        func.min(LkcRecord.timestamp_unix).label("oldest"),
        func.max(LkcRecord.timestamp_unix).label("newest"),
    )

    async with get_session_factory()() as db:
        result = await db.execute(stmt)
        row = result.one()

    return {
        "total_records": row.total or 0,
        "sessions":      row.sessions or 0,
        "transcripts":   row.transcripts or 0,
        "vision_frames": row.vision or 0,
        "agent_replies": row.agent_replies or 0,
        "summaries":     row.summaries or 0,
        "oldest_iso":    _from_unix(row.oldest) if row.oldest else None,
        "newest_iso":    _from_unix(row.newest) if row.newest else None,
    }