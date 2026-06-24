"""
supabase_store.py — Supabase persistence layer (Month 7)

Replaces / supplements the local SQLite lkc_graph.db with Supabase (Postgres +
Storage) so all data survives beyond a single machine, is queryable from the
React frontend, and can be shared across multiple backend instances.

What goes where
---------------
Postgres tables (via supabase-py)
  sessions          — one row per LiveKit room / Lab Brain session
  transcripts       — one row per WhisperX segment (text + tags + word timestamps)
  agent_replies     — one row per LLM reply
  vision_frames     — one row per analysed camera frame
  session_summaries — one row per end-of-session LLM summary
  eval_metrics      — one row per session metric snapshot
  consent_registry  — speaker consent records (mirrors consent.json)

Supabase Storage buckets
  audio-segments    — raw float32 PCM blobs (one object per WhisperX segment)
  report-exports    — generated PDF / markdown summary exports

Dual-write strategy
-------------------
Every write goes to BOTH Supabase (primary) and the local SQLite lkc_graph
(fallback / offline cache).  If the Supabase client is unavailable or the
network is down, writes succeed locally and the server keeps running — Supabase
errors are logged as warnings, never raised.

This means:
  - Zero behaviour change when SUPABASE_URL / SUPABASE_KEY are not set.
  - Full durability when they are set.
  - No blocking: Supabase writes are fire-and-forget via asyncio.create_task.

SQL schema (run once in Supabase SQL editor)
--------------------------------------------
See the docstring at the bottom of this file, or run:
    python supabase_store.py --create-schema
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Supabase client (graceful degradation) ────────────────────────────────────
SUPABASE_AVAILABLE = False
_client = None

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    log.warning(
        "[supabase] supabase-py not installed — Supabase sync disabled. "
        "Run: pip install supabase"
    )


def _get_client():
    """Return (or lazily create) the Supabase client using env vars."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_AVAILABLE:
        return None

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")   # service-role key (not anon)

    if not url or not key:
        log.warning(
            "[supabase] SUPABASE_URL or SUPABASE_KEY not set — "
            "Supabase sync disabled. Set both env vars to enable."
        )
        return None

    try:
        _client = create_client(url, key)
        log.info(f"[supabase] connected to {url}")
        return _client
    except Exception as exc:
        log.warning(f"[supabase] client init failed: {exc}")
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_upsert(table: str, row: dict, on_conflict: str = "id") -> None:
    """
    Upsert a row into a Supabase table.  Errors are logged, never raised.
    Called from _fire_and_forget so this runs in a thread-pool executor.
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.table(table).upsert(row, on_conflict=on_conflict).execute()
    except Exception as exc:
        log.warning(f"[supabase] upsert {table} failed: {exc}")


def _safe_insert(table: str, row: dict) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table(table).insert(row).execute()
    except Exception as exc:
        log.warning(f"[supabase] insert {table} failed: {exc}")


def _safe_storage_upload(bucket: str, path: str, data: bytes, content_type: str) -> Optional[str]:
    """
    Upload bytes to a Supabase Storage bucket.
    Returns the public URL on success, None on failure.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        client.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        url = client.storage.from_(bucket).get_public_url(path)
        return url
    except Exception as exc:
        log.warning(f"[supabase] storage upload {bucket}/{path} failed: {exc}")
        return None


def _fire(coro_or_fn, *args) -> None:
    """
    Schedule a sync function to run in the event-loop's default executor
    (thread pool) so Supabase HTTP calls never block the async pipeline.
    Falls back to direct call if no event loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, coro_or_fn, *args)
    except RuntimeError:
        # No event loop (e.g. called from a test or startup hook)
        coro_or_fn(*args)


# ═══════════════════════════════════════════════════════════════════════════════
# Public write API
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_session(
    session_id: str,
    host_identity: str = "browser-user",
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    livekit_room_sid: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Create or update the sessions row for a LiveKit room.
    Called by server.py on POST /livekit/room (start) and DELETE /livekit/room (end).
    """
    row = {
        "session_id":       session_id,
        "host_identity":    host_identity,
        "started_at":       datetime.utcfromtimestamp(started_at or time.time()).isoformat() + "Z",
        "livekit_room_sid": livekit_room_sid,
        "metadata":         json.dumps(metadata or {}),
        "updated_at":       _now_iso(),
    }
    if ended_at is not None:
        row["ended_at"] = datetime.utcfromtimestamp(ended_at).isoformat() + "Z"

    _fire(_safe_upsert, "sessions", row, "session_id")


def insert_transcript(
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
    """
    Persist one WhisperX transcript segment.
    tags and word_timestamps are stored as JSONB.
    """
    row = {
        "session_id":      session_id,
        "segment_index":   segment_index,
        "speaker":         speaker,
        "text":            text,
        "language":        language,
        "mode":            mode,
        "timestamp_iso":   timestamp_iso,
        "timestamp_unix":  round(timestamp_unix, 3),
        "tags":            tags,                  # JSONB — Supabase handles dict natively
        "word_timestamps": word_timestamps,        # JSONB array
        "asr_latency_ms":  asr_latency_ms,
        "e2e_latency_ms":  e2e_latency_ms,
        "created_at":      _now_iso(),
    }
    _fire(_safe_insert, "transcripts", row)


def insert_agent_reply(
    session_id: str,
    text: str,
    mode: str,
    timestamp_unix: float,
    grounded: bool = False,
    lkc_context: str = "",
) -> None:
    """Persist one LLM agent reply."""
    row = {
        "session_id":   session_id,
        "text":         text,
        "mode":         mode,
        "timestamp_iso": datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z",
        "timestamp_unix": round(timestamp_unix, 3),
        "grounded":     grounded,
        "lkc_context":  lkc_context,
        "created_at":   _now_iso(),
    }
    _fire(_safe_insert, "agent_replies", row)


def insert_vision_frame(
    session_id: str,
    timestamp_unix: float,
    scene_summary: str,
    present_speakers: list,
    engagement_cues: dict,
    environment_state: dict,
    latency_ms: int = 0,
) -> None:
    """Persist one vision frame analysis result."""
    row = {
        "session_id":        session_id,
        "timestamp_iso":     datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z",
        "timestamp_unix":    round(timestamp_unix, 3),
        "scene_summary":     scene_summary,
        "present_speakers":  present_speakers,   # JSONB array
        "engagement_cues":   engagement_cues,    # JSONB
        "environment_state": environment_state,  # JSONB
        "latency_ms":        latency_ms,
        "created_at":        _now_iso(),
    }
    _fire(_safe_insert, "vision_frames", row)


def upsert_session_summary(
    session_id: str,
    summary_md: str,
    tags: dict,
) -> None:
    """
    Persist the end-of-session LLM summary.
    Also stores a plain-text export in Supabase Storage as
    report-exports/{session_id}/summary.md for direct download.
    """
    row = {
        "session_id":  session_id,
        "summary_md":  summary_md,
        "tags":        tags,           # JSONB
        "created_at":  _now_iso(),
        "updated_at":  _now_iso(),
    }
    _fire(_safe_upsert, "session_summaries", row, "session_id")

    # Upload the markdown to Storage so the frontend can offer a download link
    md_bytes = summary_md.encode("utf-8")
    storage_path = f"{session_id}/summary.md"
    _fire(_safe_storage_upload, "report-exports", storage_path, md_bytes, "text/markdown")


def upsert_eval_metrics(session_id: str, metrics_dict: dict) -> None:
    """
    Persist a metrics snapshot (called by /metrics endpoint or on session end).
    metrics_dict is the output of SessionMetrics.summary().
    """
    row = {
        "session_id":   session_id,
        "snapshot":     metrics_dict,   # JSONB
        "snapshot_at":  _now_iso(),
        "updated_at":   _now_iso(),
    }
    _fire(_safe_upsert, "eval_metrics", row, "session_id")


def upsert_consent(
    speaker_label: str,
    consented: bool,
    real_name: Optional[str] = None,
) -> None:
    """Mirror privacy.py consent registry to Supabase."""
    row = {
        "speaker_label": speaker_label,
        "consented":     consented,
        "real_name":     real_name,
        "updated_at":    _now_iso(),
    }
    _fire(_safe_upsert, "consent_registry", row, "speaker_label")


# ── Audio segment upload ───────────────────────────────────────────────────────

def upload_audio_segment(
    session_id: str,
    segment_index: int,
    pcm_float32: "np.ndarray",  # noqa: F821
) -> Optional[str]:
    """
    Upload a raw float32 PCM audio segment to Supabase Storage.
    Returns the public URL or None if unavailable.

    Storage path: audio-segments/{session_id}/{segment_index:05d}.f32
    The .f32 extension signals raw float32 little-endian, 16kHz, mono.
    Callers can reconstruct with:
        np.frombuffer(data, dtype=np.float32)
    """
    client = _get_client()
    if client is None:
        return None

    import numpy as np  # local import — numpy is always present

    raw_bytes = pcm_float32.astype(np.float32).tobytes()
    path      = f"{session_id}/{segment_index:05d}.f32"

    def _upload():
        return _safe_storage_upload(
            "audio-segments", path, raw_bytes, "application/octet-stream"
        )

    _fire(_upload)
    # Return predictable URL without waiting for the async result
    try:
        base = _get_client().storage.from_("audio-segments").get_public_url(path)
        return base
    except Exception:
        return None


# ── Report export (markdown + future PDF) ─────────────────────────────────────

def export_report(
    session_id: str,
    summary_md: str,
    tags: dict,
    transcript_rows: list[dict],
) -> Optional[str]:
    """
    Build a full session report (markdown) and upload to
    report-exports/{session_id}/report.md

    The report includes:
      - Session metadata
      - LLM summary
      - Captured tags (action items, decisions, deadlines, entities)
      - Full transcript with speaker labels and timestamps

    Returns the public Storage URL or None.
    """
    lines: list[str] = [
        f"# Lab Brain Session Report — {session_id}",
        f"_Generated: {_now_iso()}_\n",
        "---\n",
        "## Summary\n",
        summary_md,
        "\n---\n",
        "## Captured Tags\n",
    ]

    if tags.get("action_items"):
        lines.append("**Action Items**")
        for a in tags["action_items"]:
            lines.append(f"- {a}")
        lines.append("")

    if tags.get("decisions"):
        lines.append("**Decisions**")
        for d in tags["decisions"]:
            lines.append(f"- {d}")
        lines.append("")

    if tags.get("deadlines"):
        lines.append("**Deadlines**")
        for d in tags["deadlines"]:
            lines.append(f"- {d}")
        lines.append("")

    if tags.get("entities"):
        lines.append(f"**Key Entities:** {', '.join(tags['entities'])}\n")

    lines += ["\n---\n", "## Transcript\n"]
    for row in transcript_rows:
        ts   = (row.get("timestamp_iso") or "")[:19].replace("T", " ")
        sp   = row.get("speaker", "?")
        text = row.get("text", "")
        lines.append(f"**[{ts}] {sp}:** {text}\n")

    report_md  = "\n".join(lines)
    report_bytes = report_md.encode("utf-8")
    path         = f"{session_id}/report.md"

    def _up():
        return _safe_storage_upload("report-exports", path, report_bytes, "text/markdown")

    _fire(_up)
    try:
        return _get_client().storage.from_("report-exports").get_public_url(path)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public read API (used by REST endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

def get_sessions(limit: int = 50) -> list[dict]:
    """Return recent sessions ordered by start time descending."""
    client = _get_client()
    if client is None:
        return []
    try:
        res = (
            client.table("sessions")
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.warning(f"[supabase] get_sessions failed: {exc}")
        return []


def get_transcripts(session_id: str, limit: int = 2000) -> list[dict]:
    """Return all transcript segments for a session ordered by time."""
    client = _get_client()
    if client is None:
        return []
    try:
        res = (
            client.table("transcripts")
            .select("*")
            .eq("session_id", session_id)
            .order("timestamp_unix")
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.warning(f"[supabase] get_transcripts failed: {exc}")
        return []


def get_session_summary(session_id: str) -> Optional[dict]:
    """Return the latest summary for a session."""
    client = _get_client()
    if client is None:
        return None
    try:
        res = (
            client.table("session_summaries")
            .select("*")
            .eq("session_id", session_id)
            .single()
            .execute()
        )
        return res.data
    except Exception as exc:
        log.warning(f"[supabase] get_session_summary failed: {exc}")
        return None


def get_report_url(session_id: str) -> Optional[str]:
    """Return the public Storage URL for the session report, or None."""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.storage.from_("report-exports").get_public_url(
            f"{session_id}/report.md"
        )
    except Exception:
        return None


def get_audio_segment_url(session_id: str, segment_index: int) -> Optional[str]:
    """Return the public Storage URL for a raw audio segment, or None."""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.storage.from_("audio-segments").get_public_url(
            f"{session_id}/{segment_index:05d}.f32"
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Status / diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def status() -> dict:
    """Return connectivity and configuration status for /supabase/status."""
    url      = os.environ.get("SUPABASE_URL", "")
    key_set  = bool(os.environ.get("SUPABASE_KEY", ""))
    client   = _get_client()
    reachable = False

    if client is not None:
        try:
            # Lightweight ping — list at most 1 row from sessions
            client.table("sessions").select("session_id").limit(1).execute()
            reachable = True
        except Exception as exc:
            log.debug(f"[supabase] ping failed: {exc}")

    return {
        "supabase_available": SUPABASE_AVAILABLE,
        "url_configured":     bool(url),
        "key_configured":     key_set,
        "reachable":          reachable,
        "url":                url or None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Schema SQL (for reference / --create-schema CLI)
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Run this once in the Supabase SQL editor (or via psql).
-- Enable pgcrypto for gen_random_uuid() if not already enabled.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sessions ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT        PRIMARY KEY,
    host_identity     TEXT        NOT NULL DEFAULT 'browser-user',
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at          TIMESTAMPTZ,
    livekit_room_sid  TEXT,
    metadata          JSONB       NOT NULL DEFAULT '{}',
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Transcripts ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transcripts (
    id               BIGSERIAL   PRIMARY KEY,
    session_id       TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    segment_index    INT         NOT NULL DEFAULT 0,
    speaker          TEXT        NOT NULL,
    text             TEXT        NOT NULL,
    language         TEXT,
    mode             TEXT,
    timestamp_iso    TEXT,
    timestamp_unix   DOUBLE PRECISION NOT NULL,
    tags             JSONB       NOT NULL DEFAULT '{}',
    word_timestamps  JSONB       NOT NULL DEFAULT '[]',
    asr_latency_ms   INT         NOT NULL DEFAULT 0,
    e2e_latency_ms   INT         NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts (session_id, timestamp_unix);

-- Agent replies ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_replies (
    id              BIGSERIAL   PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    text            TEXT        NOT NULL,
    mode            TEXT,
    timestamp_iso   TEXT,
    timestamp_unix  DOUBLE PRECISION NOT NULL,
    grounded        BOOLEAN     NOT NULL DEFAULT FALSE,
    lkc_context     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_replies_session ON agent_replies (session_id, timestamp_unix);

-- Vision frames ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vision_frames (
    id                BIGSERIAL   PRIMARY KEY,
    session_id        TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    timestamp_iso     TEXT,
    timestamp_unix    DOUBLE PRECISION NOT NULL,
    scene_summary     TEXT,
    present_speakers  JSONB       NOT NULL DEFAULT '[]',
    engagement_cues   JSONB       NOT NULL DEFAULT '{}',
    environment_state JSONB       NOT NULL DEFAULT '{}',
    latency_ms        INT         NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vision_frames_session ON vision_frames (session_id, timestamp_unix);

-- Session summaries ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id   TEXT        PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary_md   TEXT        NOT NULL,
    tags         JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Eval metrics ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_metrics (
    session_id   TEXT        PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    snapshot     JSONB       NOT NULL DEFAULT '{}',
    snapshot_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Consent registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS consent_registry (
    speaker_label  TEXT        PRIMARY KEY,
    consented      BOOLEAN     NOT NULL DEFAULT FALSE,
    real_name      TEXT,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Storage buckets (create via Supabase dashboard or CLI):
--   audio-segments   — raw PCM segments (private or public per your policy)
--   report-exports   — generated markdown / PDF reports (public)
"""


if __name__ == "__main__":
    import sys
    if "--create-schema" in sys.argv:
        print(SCHEMA_SQL)
    elif "--status" in sys.argv:
        import pprint
        pprint.pprint(status())
    else:
        print("Usage: python supabase_store.py [--create-schema | --status]")