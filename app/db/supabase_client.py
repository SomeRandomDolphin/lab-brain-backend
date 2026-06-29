"""
app/db/supabase_client.py — Supabase persistence layer (runtime data access).

What goes where
---------------
Postgres tables (via supabase-py):
  sessions          — one row per LiveKit room / Lab Brain session
  transcripts       — one row per WhisperX segment (text + tags + word timestamps)
  agent_replies     — one row per LLM reply
  vision_frames     — one row per analysed camera frame
  session_summaries — one row per end-of-session LLM summary
  eval_metrics      — one row per session metric snapshot
  consent_registry  — speaker consent records (mirrors consent.json)

Supabase Storage buckets:
  audio-segments    — raw float32 PCM blobs (one object per WhisperX segment)
  report-exports    — generated markdown summary exports

Schema migrations
------------------
No longer handled here. Schema migrations now run through Alembic —
see app/db/migrations.py and alembic/versions/. This module is purely
the runtime read/write data-access layer.

Dual-write strategy
-------------------
Every write goes to BOTH Supabase (primary) and the local SQLite lkc_graph
(fallback / offline cache). Supabase errors are logged as warnings, never raised.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
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

# ── Client singleton ──────────────────────────────────────────────────────────

def get_client():
    """Return (or lazily create) the Supabase client using env vars."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_AVAILABLE:
        return None

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        log.warning(
            "[supabase] SUPABASE_URL or SUPABASE_KEY not set — "
            "Supabase sync disabled."
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
    client = get_client()
    if client is None:
        return
    try:
        client.table(table).upsert(row, on_conflict=on_conflict).execute()
    except Exception as exc:
        log.warning(f"[supabase] upsert {table} failed: {exc}")


def _safe_insert(table: str, row: dict) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table(table).insert(row).execute()
    except Exception as exc:
        log.warning(f"[supabase] insert {table} failed: {exc}")


def _safe_storage_upload(
    bucket: str, path: str, data: bytes, content_type: str
) -> Optional[str]:
    client = get_client()
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


def _fire(fn, *args) -> None:
    """
    Schedule a sync function in the thread-pool so Supabase HTTP calls
    never block the async pipeline.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, fn, *args)
    except RuntimeError:
        fn(*args)


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
    row = {
        "session_id":      session_id,
        "segment_index":   segment_index,
        "speaker":         speaker,
        "text":            text,
        "language":        language,
        "mode":            mode,
        "timestamp_iso":   timestamp_iso,
        "timestamp_unix":  round(timestamp_unix, 3),
        "tags":            tags,
        "word_timestamps": word_timestamps,
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
    row = {
        "session_id":    session_id,
        "text":          text,
        "mode":          mode,
        "timestamp_iso": datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z",
        "timestamp_unix": round(timestamp_unix, 3),
        "grounded":      grounded,
        "lkc_context":   lkc_context,
        "created_at":    _now_iso(),
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
    row = {
        "session_id":        session_id,
        "timestamp_iso":     datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z",
        "timestamp_unix":    round(timestamp_unix, 3),
        "scene_summary":     scene_summary,
        "present_speakers":  present_speakers,
        "engagement_cues":   engagement_cues,
        "environment_state": environment_state,
        "latency_ms":        latency_ms,
        "created_at":        _now_iso(),
    }
    _fire(_safe_insert, "vision_frames", row)


def upsert_session_summary(session_id: str, summary_md: str, tags: dict) -> None:
    row = {
        "session_id": session_id,
        "summary_md": summary_md,
        "tags":       tags,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _fire(_safe_upsert, "session_summaries", row, "session_id")
    md_bytes     = summary_md.encode("utf-8")
    storage_path = f"{session_id}/summary.md"
    _fire(_safe_storage_upload, "report-exports", storage_path, md_bytes, "text/markdown")


def upsert_eval_metrics(session_id: str, metrics_dict: dict) -> None:
    row = {
        "session_id":  session_id,
        "snapshot":    metrics_dict,
        "snapshot_at": _now_iso(),
        "updated_at":  _now_iso(),
    }
    _fire(_safe_upsert, "eval_metrics", row, "session_id")


def upsert_consent(
    speaker_label: str,
    consented: bool,
    real_name: Optional[str] = None,
) -> None:
    row = {
        "speaker_label": speaker_label,
        "consented":     consented,
        "real_name":     real_name,
        "updated_at":    _now_iso(),
    }
    _fire(_safe_upsert, "consent_registry", row, "speaker_label")


def upload_audio_segment(
    session_id: str,
    segment_index: int,
    pcm_float32,  # np.ndarray
) -> Optional[str]:
    client = get_client()
    if client is None:
        return None

    import numpy as np
    raw_bytes = pcm_float32.astype(np.float32).tobytes()
    path      = f"{session_id}/{segment_index:05d}.f32"

    def _upload():
        return _safe_storage_upload("audio-segments", path, raw_bytes, "application/octet-stream")

    _fire(_upload)
    try:
        return get_client().storage.from_("audio-segments").get_public_url(path)
    except Exception:
        return None


def export_report(
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
            f"**[{r.get('timestamp_iso','')[:19]}] {r.get('speaker','?')}:** {r.get('text','')}\n"
            for r in transcript_rows
        ],
    ]
    report_md = "\n".join(lines)
    path      = f"{session_id}/report.md"

    def _up():
        return _safe_storage_upload(
            "report-exports", path,
            report_md.encode("utf-8"), "text/markdown",
        )

    _fire(_up)
    try:
        return get_client().storage.from_("report-exports").get_public_url(path)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public read API
# ═══════════════════════════════════════════════════════════════════════════════

def get_sessions(limit: int = 50) -> list[dict]:
    client = get_client()
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
    client = get_client()
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
    client = get_client()
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
    client = get_client()
    if client is None:
        return None
    try:
        return client.storage.from_("report-exports").get_public_url(
            f"{session_id}/report.md"
        )
    except Exception:
        return None


def get_audio_segment_url(session_id: str, segment_index: int) -> Optional[str]:
    client = get_client()
    if client is None:
        return None
    try:
        return client.storage.from_("audio-segments").get_public_url(
            f"{session_id}/{segment_index:05d}.f32"
        )
    except Exception:
        return None


def connectivity_status() -> dict:
    """Return connectivity and configuration status."""
    url       = os.environ.get("SUPABASE_URL", "")
    key_set   = bool(os.environ.get("SUPABASE_KEY", ""))
    client    = get_client()
    reachable = False

    if client is not None:
        try:
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