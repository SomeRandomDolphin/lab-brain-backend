"""
lkc_graph.py — Persistent LKC Graph (Module 5, Month 5)

Replaces the flat lkc_stream.jsonl append-only store with a SQLite-backed
graph that:

  1. Survives server restarts — records are durable across process exits.
  2. Indexes the full session history, not just the current file.
  3. Supports fast record-type and session-ID filtering without scanning
     the whole file on every /lkc or /summary request.
  4. Exposes the same write_to_lkc() / read_lkc() interface as Month 4
     so server.py and capture.py need only minimal changes.
  5. Keeps backward compatibility by also writing to lkc_stream.jsonl so
     existing tooling (Rifqi / Wildan pipelines) still works.

Schema (one table):

  lkc_records
  ┌────────────────┬────────────────────────────────────────────────────┐
  │ id             │ INTEGER PRIMARY KEY AUTOINCREMENT                  │
  │ session_id     │ TEXT    (indexed)                                  │
  │ record_type    │ TEXT    (transcript | vision | agent_reply |       │
  │                │          session_summary)                           │
  │ timestamp_unix │ REAL    (indexed)                                  │
  │ timestamp_iso  │ TEXT                                               │
  │ speaker        │ TEXT    (nullable)                                 │
  │ text           │ TEXT    (nullable — main payload for retrieval)    │
  │ mode           │ TEXT    (nullable)                                 │
  │ language       │ TEXT    (nullable)                                 │
  │ payload        │ TEXT    (full JSON blob — everything else)         │
  └────────────────┴────────────────────────────────────────────────────┘

Thread safety: SQLite WAL mode + a module-level threading.Lock guard all
write operations so the FastAPI thread-pool can call write_to_lkc() from
multiple async executors without corruption.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

_DB_PATH   = Path("lkc_graph.db")
_JSONL_PATH = Path("lkc_stream.jsonl")  # kept for backward compat

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS lkc_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    record_type    TEXT    NOT NULL,
    timestamp_unix REAL    NOT NULL,
    timestamp_iso  TEXT    NOT NULL,
    speaker        TEXT,
    text           TEXT,
    mode           TEXT,
    language       TEXT,
    payload        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session    ON lkc_records (session_id);
CREATE INDEX IF NOT EXISTS idx_type       ON lkc_records (record_type);
CREATE INDEX IF NOT EXISTS idx_ts         ON lkc_records (timestamp_unix);
CREATE INDEX IF NOT EXISTS idx_session_ts ON lkc_records (session_id, timestamp_unix);
"""


def _get_conn() -> sqlite3.Connection:
    """Return (or initialise) the module-level SQLite connection."""
    global _conn
    if _conn is not None:
        return _conn
    db_path = _DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_CREATE_SQL)
    conn.commit()
    _conn = conn
    log.info(f"[lkc_graph] SQLite graph opened at {db_path.resolve()}")
    return _conn


def configure(db_path: Path = _DB_PATH, jsonl_path: Path = _JSONL_PATH) -> None:
    """
    Override default paths (call before first write, e.g. from server.py).
    Useful for tests or when config.json specifies a different location.
    """
    global _DB_PATH, _JSONL_PATH, _conn
    _DB_PATH    = db_path
    _JSONL_PATH = jsonl_path
    _conn = None   # force reconnect on next use


# ── Write ─────────────────────────────────────────────────────────────────────

def write_to_lkc(record: dict) -> None:
    """
    Persist a record to both SQLite (primary) and lkc_stream.jsonl (compat).

    Accepts any dict that follows the LKC record schema:
      {type, session_id, timestamp_unix?, timestamp_iso?, speaker?, text?, ...}
    Missing timestamps are filled automatically.
    """
    now = time.time()
    record.setdefault("timestamp_unix", now)
    record.setdefault(
        "timestamp_iso",
        datetime.utcfromtimestamp(record["timestamp_unix"]).isoformat() + "Z",
    )

    session_id    = record.get("session_id", "unknown")
    record_type   = record.get("type", "unknown")
    timestamp_unix = record["timestamp_unix"]
    timestamp_iso  = record["timestamp_iso"]
    speaker        = record.get("speaker")
    text           = record.get("text") or record.get("summary")
    mode           = record.get("mode")
    language       = record.get("language")
    payload        = json.dumps(record, ensure_ascii=False)

    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO lkc_records
              (session_id, record_type, timestamp_unix, timestamp_iso,
               speaker, text, mode, language, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, record_type, timestamp_unix, timestamp_iso,
             speaker, text, mode, language, payload),
        )
        conn.commit()

    # Backward-compat JSONL mirror
    try:
        with _JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")
    except Exception as exc:
        log.warning(f"[lkc_graph] JSONL mirror write failed: {exc}")


# ── Read ──────────────────────────────────────────────────────────────────────

def read_lkc(
    session_id: Optional[str] = None,
    record_type: Optional[str] = None,
    since_unix: Optional[float] = None,
    limit: int = 2000,
) -> list[dict]:
    """
    Query LKC records with optional filters.

    Parameters
    ----------
    session_id  : restrict to one session (None = all sessions)
    record_type : one of transcript | vision | agent_reply | session_summary
    since_unix  : return only records newer than this UNIX timestamp
    limit       : hard cap on rows returned (most-recent first)

    Returns a list of record dicts, ordered oldest → newest.
    """
    clauses: list[str] = []
    params:  list      = []

    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if record_type:
        clauses.append("record_type = ?")
        params.append(record_type)
    if since_unix is not None:
        clauses.append("timestamp_unix > ?")
        params.append(since_unix)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT payload FROM lkc_records
        {where}
        ORDER BY timestamp_unix ASC
        LIMIT ?
    """
    params.append(limit)

    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    records = []
    for (payload,) in rows:
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return records


def read_sessions() -> list[dict]:
    """
    Return a summary of all unique session_ids in the graph with start
    time, end time, and record counts per type.
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            session_id,
            MIN(timestamp_unix)  AS started,
            MAX(timestamp_unix)  AS ended,
            COUNT(*)             AS total,
            SUM(record_type = 'transcript')      AS transcripts,
            SUM(record_type = 'vision')          AS vision_frames,
            SUM(record_type = 'agent_reply')     AS agent_replies,
            SUM(record_type = 'session_summary') AS summaries
        FROM lkc_records
        GROUP BY session_id
        ORDER BY started DESC
    """).fetchall()

    return [
        {
            "session_id":    r[0],
            "started_iso":   datetime.utcfromtimestamp(r[1]).isoformat() + "Z",
            "ended_iso":     datetime.utcfromtimestamp(r[2]).isoformat() + "Z",
            "total_records": r[3],
            "transcripts":   r[4],
            "vision_frames": r[5],
            "agent_replies": r[6],
            "summaries":     r[7],
        }
        for r in rows
    ]


def session_text_corpus(session_id: str) -> list[dict]:
    """
    Return all transcript records for a session as lightweight dicts
    suitable for dense-embedding retrieval indexing.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT timestamp_iso, speaker, text
        FROM lkc_records
        WHERE session_id = ? AND record_type = 'transcript'
          AND text IS NOT NULL AND text != ''
        ORDER BY timestamp_unix ASC
        """,
        (session_id,),
    ).fetchall()
    return [{"timestamp_iso": r[0], "speaker": r[1], "text": r[2]} for r in rows]


def clear_session(session_id: str) -> int:
    """Delete all records for a session. Returns row count deleted."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM lkc_records WHERE session_id = ?", (session_id,)
        )
        conn.commit()
    return cur.rowcount


def clear_all() -> int:
    """Wipe the entire graph. Returns row count deleted."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM lkc_records")
        conn.commit()
    # Also truncate the JSONL mirror
    try:
        _JSONL_PATH.write_text("")
    except Exception:
        pass
    return cur.rowcount


def graph_stats() -> dict:
    """Return aggregate graph statistics for the /lkc/stats endpoint."""
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*)                              AS total,
            COUNT(DISTINCT session_id)            AS sessions,
            SUM(record_type='transcript')         AS transcripts,
            SUM(record_type='vision')             AS vision,
            SUM(record_type='agent_reply')        AS agent_replies,
            SUM(record_type='session_summary')    AS summaries,
            MIN(timestamp_unix)                   AS oldest,
            MAX(timestamp_unix)                   AS newest
        FROM lkc_records
    """).fetchone()

    return {
        "total_records": row[0],
        "sessions":      row[1],
        "transcripts":   row[2],
        "vision_frames": row[3],
        "agent_replies": row[4],
        "summaries":     row[5],
        "oldest_iso": (
            datetime.utcfromtimestamp(row[6]).isoformat() + "Z" if row[6] else None
        ),
        "newest_iso": (
            datetime.utcfromtimestamp(row[7]).isoformat() + "Z" if row[7] else None
        ),
        "db_path": str(_DB_PATH.resolve()),
    }
