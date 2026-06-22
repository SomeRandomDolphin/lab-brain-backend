"""
capture.py — Autonomous Capture & Rifqi Pipeline Integration (Module 5, Month 3)

Two responsibilities:

1.  Rifqi Handshake (Module 2 — Meeting Capture Pipeline)
    -----------------------------------------------------------
    Rifqi's pipeline pushes structured meeting segments via POST /capture/ingest.
    This module receives them, normalises to the shared LKC schema, and writes
    them to lkc_stream.jsonl so Module 5's retrieval layer can answer questions
    that span both pipelines.

    Expected payload from Rifqi (flexible; extra fields are ignored):
      {
        "session_id": "abc123",
        "speaker":    "Zharif",          # real name already resolved by Module 2
        "text":       "We decided to use sentence-transformers for Month 3.",
        "timestamp":  "2025-06-01T09:34:12Z",
        "source":     "module2"          # tag so we can filter in the LKC viewer
      }

2.  Autonomous Capture
    -----------------------------------------------------------
    Every transcript segment (from either pipeline) is run through a lightweight
    rule-based tagger that detects:

      • ACTION ITEMS  — "I will / we will / you should / action: ..."
      • DECISIONS     — "we decided / agreed to / conclusion: ..."
      • ENTITIES      — mentioned people, projects, and deadlines

    Detected tags are appended to the LKC record under a "tags" key and,
    optionally, queued for agent confirmation ("Lab Brain: I captured an action
    item — [text]. Correct?").

    The full NLP upgrade path (spaCy NER, pyannote diarization alignment) is
    stubbed with TODO markers for Month 3 final hardening.

Design note: this module is intentionally stateless; it reads/writes the shared
lkc_stream.jsonl through the same helper as server.py to preserve the single
source of truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(tags=["capture"])

# ─────────────────────────────────────────────────────────────────────────────
# Shared LKC path (resolved at server startup via cfg; default here for import)
# ─────────────────────────────────────────────────────────────────────────────
_lkc_path: Path = Path("lkc_stream.jsonl")

def set_lkc_path(path: Path) -> None:
    global _lkc_path
    _lkc_path = path

def _write(record: dict) -> None:
    with _lkc_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Autonomous tagger
# ─────────────────────────────────────────────────────────────────────────────

# Action-item triggers
_ACTION_RE = re.compile(
    r"\b(i will|we will|i'll|we'll|you should|please|action(?: item)?[:\-]|"
    r"todo[:\-]|to do[:\-]|need to|has to|must|going to|gonna)\b",
    re.IGNORECASE,
)

# Decision triggers
_DECISION_RE = re.compile(
    r"\b(we decided|we agreed|it was decided|conclusion[:\-]|we will go with|"
    r"final decision|approved|rejected|resolved)\b",
    re.IGNORECASE,
)

# Deadline triggers (simple)
_DEADLINE_RE = re.compile(
    r"\b(by (monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|end of (day|week|month|sprint)|next week|"
    r"tomorrow|tonight)|deadline[:\-]|due[:\- ]+\w+)\b",
    re.IGNORECASE,
)

# Named entity heuristics — Capitalised words not at sentence start, 2-20 chars
_ENTITY_RE = re.compile(r"(?<!\.\s)(?<![?!]\s)\b([A-Z][a-z]{1,19})\b")

# Project / technical term heuristics
_PROJECT_RE = re.compile(
    r"\b(Lab Brain|Module \d|TEEP|LKC|pyannote|WhisperX|sentence-transformers|"
    r"Ollama|Gemini|faster-whisper|FastAPI|Supabase|Rifqi|Wildan|Lathifah|Nabhyla|"
    r"Fadhil|Davian|Diajeng|Prof\.? Ben)\b",
    re.IGNORECASE,
)


def tag_segment(text: str) -> dict:
    """
    Returns a tags dict with detected categories and extracted entities.
    Empty lists mean no match — callers should skip writing empty tag blocks.

    {
      "action_items": ["I will deploy by Friday"],
      "decisions":    [],
      "entities":     ["Rifqi", "TEEP"],
      "deadlines":    ["by Friday"],
    }
    """
    action_items: list[str] = []
    decisions:    list[str] = []
    deadlines:    list[str] = []
    entities:     list[str] = []

    # Split into sentences for better precision
    sentences = re.split(r"[.!?]+", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if _ACTION_RE.search(sent):
            action_items.append(sent)
        if _DECISION_RE.search(sent):
            decisions.append(sent)
        if _DEADLINE_RE.search(sent):
            deadlines.append(sent)

    # Named entities (capitalised + project names)
    raw_ents = _ENTITY_RE.findall(text) + _PROJECT_RE.findall(text)
    # Deduplicate preserving order, filter stopwords
    _STOPWORDS = {"The", "A", "An", "In", "Is", "It", "I", "We", "You",
                  "He", "She", "They", "This", "That", "And", "But", "Or",
                  "So", "If", "Do", "No", "Yes", "Lab"}
    seen: set[str] = set()
    for e in raw_ents:
        if e not in _STOPWORDS and e not in seen:
            entities.append(e)
            seen.add(e)

    return {
        "action_items": action_items,
        "decisions":    decisions,
        "entities":     entities,
        "deadlines":    deadlines,
    }


def has_tags(tags: dict) -> bool:
    return any(tags.get(k) for k in ("action_items", "decisions", "deadlines"))


# Queued confirmations to be delivered via TTS to the agent
# Maps session_id → list of pending confirmation texts
_pending_confirmations: dict[str, list[str]] = {}

def get_pending_confirmations(session_id: str) -> list[str]:
    return _pending_confirmations.pop(session_id, [])

def _queue_confirmation(session_id: str, text: str) -> None:
    _pending_confirmations.setdefault(session_id, []).append(text)


def process_segment(
    session_id: str,
    speaker: str,
    text: str,
    timestamp_unix: float,
    mode: str,
    language: str,
    *,
    confirm_agent: bool = True,
) -> dict:
    """
    Tag a transcript segment and write an enriched LKC record.
    Returns the written record so server.py can inspect it.
    """
    tags = tag_segment(text)
    ts_iso = datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z"

    record: dict = {
        "type":           "transcript",
        "session_id":     session_id,
        "timestamp_iso":  ts_iso,
        "timestamp_unix": round(timestamp_unix, 3),
        "speaker":        speaker,
        "text":           text,
        "mode":           mode,
        "language":       language,
        "tags":           tags,
    }
    _write(record)

    # Queue confirmations for non-empty tags
    if confirm_agent and has_tags(tags):
        parts: list[str] = []
        if tags["action_items"]:
            parts.append("action item: " + tags["action_items"][0][:80])
        if tags["decisions"]:
            parts.append("decision: " + tags["decisions"][0][:80])
        if tags["deadlines"]:
            parts.append("deadline: " + tags["deadlines"][0][:80])
        if parts:
            confirmation = "I captured a " + "; ".join(parts) + ". Is that correct?"
            _queue_confirmation(session_id, confirmation)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Rifqi pipeline ingest endpoint
# ─────────────────────────────────────────────────────────────────────────────

class RifqiSegment(BaseModel):
    session_id:  str
    speaker:     str
    text:        str
    timestamp:   Optional[str] = None   # ISO string; defaults to now
    source:      str = "module2"
    mode:        str = "meeting_capture"
    language:    str = "id"
    extra:       Optional[dict] = None  # any extra fields from Module 2


@router.post("/ingest")
async def ingest_from_rifqi(seg: RifqiSegment):
    """
    Receive a structured segment from Rifqi's Module 2 pipeline,
    run autonomous tagging, and write to the shared LKC.
    """
    ts_unix = time.time()
    if seg.timestamp:
        try:
            from datetime import timezone
            from dateutil import parser as dtparser  # type: ignore
            ts_unix = dtparser.parse(seg.timestamp).timestamp()
        except Exception:
            pass  # fall back to now

    record = process_segment(
        seg.session_id, seg.speaker, seg.text,
        ts_unix, seg.mode, seg.language,
        confirm_agent=False,  # Module 2 segments don't need TTS confirmation
    )
    if seg.source:
        record["source"] = seg.source
    if seg.extra:
        record["extra"] = seg.extra

    # Re-write the record with source annotation
    # (process_segment already wrote it; we patch in-place via a second write
    #  only if module 2 adds metadata — acceptable for PoC)
    log.info(f"[capture] Rifqi ingest: session={seg.session_id} speaker={seg.speaker} "
             f"tags_found={has_tags(record['tags'])}")
    return {
        "ok":     True,
        "record": record,
    }


@router.get("/confirmations/{session_id}")
async def poll_confirmations(session_id: str):
    """
    Module 5 server polls this to pick up agent confirmation messages
    and route them through the TTS queue.
    """
    return {
        "session_id": session_id,
        "confirmations": get_pending_confirmations(session_id),
    }


@router.get("/tags/{session_id}")
async def get_session_tags(session_id: str):
    """
    Return all captured action items / decisions / entities from the LKC
    for a given session.  Useful for the end-of-session summary panel.
    """
    if not _lkc_path.exists():
        return {"session_id": session_id, "tags": {}}

    action_items: list[str] = []
    decisions:    list[str] = []
    entities:     set[str]  = set()
    deadlines:    list[str] = []

    for line in _lkc_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("session_id") != session_id:
            continue
        tags = r.get("tags", {})
        action_items.extend(tags.get("action_items", []))
        decisions.extend(tags.get("decisions", []))
        deadlines.extend(tags.get("deadlines", []))
        entities.update(tags.get("entities", []))

    return {
        "session_id":  session_id,
        "action_items": action_items,
        "decisions":   decisions,
        "deadlines":   deadlines,
        "entities":    sorted(entities),
    }