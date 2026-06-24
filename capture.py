"""
capture.py — Autonomous Capture & Rifqi Pipeline Integration (Module 5, Month 5)

Month 5 changes
---------------
1.  spaCy NER replaces regex entity extraction.
    The default model is `en_core_web_sm` (downloaded automatically on first
    use via `python -m spacy download en_core_web_sm`).  The regex _ENTITY_RE
    and _PROJECT_RE patterns are kept as a fallback when spaCy is unavailable.
    Entity types extracted: PERSON, ORG, PRODUCT, GPE, DATE, EVENT, WORK_OF_ART

2.  Wake-word / summon system — the agent now replies ONLY when explicitly
    addressed.  Any transcript containing a configured summon phrase (default:
    "lab brain", "hey brain", "brain", "@lab") sets a session-level summon
    flag that server.py reads before triggering QA mode.
    This prevents the agent from interrupting every question in the room.

    API:
        summoned = capture.check_summon(session_id, text)   # True/False
        capture.clear_summon(session_id)                     # reset after reply

Unchanged from Month 3/4
------------------------
* Rifqi Module 2 ingest endpoint (POST /capture/ingest)
* Autonomous action-item / decision / deadline detection
* Confirmation queue for agent TTS verification
* write_to_lkc() now delegates to lkc_graph.write_to_lkc() (Month 5)
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

# ── Month 5: delegate LKC writes to the persistent graph ─────────────────────
import lkc_graph

def set_lkc_path(path: Path) -> None:
    """Keep backward-compat signature; Month 5 uses the graph, not raw JSONL."""
    lkc_graph.configure(
        db_path=path.with_suffix(".db"),
        jsonl_path=path,
    )

def _write(record: dict) -> None:
    lkc_graph.write_to_lkc(record)


# ── Month 5: spaCy NER ────────────────────────────────────────────────────────
SPACY_AVAILABLE = False
_nlp = None

_SPACY_ENTITY_TYPES = {"PERSON", "ORG", "PRODUCT", "GPE", "DATE", "EVENT", "WORK_OF_ART"}

def _load_spacy():
    global _nlp, SPACY_AVAILABLE
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning(
                "[capture] spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
                " — falling back to regex NER."
            )
            return None
        SPACY_AVAILABLE = True
        log.info("[capture] spaCy NER loaded (en_core_web_sm).")
        return _nlp
    except ImportError:
        log.warning("[capture] spaCy not installed — falling back to regex NER.")
        return None


# ── Regex patterns (kept as fallback) ─────────────────────────────────────────

_ACTION_RE = re.compile(
    r"\b(i will|we will|i'll|we'll|you should|please|action(?: item)?[:\-]|"
    r"todo[:\-]|to do[:\-]|need to|has to|must|going to|gonna)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(we decided|we agreed|it was decided|conclusion[:\-]|we will go with|"
    r"final decision|approved|rejected|resolved)\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(by (monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|end of (day|week|month|sprint)|next week|"
    r"tomorrow|tonight)|deadline[:\-]|due[:\- ]+\w+)\b",
    re.IGNORECASE,
)
# Regex fallback entity patterns (used only when spaCy is unavailable)
_ENTITY_RE  = re.compile(r"(?<!\.\s)(?<![?!]\s)\b([A-Z][a-z]{1,19})\b")
_PROJECT_RE = re.compile(
    r"\b(Lab Brain|Module \d|TEEP|LKC|pyannote|WhisperX|sentence-transformers|"
    r"Ollama|Gemini|faster-whisper|FastAPI|Supabase|Rifqi|Wildan|Lathifah|Nabhyla|"
    r"Fadhil|Davian|Diajeng|Prof\.? Ben)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "The", "A", "An", "In", "Is", "It", "I", "We", "You",
    "He", "She", "They", "This", "That", "And", "But", "Or",
    "So", "If", "Do", "No", "Yes", "Lab",
}


def _extract_entities_spacy(text: str) -> list[str]:
    """Use spaCy NER to extract meaningful named entities."""
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        label = ent.label_
        if label not in _SPACY_ENTITY_TYPES:
            continue
        norm = ent.text.strip()
        if norm and norm not in seen and len(norm) > 1:
            entities.append(norm)
            seen.add(norm)
    return entities


def _extract_entities_regex(text: str) -> list[str]:
    """Regex-based entity extraction (Month 3/4 fallback)."""
    raw_ents = _ENTITY_RE.findall(text) + _PROJECT_RE.findall(text)
    seen: set[str] = set()
    entities: list[str] = []
    for e in raw_ents:
        if e not in _STOPWORDS and e not in seen:
            entities.append(e)
            seen.add(e)
    return entities


def _extract_entities(text: str) -> list[str]:
    """Try spaCy first; fall back to regex."""
    if SPACY_AVAILABLE or _load_spacy() is not None:
        return _extract_entities_spacy(text)
    return _extract_entities_regex(text)


# ── Month 5: Wake-word / Summon System ───────────────────────────────────────
# The agent only enters QA mode when explicitly summoned.
# Add / remove phrases here; matching is case-insensitive.
SUMMON_PHRASES: list[str] = [
    "lab brain",
    "hey brain",
    "hey lab brain",
    "@lab",
    "brain,",
    "brain?",
    "lab,",
]

_SUMMON_RE = re.compile(
    "|".join(re.escape(p) for p in SUMMON_PHRASES),
    re.IGNORECASE,
)

# Per-session summon state: session_id → True if the agent was summoned in the
# last segment and has not yet replied.
_summon_state: dict[str, bool] = {}


def check_summon(session_id: str, text: str) -> bool:
    """
    Return True if the transcript explicitly addresses Lab Brain.
    Sets a sticky flag so the agent can reply on the same turn.

    Call this BEFORE update_mode() in server.py.
    """
    if _SUMMON_RE.search(text):
        _summon_state[session_id] = True
        log.info(f"[capture:{session_id}] Agent summoned via wake-word.")
        return True
    return False


def is_summoned(session_id: str) -> bool:
    """Check whether the session has a pending (unanswered) summon."""
    return _summon_state.get(session_id, False)


def clear_summon(session_id: str) -> None:
    """Reset the summon flag after the agent has replied."""
    _summon_state.pop(session_id, None)


# ── Tagger ────────────────────────────────────────────────────────────────────

def tag_segment(text: str) -> dict:
    """
    Returns a tags dict with detected categories and extracted entities.

    Month 5: entity extraction uses spaCy NER (regex fallback).

    {
      "action_items": ["I will deploy by Friday"],
      "decisions":    [],
      "entities":     ["Rifqi", "TEEP", "2026-06-03"],
      "deadlines":    ["by Friday"],
    }
    """
    action_items: list[str] = []
    decisions:    list[str] = []
    deadlines:    list[str] = []

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

    entities = _extract_entities(text)

    return {
        "action_items": action_items,
        "decisions":    decisions,
        "entities":     entities,
        "deadlines":    deadlines,
    }


def has_tags(tags: dict) -> bool:
    return any(tags.get(k) for k in ("action_items", "decisions", "deadlines"))


# ── Confirmation queue ────────────────────────────────────────────────────────

_pending_confirmations: dict[str, list[str]] = {}

def get_pending_confirmations(session_id: str) -> list[str]:
    return _pending_confirmations.pop(session_id, [])

def _queue_confirmation(session_id: str, text: str) -> None:
    _pending_confirmations.setdefault(session_id, []).append(text)


# ── Core segment processor ────────────────────────────────────────────────────

def process_segment(
    session_id: str,
    speaker: str,
    text: str,
    timestamp_unix: float,
    mode: str,
    language: str,
    *,
    confirm_agent: bool = True,
    word_timestamps: Optional[list[dict]] = None,
) -> dict:
    """
    Tag a transcript segment and write an enriched LKC record to the graph.
    Returns the written record so server.py can inspect tags / word timestamps.
    """
    tags   = tag_segment(text)
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
    if word_timestamps:
        record["word_timestamps"] = word_timestamps

    _write(record)

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


# ── Rifqi Module 2 ingest endpoint ────────────────────────────────────────────

class RifqiSegment(BaseModel):
    session_id: str
    speaker:    str
    text:       str
    timestamp:  Optional[str] = None
    source:     str = "module2"
    mode:       str = "meeting_capture"
    language:   str = "id"
    extra:      Optional[dict] = None


@router.post("/ingest")
async def ingest_from_rifqi(seg: RifqiSegment):
    """
    Receive a structured segment from Rifqi's Module 2 pipeline,
    run autonomous tagging (including spaCy NER), and write to the LKC graph.
    """
    ts_unix = time.time()
    if seg.timestamp:
        try:
            from datetime import timezone
            from dateutil import parser as dtparser
            ts_unix = dtparser.parse(seg.timestamp).timestamp()
        except Exception:
            pass

    record = process_segment(
        seg.session_id, seg.speaker, seg.text,
        ts_unix, seg.mode, seg.language,
        confirm_agent=False,
    )
    if seg.source:
        record["source"] = seg.source
    if seg.extra:
        record["extra"] = seg.extra

    log.info(
        f"[capture] Rifqi ingest: session={seg.session_id} speaker={seg.speaker} "
        f"tags={has_tags(record['tags'])}"
    )
    return {"ok": True, "record": record}


@router.get("/confirmations/{session_id}")
async def poll_confirmations(session_id: str):
    return {
        "session_id":    session_id,
        "confirmations": get_pending_confirmations(session_id),
    }


@router.get("/tags/{session_id}")
async def get_session_tags(session_id: str):
    """
    Return all captured tags from the persistent LKC graph for a session.
    Month 5: reads from SQLite, not by scanning lkc_stream.jsonl.
    """
    records = lkc_graph.read_lkc(
        session_id=session_id, record_type="transcript"
    )

    action_items: list[str] = []
    decisions:    list[str] = []
    entities:     set[str]  = set()
    deadlines:    list[str] = []

    for r in records:
        tags = r.get("tags", {})
        action_items.extend(tags.get("action_items", []))
        decisions.extend(tags.get("decisions", []))
        deadlines.extend(tags.get("deadlines", []))
        entities.update(tags.get("entities", []))

    return {
        "session_id":   session_id,
        "action_items": action_items,
        "decisions":    decisions,
        "deadlines":    deadlines,
        "entities":     sorted(entities),
    }


@router.get("/ner_backend")
async def ner_backend_status():
    """Report which NER backend is active."""
    _load_spacy()
    return {
        "backend":   "spacy_en_core_web_sm" if SPACY_AVAILABLE else "regex_fallback",
        "spacy_available": SPACY_AVAILABLE,
    }
