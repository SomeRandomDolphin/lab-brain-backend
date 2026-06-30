"""
app/services/capture.py — Autonomous Capture & Tagging Service.

Responsibilities:
  - spaCy NER entity extraction (regex fallback)
  - Wake-word / summon system
  - Action item / decision / deadline tagging
  - Segment enrichment and LKC write delegation
  - Rifqi Module 2 ingest compatibility
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.db import lkc_graph

log = logging.getLogger(__name__)


# ── spaCy NER ─────────────────────────────────────────────────────────────────
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
                "[capture] spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None
        SPACY_AVAILABLE = True
        log.info("[capture] spaCy NER loaded (en_core_web_sm).")
        return _nlp
    except ImportError:
        log.warning("[capture] spaCy not installed — using regex NER.")
        return None


# ── Regex patterns (fallback) ──────────────────────────────────────────────────
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
    r"\b(by (monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"end of (day|week|month|sprint)|next week|tomorrow|tonight)|"
    r"deadline[:\-]|due[:\- ]+\w+)\b",
    re.IGNORECASE,
)
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
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in _SPACY_ENTITY_TYPES:
            continue
        norm = ent.text.strip()
        if norm and norm not in seen and len(norm) > 1:
            entities.append(norm)
            seen.add(norm)
    return entities


def _extract_entities_regex(text: str) -> list[str]:
    raw_ents = _ENTITY_RE.findall(text) + _PROJECT_RE.findall(text)
    seen: set[str] = set()
    entities: list[str] = []
    for e in raw_ents:
        if e not in _STOPWORDS and e not in seen:
            entities.append(e)
            seen.add(e)
    return entities


def _extract_entities(text: str) -> list[str]:
    if SPACY_AVAILABLE or _load_spacy() is not None:
        return _extract_entities_spacy(text)
    return _extract_entities_regex(text)


# ── Wake-word / Summon System ─────────────────────────────────────────────────
SUMMON_PHRASES: list[str] = [
    "lab brain", "hey brain", "hey lab brain",
    "@lab", "brain,", "brain?", "lab,",
]

_SUMMON_RE = re.compile(
    "|".join(re.escape(p) for p in SUMMON_PHRASES),
    re.IGNORECASE,
)

_summon_state: dict[str, bool] = {}


def check_summon(session_id: str, text: str) -> bool:
    if _SUMMON_RE.search(text):
        _summon_state[session_id] = True
        log.info(f"[capture:{session_id}] Agent summoned via wake-word.")
        return True
    return False


def is_summoned(session_id: str) -> bool:
    return _summon_state.get(session_id, False)


def clear_summon(session_id: str) -> None:
    _summon_state.pop(session_id, None)


def force_summon(session_id: str) -> None:
    _summon_state[session_id] = True


# ── Tagger ────────────────────────────────────────────────────────────────────

def tag_segment(text: str) -> dict:
    action_items: list[str] = []
    decisions:    list[str] = []
    deadlines:    list[str] = []

    for sent in re.split(r"[.!?]+", text):
        sent = sent.strip()
        if not sent:
            continue
        if _ACTION_RE.search(sent):
            action_items.append(sent)
        if _DECISION_RE.search(sent):
            decisions.append(sent)
        if _DEADLINE_RE.search(sent):
            deadlines.append(sent)

    return {
        "action_items": action_items,
        "decisions":    decisions,
        "entities":     _extract_entities(text),
        "deadlines":    deadlines,
    }


def has_tags(tags: dict) -> bool:
    return any(tags.get(k) for k in ("action_items", "decisions", "deadlines"))


# ── Confirmation queue ─────────────────────────────────────────────────────────
_pending_confirmations: dict[str, list[str]] = {}


def get_pending_confirmations(session_id: str) -> list[str]:
    return _pending_confirmations.pop(session_id, [])


def _queue_confirmation(session_id: str, text: str) -> None:
    _pending_confirmations.setdefault(session_id, []).append(text)


# ── Core segment processor ────────────────────────────────────────────────────

async def process_segment(
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

    await lkc_graph.write_to_lkc(record)

    if confirm_agent and has_tags(tags):
        parts: list[str] = []
        if tags["action_items"]:
            parts.append("action item: " + tags["action_items"][0][:80])
        if tags["decisions"]:
            parts.append("decision: " + tags["decisions"][0][:80])
        if tags["deadlines"]:
            parts.append("deadline: " + tags["deadlines"][0][:80])
        if parts:
            _queue_confirmation(
                session_id,
                "I captured a " + "; ".join(parts) + ". Is that correct?",
            )

    return record
