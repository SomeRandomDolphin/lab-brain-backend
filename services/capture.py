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

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

import os
from db import lkc_graph

log = logging.getLogger(__name__)

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

# ── spaCy NER ─────────────────────────────────────────────────────────────────
# Loaded eagerly at import time (mirrors asr.py's WhisperX load) so the model
# is warm before the first meeting starts. Previously this module loaded spaCy
# here and then immediately overwrote `_nlp` back to None two lines later,
# which forced a ~60s synchronous re-load on the first transcribed segment of
# every meeting — during which incoming audio silently piled up and dropped.
_SPACY_ENTITY_TYPES = {"PERSON", "ORG", "PRODUCT", "GPE", "DATE", "EVENT", "WORK_OF_ART"}

SPACY_AVAILABLE = False
_nlp = None

try:
    import spacy
    log.info(f"[capture] loading spaCy NER model '{SPACY_MODEL}'…")
    try:
        _nlp = spacy.load(SPACY_MODEL)
        SPACY_AVAILABLE = True
        log.info("[capture] spaCy NER loaded.")
    except OSError:
        log.warning(
            f"[capture] spaCy model '{SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        )
except ImportError:
    log.warning("[capture] spaCy not installed — using regex NER.")


def _load_spacy():
    """Returns the eagerly-loaded model, or None if it failed/isn't installed.
    Kept as a function (rather than inlining `_nlp`) so call sites don't need
    to change, but it no longer does any loading itself — that always happens
    once, at import time, above."""
    return _nlp


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


def dismiss_agent(session_id: str) -> None:
    """
    Manual "dismiss agent" action (the summon button's DELETE path).

    Deliberately NOT the same thing as clear_summon(): clear_summon() is
    also called by session_pipeline.py's _handle_segment the instant QA
    mode is entered, to consume the summon flag before the reply task is
    spawned -- that happens on every QA turn, well before the user has any
    chance to click dismiss. If this function's mode-reset were folded into
    clear_summon() instead, it would fire at THAT point too and kill every
    QA reply before it starts.

    This is for the user-initiated case only: clear the summon flag (in
    case a reply is still pending) AND force the dialogue state out of QA
    immediately, regardless of whether a reply has finished generating.
    See dialogue_service.force_exit_qa for why a dedicated escape hatch was
    needed there.
    """
    _summon_state.pop(session_id, None)
    from pipeline.dialogue_service import force_exit_qa
    force_exit_qa(session_id)


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
    # Drop empty transcriptions early — these are silence flushes from the VAD
    # and would pollute the LKC graph and the SSE stream with blank records.
    if not text.strip():
        return {}

    # tag_segment() is synchronous CPU work — critically, _extract_entities
    # runs a real spaCy inference pass (nlp(text)) when spaCy is available.
    # Calling it inline here (as before) blocks the ENTIRE event loop for its
    # duration: not just this session's next segment, but every other
    # session's audio/video consumers and any in-flight SSE/metrics request,
    # process-wide. asr.py already guards against exactly this for
    # WhisperX/faster-whisper via run_in_executor; tag_segment never got the
    # same treatment. Offloading it here fixes that.
    loop = asyncio.get_event_loop()
    tags = await loop.run_in_executor(None, tag_segment, text)
    ts_iso = datetime.utcfromtimestamp(timestamp_unix).isoformat() + "Z"

    # ── Speaker consistency check ─────────────────────────────────────────────
    # The segment-level speaker (from pyannote coarse diarization) and the
    # word-level speakers (from whisperx.assign_word_speakers fine diarization)
    # can disagree when a turn boundary falls mid-segment.  When they conflict,
    # use the majority-vote word-level speaker as the authoritative label and
    # log a warning so the diarization pipeline can be audited.
    resolved_speaker = speaker
    if word_timestamps:
        word_speakers = [w.get("speaker") for w in word_timestamps if w.get("speaker")]
        if word_speakers:
            # Majority vote among word-level speaker labels
            majority = max(set(word_speakers), key=word_speakers.count)
            if majority != speaker:
                log.warning(
                    f"[capture:{session_id}] speaker mismatch — "
                    f"segment={speaker!r}, word-level majority={majority!r} "
                    f"(votes: {word_speakers}). Using word-level majority."
                )
                resolved_speaker = majority

    record: dict = {
        "type":           "transcript",
        "session_id":     session_id,
        "timestamp_iso":  ts_iso,
        "timestamp_unix": round(timestamp_unix, 3),
        "speaker":        resolved_speaker,
        "text":           text,
        "mode":           mode,
        "language":       language,
        "tags":           tags,
    }
    if word_timestamps:
        record["word_timestamps"] = word_timestamps

    try:
        await lkc_graph.write_to_lkc(record)
    except Exception as exc:
        # Log but do NOT propagate — the record is still valid.  Callers must
        # be able to broadcast it to SSE even when LKC persistence is degraded.
        log.error(
            f"[capture:{session_id}] lkc_graph.write_to_lkc failed: {exc}",
            exc_info=True,
        )

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