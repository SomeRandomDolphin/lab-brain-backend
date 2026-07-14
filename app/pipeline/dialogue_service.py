"""
app/pipeline/dialogue_service.py — Conversation Mode FSM + Agent Replies.

Ported from the flat dialogue.py into the structured pipeline package.
No logic changes — only import paths updated.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from app.core.config import cfg

log = logging.getLogger(__name__)

# ── pyannote diarization ──────────────────────────────────────────────────────
try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

_diarization_pipeline: Optional["PyannotePipeline"] = None
_diarization_pipeline_error: Optional[str] = None


def _extract_turns(diarization) -> list[tuple[float, float, str]]:
    """
    Normalize pyannote diarization output across API versions into a flat
    list of (start, end, speaker_label) tuples.

    - pyannote.audio <= 3.x: pipeline(...) returns a pyannote.core.Annotation
      directly, with turns via annotation.itertracks(yield_label=True) ->
      (Segment, track_id, label).
    - pyannote.audio 4.x (community-1 / precision-2 pipelines): pipeline(...)
      returns a DiarizeOutput wrapper; the Annotation lives at
      output.speaker_diarization, and per pyannote's own docs is iterated
      directly as (Segment, label) pairs rather than via itertracks().
    """
    ann = getattr(diarization, "speaker_diarization", diarization)

    if hasattr(ann, "itertracks"):
        return [(turn.start, turn.end, speaker) for turn, _, speaker in ann.itertracks(yield_label=True)]

    # Newer wrapper: iterate directly as (Segment, label) pairs.
    return [(turn.start, turn.end, speaker) for turn, speaker in ann]


def _get_diarization_pipeline():
    global _diarization_pipeline, _diarization_pipeline_error
    if _diarization_pipeline is not None:
        return _diarization_pipeline
    if _diarization_pipeline_error is not None:
        return None
    if not PYANNOTE_AVAILABLE:
        _diarization_pipeline_error = "pyannote.audio not installed"
        log.warning("[diarization] pyannote.audio not installed — using round-robin fallback")
        return None
    try:
        import os
        hf_token = (
            cfg.local_llm.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        use_auth  = {"token": hf_token} if hf_token else {}
        pipeline  = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", **use_auth
        )
        device    = "cuda" if (PYANNOTE_AVAILABLE and torch.cuda.is_available()) else "cpu"
        pipeline.to(torch.device(device))
        _diarization_pipeline = pipeline
        log.info(f"[diarization] pyannote pipeline loaded on {device}")
        return pipeline
    except Exception as exc:
        _diarization_pipeline_error = str(exc)
        log.warning(f"[diarization] pipeline load failed ({exc}) — using round-robin fallback")
        return None


# Pre-load the diarization pipeline at process startup rather than lazily on
# the first live audio segment. Loading it can trigger multi-file model
# downloads (config.yaml, pytorch_model.bin, xvec_transform.npz, etc.) from
# Hugging Face — if that happens on the first segment of a live meeting, it
# blocks real-time speaker assignment for however long the download takes,
# silently dropping speaker attribution (and, transitively, transcription
# timing) for the start of the session. Doing it here means any download or
# gating failure surfaces once at boot, not mid-meeting.
if PYANNOTE_AVAILABLE:
    log.info("[diarization] pre-loading pipeline at startup…")
    _get_diarization_pipeline()


# ── Local LLM client ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _dialogue_client = OpenAI(
        base_url=cfg.local_llm.base_url,
        api_key=cfg.local_llm.api_key,
    )
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

_SYSTEM_PROMPT = (
    "You are Lab Brain, a helpful AI assistant embedded in a research "
    "laboratory meeting room. You are concise (≤2 sentences), professional, "
    "and always grounded in the lab context provided. Never hallucinate "
    "citations or project details. You only respond when directly addressed."
)


# ── Mode FSM ──────────────────────────────────────────────────────────────────
class ConvMode(str, Enum):
    GREETING        = "greeting"
    MEETING_CAPTURE = "meeting_capture"
    QA              = "qa"
    AMBIENT         = "ambient"
    CONFIRMATION    = "confirmation"


_QUESTION_RE = re.compile(
    r"\b(what|who|where|when|why|how|is|are|was|were|can|could|should|would|"
    r"did|do|does|tell me|explain|summarize|recap|define)\b",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_RE.search(text))


_AFFIRM_RE = re.compile(
    r"\b(yes|correct|right|sure|exactly|affirmative|yep|yeah|confirmed|ok|okay|go ahead)\b",
    re.IGNORECASE,
)
_DENY_RE = re.compile(
    r"\b(no|nope|wrong|incorrect|never mind|nevermind|skip|cancel|ignore|not quite)\b",
    re.IGNORECASE,
)


def _is_affirmation(text: str) -> bool:
    return bool(_AFFIRM_RE.search(text))


def _is_denial(text: str) -> bool:
    return bool(_DENY_RE.search(text))


# ── Dialogue state ────────────────────────────────────────────────────────────
@dataclass
class DialogueState:
    session_id:           str
    mode:                 ConvMode      = ConvMode.AMBIENT
    mode_entered_at:      float         = field(default_factory=time.time)
    greeted_speakers:     set[str]      = field(default_factory=set)
    _chat_history:        list          = field(default_factory=list, repr=False)
    transcript_context:   list[str]     = field(default_factory=list)
    CONTEXT_WINDOW:       int           = field(default_factory=lambda: cfg.dialogue.context_window)
    confirmation_pending: Optional[str] = None
    _speaker_counter:     int           = field(default=0, repr=False)

    def push_context(self, speaker: str, text: str) -> None:
        self.transcript_context.append(f"{speaker}: {text}")
        if len(self.transcript_context) > self.CONTEXT_WINDOW:
            self.transcript_context.pop(0)

    def context_block(self) -> str:
        return "\n".join(self.transcript_context)

    def get_chat_history(self) -> list:
        return self._chat_history


_SPEAKER_LABELS = [f"Person {chr(65+i)}" for i in range(8)]
_SAMPLE_RATE    = 16000

_dialogue_states: dict[str, DialogueState] = {}


def get_dialogue(session_id: str) -> DialogueState:
    if session_id not in _dialogue_states:
        _dialogue_states[session_id] = DialogueState(session_id=session_id)
    return _dialogue_states[session_id]


def clear_dialogue(session_id: str) -> None:
    _dialogue_states.pop(session_id, None)


def push_context(state: DialogueState, speaker: str, text: str) -> None:
    state.push_context(speaker, text)


# ── Speaker assignment ────────────────────────────────────────────────────────

async def assign_speaker(
    state: DialogueState,
    audio_segment=None,
) -> str:
    pipeline = _get_diarization_pipeline()
    if pipeline is not None and audio_segment is not None:
        try:
            import torch
            wav = (
                np.frombuffer(audio_segment, dtype=np.float32).copy()
                if isinstance(audio_segment, bytes)
                else audio_segment.astype(np.float32)
            )
            min_samples = int(0.5 * _SAMPLE_RATE)
            if wav.size < min_samples:
                raise ValueError(f"Segment too short ({wav.size} samples)")

            waveform = torch.from_numpy(wav).unsqueeze(0)
            # Run the blocking pyannote inference in a thread so it doesn't
            # freeze the event loop — this is what was starving the LiveKit
            # audio/video drain coroutines and causing dropped frames.
            loop = asyncio.get_event_loop()
            diarization = await loop.run_in_executor(
                None, lambda: pipeline({"waveform": waveform, "sample_rate": _SAMPLE_RATE})
            )

            duration_per_speaker: dict[str, float] = {}
            for start, end, speaker in _extract_turns(diarization):
                duration_per_speaker[speaker] = duration_per_speaker.get(speaker, 0.0) + (end - start)

            if not duration_per_speaker:
                raise ValueError("No speakers detected")

            dominant_raw = max(duration_per_speaker, key=duration_per_speaker.get)

            if not hasattr(state, "_speaker_map"):
                state._speaker_map: dict[str, str] = {}
            if dominant_raw not in state._speaker_map:
                idx = len(state._speaker_map)
                state._speaker_map[dominant_raw] = (
                    _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS)
                    else f"Person {idx + 1}"
                )
            return state._speaker_map[dominant_raw]

        except Exception as exc:
            log.warning(f"[diarization:{state.session_id}] inference failed ({exc}) — round-robin fallback")

    label = _SPEAKER_LABELS[state._speaker_counter % len(_SPEAKER_LABELS)]
    state._speaker_counter += 1
    return label


async def assign_speaker_words(
    state: DialogueState,
    word_timestamps: list[dict],
    audio_segment: np.ndarray,
) -> list[dict]:
    if not word_timestamps:
        return word_timestamps

    pipeline = _get_diarization_pipeline()
    if pipeline is None:
        fallback = await assign_speaker(state, audio_segment)
        return [{**w, "speaker": fallback} for w in word_timestamps]

    try:
        import torch
        wav = audio_segment.astype(np.float32)
        # Offload the blocking pyannote inference to a thread so it doesn't
        # freeze the event loop (see assign_speaker for the same fix).
        loop = asyncio.get_event_loop()
        diarization = await loop.run_in_executor(
            None,
            lambda: pipeline({"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": _SAMPLE_RATE}),
        )

        if not hasattr(state, "_speaker_map"):
            state._speaker_map = {}

        turns: list[tuple[float, float, str]] = []
        for start, end, raw_label in _extract_turns(diarization):
            if raw_label not in state._speaker_map:
                idx = len(state._speaker_map)
                state._speaker_map[raw_label] = (
                    _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS)
                    else f"Person {idx + 1}"
                )
            turns.append((start, end, state._speaker_map[raw_label]))

        # Fallback label for words that fall outside every detected turn and
        # there are no turns at all — resolved once, up front, since this
        # helper must stay synchronous (it's used inside a list comprehension).
        no_turns_fallback = await assign_speaker(state, None) if not turns else None

        def _speaker_at(t: float) -> str:
            for start, end, label in turns:
                if start <= t <= end:
                    return label
            if not turns:
                return no_turns_fallback
            nearest = min(turns, key=lambda x: min(abs(x[0] - t), abs(x[1] - t)))
            return nearest[2]

        enriched = [
            {**w, "speaker": _speaker_at((w.get("start", 0.0) + w.get("end", 0.0)) / 2.0)}
            for w in word_timestamps
        ]
        log.debug(
            f"[diarization:{state.session_id}] word-level: "
            f"{len(enriched)} words, {len(turns)} turns"
        )
        return enriched

    except Exception as exc:
        log.warning(f"[diarization:{state.session_id}] word-level failed ({exc}) — segment fallback")
        fallback = await assign_speaker(state, audio_segment)
        return [{**w, "speaker": fallback} for w in word_timestamps]


# ── Mode transition FSM ───────────────────────────────────────────────────────

def update_mode(
    state: DialogueState,
    transcript: str,
    present_speakers: list[str],
    new_speakers: list[str],
    pending_confirmation: Optional[str] = None,
    *,
    summoned: bool = False,
) -> tuple[ConvMode, Optional[str]]:
    utterance: Optional[str] = None

    # 0. Resolve pending confirmation
    if state.mode == ConvMode.CONFIRMATION and transcript:
        if _is_affirmation(transcript):
            state.confirmation_pending = None
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
            return state.mode, "Got it, I've logged that."
        elif _is_denial(transcript):
            state.confirmation_pending = None
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
            return state.mode, "Understood, I'll discard that."

    # 0b. Enter CONFIRMATION from pending capture item
    if pending_confirmation and state.mode not in (ConvMode.GREETING, ConvMode.QA):
        state.confirmation_pending = pending_confirmation
        state.mode = ConvMode.CONFIRMATION
        state.mode_entered_at = time.time()
        return state.mode, pending_confirmation

    # 1. New speaker → GREETING
    if new_speakers:
        names = " and ".join(new_speakers)
        state.greeted_speakers.update(new_speakers)
        utterance = f"Hello {names}, welcome. Lab Brain is active and capturing this session."
        state.mode = ConvMode.MEETING_CAPTURE
        state.mode_entered_at = time.time()
        return state.mode, utterance

    # 2. Summoned → QA (question or direct command)
    if transcript and summoned:
        state.mode = ConvMode.QA
        state.mode_entered_at = time.time()
        return state.mode, None

    # 3. Active transcript (non-summoned) → silent MEETING_CAPTURE
    if transcript and len(transcript.split()) >= 3:
        if state.mode in (ConvMode.AMBIENT, ConvMode.QA, ConvMode.CONFIRMATION):
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
        return state.mode, None

    # 4. No speakers → AMBIENT
    if not present_speakers:
        if state.mode != ConvMode.AMBIENT:
            state.mode = ConvMode.AMBIENT
            state.mode_entered_at = time.time()
        return state.mode, None

    return state.mode, utterance


# ── Response generator ────────────────────────────────────────────────────────

async def generate_response(
    state: DialogueState,
    transcript: str,
    lkc_context: str,
) -> Optional[str]:
    if state.mode in (ConvMode.MEETING_CAPTURE, ConvMode.AMBIENT, ConvMode.CONFIRMATION):
        return None

    if state.mode in (ConvMode.QA, ConvMode.GREETING):
        if not LOCAL_LLM_AVAILABLE:
            return "[Local LLM not available — configure local_llm in config.json]"

        lkc_section = (
            f"\n\nRelevant LKC knowledge:\n{lkc_context}"
            if lkc_context.strip() else ""
        )
        user_message = (
            f"Recent conversation:\n{state.context_block()}"
            f"{lkc_section}\n\n"
            f'The speaker directly addressed you and said: "{transcript}"\n'
            f"Respond as Lab Brain in ≤2 sentences."
        )

        history = state.get_chat_history()
        history.append({"role": "user", "content": user_message})

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: _dialogue_client.chat.completions.create(
                    model=cfg.local_llm.dialogue_model,
                    messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + history,
                    max_tokens=120,
                    temperature=0.4,
                )
            )
            reply = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": reply})
            max_turns = state.CONTEXT_WINDOW * 2
            if len(history) > max_turns:
                state._chat_history = history[-max_turns:]
            return reply
        except Exception as exc:
            log.warning(f"[dialogue:{state.session_id}] LLM error: {exc}")
            return None

    return None


# ── Summary generator ─────────────────────────────────────────────────────────

async def generate_summary(state: DialogueState, session_tags: dict) -> str:
    if not LOCAL_LLM_AVAILABLE:
        return "## Session Summary (stub)\nLocal LLM unavailable — configure local_llm.\n"

    action_block   = "\n".join(f"- {a}" for a in session_tags.get("action_items", [])) or "None"
    decision_block = "\n".join(f"- {d}" for d in session_tags.get("decisions",    [])) or "None"
    deadline_block = "\n".join(f"- {d}" for d in session_tags.get("deadlines",    [])) or "None"
    entity_block   = ", ".join(session_tags.get("entities", [])) or "None"

    user_message = (
        f"You are Lab Brain summarising a research meeting.\n\n"
        f"Recent transcript (last {state.CONTEXT_WINDOW} turns):\n{state.context_block()}\n\n"
        f"Action items:\n{action_block}\n\nDecisions:\n{decision_block}\n\n"
        f"Deadlines:\n{deadline_block}\n\nKey entities: {entity_block}\n\n"
        f"Produce a concise meeting summary in markdown: "
        f"## Summary, ## Decisions, ## Action Items, ## Open Questions. "
        f"≤4 bullet points per section."
    )

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _dialogue_client.chat.completions.create(
                model=cfg.local_llm.dialogue_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=400,
                temperature=0.3,
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        log.warning(f"[dialogue:{state.session_id}] summary failed: {exc}")
        return (
            f"## Session Summary (fallback)\n"
            f"**Action items:**\n{action_block}\n\n"
            f"**Decisions:**\n{decision_block}\n\n"
            f"**Deadlines:**\n{deadline_block}\n"
        )