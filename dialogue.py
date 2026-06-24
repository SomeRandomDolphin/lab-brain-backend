"""
dialogue.py — Conversation Mode FSM + Agent Replies (Module 5, Month 5)

Month 5 changes
---------------
1.  Summon-gated QA — the agent replies ONLY when explicitly addressed via a
    wake-word ("Lab Brain", "hey brain", "@lab", …).  The QA mode transition
    in update_mode() now requires capture.is_summoned(session_id) to be True.
    After the agent replies, capture.clear_summon() resets the flag so the
    agent goes back to silent capture mode.

    This prevents the agent from constantly jumping into QA mode whenever
    anyone in the room asks a question to each other.

2.  WhisperX + pyannote word-level speaker alignment — `assign_speaker_words()`
    is a new function that takes pyannote diarization output and WhisperX
    word-level timestamps and returns a word list annotated with speaker IDs.
    This enables the LKC to store who said each individual word, not just
    which speaker dominated a segment.

    The function is called from server.py only when both pyannote AND whisperx
    word-level timestamps are available; the existing `assign_speaker()` segment-
    level fallback is unchanged.

All Month 3/4 features (CONFIRMATION mode, pyannote diarization, summary
generation, context window) are retained unchanged.
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

from config import cfg

log = logging.getLogger(__name__)

# ── Month 4: pyannote.audio diarization (unchanged) ──────────────────────────
try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

_diarization_pipeline: Optional["PyannotePipeline"] = None
_diarization_pipeline_error: Optional[str] = None


def _get_diarization_pipeline() -> Optional["PyannotePipeline"]:
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
        use_auth = {"use_auth_token": hf_token} if hf_token else {}
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", **use_auth
        )
        device = "cuda" if (PYANNOTE_AVAILABLE and torch.cuda.is_available()) else "cpu"
        pipeline.to(torch.device(device))
        _diarization_pipeline = pipeline
        log.info(f"[diarization] pyannote pipeline loaded on {device}")
        return pipeline
    except Exception as exc:
        _diarization_pipeline_error = str(exc)
        log.warning(f"[diarization] pipeline load failed ({exc}) — using round-robin fallback")
        return None


# ── OpenAI-compatible local client ────────────────────────────────────────────
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


# ── Mode enum ─────────────────────────────────────────────────────────────────
class ConvMode(str, Enum):
    GREETING        = "greeting"
    MEETING_CAPTURE = "meeting_capture"
    QA              = "qa"
    AMBIENT         = "ambient"
    CONFIRMATION    = "confirmation"


# ── Question detection ────────────────────────────────────────────────────────
_QUESTION_RE = re.compile(
    r"\b(what|who|where|when|why|how|is|are|was|were|can|could|should|would|"
    r"did|do|does|tell me|explain|summarize|recap|define)\b",
    re.IGNORECASE,
)

def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_RE.search(text))


# ── Dialogue state ─────────────────────────────────────────────────────────────
@dataclass
class DialogueState:
    session_id:           str
    mode:                 ConvMode     = ConvMode.AMBIENT
    mode_entered_at:      float        = field(default_factory=time.time)
    greeted_speakers:     set[str]     = field(default_factory=set)
    _chat_history:        list         = field(default_factory=list, repr=False)
    transcript_context:   list[str]    = field(default_factory=list)
    CONTEXT_WINDOW:       int          = field(default_factory=lambda: cfg.dialogue.context_window)
    confirmation_pending: Optional[str] = None
    _speaker_counter:     int          = field(default=0, repr=False)

    def push_context(self, speaker: str, text: str) -> None:
        self.transcript_context.append(f"{speaker}: {text}")
        if len(self.transcript_context) > self.CONTEXT_WINDOW:
            self.transcript_context.pop(0)

    def context_block(self) -> str:
        return "\n".join(self.transcript_context)

    def get_chat_history(self) -> list:
        return self._chat_history


# ── Speaker labels ─────────────────────────────────────────────────────────────
_SPEAKER_LABELS = [f"Person {chr(65+i)}" for i in range(8)]
_SAMPLE_RATE    = 16000


def assign_speaker(
    state: DialogueState,
    audio_segment: bytes | np.ndarray | None = None,
) -> str:
    """Segment-level speaker assignment (unchanged from Month 4)."""
    pipeline = _get_diarization_pipeline()

    if pipeline is not None and audio_segment is not None:
        try:
            import torch
            if isinstance(audio_segment, bytes):
                wav = np.frombuffer(audio_segment, dtype=np.float32).copy()
            elif isinstance(audio_segment, np.ndarray):
                wav = audio_segment.astype(np.float32)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio_segment)}")

            min_samples = int(0.5 * _SAMPLE_RATE)
            if wav.size < min_samples:
                raise ValueError(f"Segment too short ({wav.size} samples)")

            waveform    = torch.from_numpy(wav).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": _SAMPLE_RATE}
            diarization = pipeline(audio_input)

            duration_per_speaker: dict[str, float] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration_per_speaker[speaker] = (
                    duration_per_speaker.get(speaker, 0.0) + turn.duration
                )

            if not duration_per_speaker:
                raise ValueError("No speakers detected by pyannote")

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
            log.warning(
                f"[diarization:{state.session_id}] inference failed ({exc})"
                " — falling back to round-robin"
            )

    label = _SPEAKER_LABELS[state._speaker_counter % len(_SPEAKER_LABELS)]
    state._speaker_counter += 1
    return label


# ── Month 5: Word-level speaker alignment ─────────────────────────────────────

def assign_speaker_words(
    state: DialogueState,
    word_timestamps: list[dict],
    audio_segment: np.ndarray,
) -> list[dict]:
    """
    Combine pyannote diarization with WhisperX word-level timestamps to
    produce per-word speaker attribution.

    For each word in word_timestamps (each has 'word', 'start', 'end', 'score'),
    we find which pyannote speaker turn contains the word's midpoint and
    annotate the word dict with a 'speaker' key using the session-stable label.

    Returns the enriched word list.  Falls back to assign_speaker() segment
    attribution (all words get the same speaker) when pyannote is unavailable
    or when word_timestamps is empty.

    Parameters
    ----------
    state           : DialogueState (carries _speaker_map for label stability)
    word_timestamps : list of {word, start, end, score} from WhisperX alignment
    audio_segment   : float32 np.ndarray at 16 kHz (same segment)
    """
    if not word_timestamps:
        return word_timestamps

    pipeline = _get_diarization_pipeline()
    if pipeline is None:
        # Segment-level fallback: every word gets the same dominant speaker
        fallback_speaker = assign_speaker(state, audio_segment)
        return [{**w, "speaker": fallback_speaker} for w in word_timestamps]

    try:
        import torch

        wav         = audio_segment.astype(np.float32)
        waveform    = torch.from_numpy(wav).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": _SAMPLE_RATE}
        diarization = pipeline(audio_input)

        # Build stable label map
        if not hasattr(state, "_speaker_map"):
            state._speaker_map: dict[str, str] = {}

        # Materialise diarization turns into a list for O(N*M) word-level lookup.
        # For typical short segments (< 30 s) the turn count is small (< 10).
        turns: list[tuple[float, float, str]] = []
        for turn, _, raw_label in diarization.itertracks(yield_label=True):
            if raw_label not in state._speaker_map:
                idx = len(state._speaker_map)
                state._speaker_map[raw_label] = (
                    _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS)
                    else f"Person {idx + 1}"
                )
            turns.append((turn.start, turn.end, state._speaker_map[raw_label]))

        def _speaker_at(t: float) -> str:
            """Return the speaker label whose turn contains time t."""
            for start, end, label in turns:
                if start <= t <= end:
                    return label
            # If t falls in a gap, pick the nearest turn
            if not turns:
                return assign_speaker(state, None)
            nearest = min(turns, key=lambda x: min(abs(x[0] - t), abs(x[1] - t)))
            return nearest[2]

        enriched: list[dict] = []
        for w in word_timestamps:
            midpoint = (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0
            enriched.append({**w, "speaker": _speaker_at(midpoint)})

        log.debug(
            f"[diarization:{state.session_id}] word-level alignment: "
            f"{len(enriched)} words, {len(turns)} turns, "
            f"{len(state._speaker_map)} speakers"
        )
        return enriched

    except Exception as exc:
        log.warning(
            f"[diarization:{state.session_id}] word-level alignment failed ({exc})"
            " — using segment-level fallback"
        )
        fallback_speaker = assign_speaker(state, audio_segment)
        return [{**w, "speaker": fallback_speaker} for w in word_timestamps]


# ── Confirmation heuristics ────────────────────────────────────────────────────
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


# ── Dialogue state registry ────────────────────────────────────────────────────
_dialogue_states: dict[str, DialogueState] = {}

def get_dialogue(session_id: str) -> DialogueState:
    if session_id not in _dialogue_states:
        _dialogue_states[session_id] = DialogueState(session_id=session_id)
    return _dialogue_states[session_id]

def clear_dialogue(session_id: str) -> None:
    _dialogue_states.pop(session_id, None)


# ── Mode transition FSM ────────────────────────────────────────────────────────

def update_mode(
    state: DialogueState,
    transcript: str,
    present_speakers: list[str],
    new_speakers: list[str],
    pending_confirmation: Optional[str] = None,
    *,
    summoned: bool = False,          # Month 5: explicit wake-word flag
) -> tuple[ConvMode, Optional[str]]:
    """
    Evaluate mode transitions.

    Month 5 change: QA mode now requires `summoned=True`.  Without an
    explicit wake-word, questions in the room are captured silently and the
    agent does NOT interrupt with a spoken reply.

    Returns (new_mode, entry_utterance | None).
    """
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

    # 0b. Enter CONFIRMATION when capture.py has a pending item
    if pending_confirmation and state.mode not in (ConvMode.GREETING, ConvMode.QA):
        state.confirmation_pending = pending_confirmation
        state.mode = ConvMode.CONFIRMATION
        state.mode_entered_at = time.time()
        return state.mode, pending_confirmation

    # 1. New speaker detected → GREETING (takes priority)
    if new_speakers:
        names = " and ".join(new_speakers)
        state.greeted_speakers.update(new_speakers)
        utterance = f"Hello {names}, welcome. Lab Brain is active and capturing this session."
        state.mode = ConvMode.MEETING_CAPTURE
        state.mode_entered_at = time.time()
        return state.mode, utterance

    # 2. Active transcript + summon detected + question → QA
    #    Month 5: guard requires `summoned=True`
    if transcript and summoned and _looks_like_question(transcript):
        state.mode = ConvMode.QA
        state.mode_entered_at = time.time()
        return state.mode, None

    # 2b. Summoned but NOT a question → still answer (direct command)
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


# ── Response generator ─────────────────────────────────────────────────────────

async def generate_response(
    state: DialogueState,
    transcript: str,
    lkc_context: str,
) -> Optional[str]:
    """
    Generate an agent reply.  Returns None for silent modes.

    Month 5: after returning a reply, the caller must call
    capture.clear_summon(session_id) to reset the wake-word flag.
    """
    if state.mode in (ConvMode.MEETING_CAPTURE, ConvMode.AMBIENT, ConvMode.CONFIRMATION):
        return None

    if state.mode in (ConvMode.QA, ConvMode.GREETING):
        if not LOCAL_LLM_AVAILABLE:
            return "[Local LLM not available — configure local_llm in config.json]"

        context_lines = state.context_block()
        lkc_section = (
            f"\n\nRelevant LKC knowledge:\n{lkc_context}"
            if lkc_context.strip() else ""
        )
        user_message = (
            f"Recent conversation:\n{context_lines}"
            f"{lkc_section}\n\n"
            f"The speaker directly addressed you and said: \"{transcript}\"\n"
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


# ── End-of-session summary ─────────────────────────────────────────────────────

async def generate_summary(
    state: DialogueState,
    session_tags: dict,
) -> str:
    if not LOCAL_LLM_AVAILABLE:
        return (
            "## Session Summary (stub)\n"
            "Local LLM unavailable — configure local_llm.\n"
        )

    transcript_excerpt = state.context_block()
    action_block   = "\n".join(f"- {a}" for a in session_tags.get("action_items", [])) or "None"
    decision_block = "\n".join(f"- {d}" for d in session_tags.get("decisions",    [])) or "None"
    deadline_block = "\n".join(f"- {d}" for d in session_tags.get("deadlines",    [])) or "None"
    entity_block   = ", ".join(session_tags.get("entities", [])) or "None"

    user_message = (
        f"You are Lab Brain summarising a research meeting.\n\n"
        f"Recent transcript (last {state.CONTEXT_WINDOW} turns):\n{transcript_excerpt}\n\n"
        f"Captured tags:\n"
        f"Action items:\n{action_block}\n\nDecisions:\n{decision_block}\n\n"
        f"Deadlines:\n{deadline_block}\n\nKey entities/people: {entity_block}\n\n"
        f"Produce a concise meeting summary in markdown with sections: "
        f"## Summary, ## Decisions, ## Action Items, ## Open Questions. "
        f"Keep each section to ≤4 bullet points."
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
