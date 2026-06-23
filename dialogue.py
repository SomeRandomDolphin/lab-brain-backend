"""
dialogue.py — Conversation Mode FSM + Agent Replies (Module 5, Month 4)

Manages the four conversation modes defined in the research plan:
  GREETING      → triggered when a new face is detected or session starts
  MEETING_CAPTURE → default active-session mode; logs everything to LKC
  QA            → triggered when a question is detected in the transcript
  AMBIENT       → low-activity background mode (no active speaker detected)

Month 3 additions
-----------------
1.  CONFIRMATION mode — agent seeks speaker acknowledgement for a
    captured action item or decision detected by capture.py.
    Activated when capture.py queues a confirmation text; exits back to
    MEETING_CAPTURE when the speaker says "yes / correct / right" or
    dismisses with "no / never mind / skip".

2.  Diarization upgrade stub — the round-robin speaker assignment from
    Month 2 is wrapped in `assign_speaker()`.  Replace the body with a
    real pyannote / WhisperX call once the dependency is added.

3.  End-of-session summary — `generate_summary()` calls the local LLM
    with the full transcript context and tag list to produce a structured
    meeting summary (decisions, action items, open questions).

Month 4 additions
-----------------
1.  Real pyannote.audio >= 3.3 speaker diarization — `assign_speaker()`
    now runs `pyannote/speaker-diarization-3.1` on the audio segment.
    Falls back to round-robin if pyannote is unavailable (no HF token,
    no GPU, etc.) so the server stays runnable in any environment.
    Pipeline is loaded once per process via `_get_diarization_pipeline()`.

LKC Retrieval Integration
--------------------------
In QA mode the agent queries lkc_retrieval.py to answer questions grounded
in prior session knowledge before falling back to the local LLM.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from config import cfg

# ── Month 4: pyannote.audio diarization ──────────────────────────────────────
try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Lazy-loaded singleton — initialised on first call to assign_speaker().
# This avoids blocking server startup if the HF model download takes time.
_diarization_pipeline: Optional["PyannotePipeline"] = None
_diarization_pipeline_error: Optional[str] = None  # cached failure reason

def _get_diarization_pipeline() -> Optional["PyannotePipeline"]:
    """
    Load pyannote/speaker-diarization-3.1 exactly once per process.

    Requires:
      • pyannote.audio >= 3.3.2  (see requirements.txt)
      • A Hugging Face token that has accepted the model terms:
          huggingface-cli login
        OR set `local_llm.hf_token` in config.json,
        OR set the HF_TOKEN / HUGGINGFACE_TOKEN environment variable.
      • torch (CPU or CUDA — GPU recommended for real-time use)

    Returns None on any initialisation failure so the server degrades
    gracefully to the round-robin fallback.
    """
    global _diarization_pipeline, _diarization_pipeline_error

    if _diarization_pipeline is not None:
        return _diarization_pipeline
    if _diarization_pipeline_error is not None:
        return None   # already failed; don't retry every call
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
            "pyannote/speaker-diarization-3.1",
            **use_auth,
        )
        device = "cuda" if (PYANNOTE_AVAILABLE and torch.cuda.is_available()) else "cpu"
        pipeline.to(torch.device(device))
        _diarization_pipeline = pipeline
        log.info(f"[diarization] pyannote pipeline loaded on {device}")
        return pipeline
    except Exception as exc:
        _diarization_pipeline_error = str(exc)
        log.warning(f"[diarization] pipeline load failed ({exc}) — using round-robin fallback")

log = logging.getLogger(__name__)

# ── OpenAI-compatible local client for dialogue ───────────────────────────────
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
    "citations or project details."
)


# ── Mode enum ─────────────────────────────────────────────────────────────────
class ConvMode(str, Enum):
    GREETING        = "greeting"
    MEETING_CAPTURE = "meeting_capture"
    QA              = "qa"
    AMBIENT         = "ambient"
    CONFIRMATION    = "confirmation"   # Month 3: agent awaits speaker ack


# ── Question detection heuristic ─────────────────────────────────────────────
_QUESTION_RE = re.compile(
    r"\b(what|who|where|when|why|how|is|are|was|were|can|could|should|would|"
    r"did|do|does|tell me|explain|summarize|recap|define)\b",
    re.IGNORECASE,
)

def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_RE.search(text))


# ── Dialogue state per session ────────────────────────────────────────────────
@dataclass
class DialogueState:
    session_id: str
    mode: ConvMode = ConvMode.AMBIENT
    mode_entered_at: float = field(default_factory=time.time)
    greeted_speakers: set[str] = field(default_factory=set)
    # Rolling message history for the local LLM (list of {role, content} dicts)
    _chat_history: list = field(default_factory=list, repr=False)
    # Rolling context window sent to the LLM (last N transcript lines)
    transcript_context: list[str] = field(default_factory=list)
    CONTEXT_WINDOW: int = field(default_factory=lambda: cfg.dialogue.context_window)
    # Month 3: pending confirmation text queued by capture.py
    confirmation_pending: Optional[str] = None
    # Month 3: round-robin speaker counter (placeholder until pyannote)
    _speaker_counter: int = field(default=0, repr=False)

    def push_context(self, speaker: str, text: str) -> None:
        self.transcript_context.append(f"{speaker}: {text}")
        if len(self.transcript_context) > self.CONTEXT_WINDOW:
            self.transcript_context.pop(0)

    def context_block(self) -> str:
        return "\n".join(self.transcript_context)

    def get_chat_history(self) -> list:
        return self._chat_history


# ── Month 4: Speaker diarization via pyannote.audio 3.3 ─────────────────────
_SPEAKER_LABELS = [f"Person {chr(65+i)}" for i in range(8)]  # A-H fallback
_SAMPLE_RATE    = 16000   # pyannote expects 16 kHz mono float32

def assign_speaker(
    state: DialogueState,
    audio_segment: bytes | np.ndarray | None = None,
) -> str:
    """
    Return the dominant speaker label for an audio segment.

    Month 4: real pyannote/speaker-diarization-3.1 inference.

    Steps
    -----
    1. Convert raw bytes or np.ndarray to a torch tensor shaped (1, T).
    2. Run the pyannote pipeline on the in-memory waveform dict.
    3. Find the speaker who occupies the most total time in the annotation.
    4. Map the raw pyannote label (e.g. "SPEAKER_01") to a stable
       session-scoped human-readable label ("Person A", "Person B" …)
       so labels are consistent across segments within one session.

    Fallback
    --------
    If pyannote is unavailable, the pipeline failed to load, the segment
    is too short (< 0.5 s), or inference raises, we fall back to
    round-robin "Person X" assignment and log a warning.

    Parameters
    ----------
    state         : DialogueState for this session (carries speaker counter
                    and the new `_speaker_map` dict for label stability).
    audio_segment : Raw audio — either float32 bytes from the WebSocket
                    or an np.ndarray[float32] already decoded.
                    Pass None to force the round-robin fallback.
    """
    pipeline = _get_diarization_pipeline()

    if pipeline is not None and audio_segment is not None:
        try:
            import torch

            # --- Convert to np.ndarray float32 ---
            if isinstance(audio_segment, bytes):
                wav = np.frombuffer(audio_segment, dtype=np.float32).copy()
            elif isinstance(audio_segment, np.ndarray):
                wav = audio_segment.astype(np.float32)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio_segment)}")

            # Require at least 0.5 s to get a meaningful diarization
            min_samples = int(0.5 * _SAMPLE_RATE)
            if wav.size < min_samples:
                raise ValueError(f"Segment too short ({wav.size} samples)")

            # pyannote expects a dict with "waveform" (C, T) tensor and "sample_rate"
            waveform = torch.from_numpy(wav).unsqueeze(0)   # shape: (1, T)
            audio_input = {"waveform": waveform, "sample_rate": _SAMPLE_RATE}

            diarization = pipeline(audio_input)

            # Find the speaker with the most total speaking time
            duration_per_speaker: dict[str, float] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration_per_speaker[speaker] = (
                    duration_per_speaker.get(speaker, 0.0) + turn.duration
                )

            if not duration_per_speaker:
                raise ValueError("No speakers detected by pyannote")

            dominant_raw = max(duration_per_speaker, key=duration_per_speaker.get)

            # Map pyannote's raw label to a session-stable human label
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
                f"[diarization:{state.session_id}] pyannote inference failed ({exc})"
                " — falling back to round-robin"
            )

    # Round-robin fallback (no pyannote, short segment, or inference error)
    label = _SPEAKER_LABELS[state._speaker_counter % len(_SPEAKER_LABELS)]
    state._speaker_counter += 1
    return label


# ── Month 3: Confirmation heuristics ─────────────────────────────────────────
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


_dialogue_states: dict[str, DialogueState] = {}

def get_dialogue(session_id: str) -> DialogueState:
    if session_id not in _dialogue_states:
        _dialogue_states[session_id] = DialogueState(session_id=session_id)
    return _dialogue_states[session_id]

def clear_dialogue(session_id: str) -> None:
    _dialogue_states.pop(session_id, None)


# ── Mode transition logic ─────────────────────────────────────────────────────
def update_mode(
    state: DialogueState,
    transcript: str,
    present_speakers: list[str],
    new_speakers: list[str],
    pending_confirmation: Optional[str] = None,
) -> tuple[ConvMode, Optional[str]]:
    """
    Evaluate mode transitions. Returns (new_mode, entry_utterance | None).
    entry_utterance is the text Lab Brain should speak on mode entry.

    Month 3: handles CONFIRMATION mode transitions driven by capture.py tags.
    """
    utterance: Optional[str] = None

    # 0. Month 3: If we are IN confirmation mode, resolve it first
    if state.mode == ConvMode.CONFIRMATION and transcript:
        if _is_affirmation(transcript):
            log.info(f"[dialogue:{state.session_id}] Confirmation accepted.")
            state.confirmation_pending = None
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
            return state.mode, "Got it, I've logged that."
        elif _is_denial(transcript):
            log.info(f"[dialogue:{state.session_id}] Confirmation denied — discarding tag.")
            state.confirmation_pending = None
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
            return state.mode, "Understood, I'll discard that."

    # 0b. Month 3: Enter CONFIRMATION mode when capture.py has a pending item
    if pending_confirmation and state.mode not in (ConvMode.GREETING, ConvMode.QA):
        state.confirmation_pending = pending_confirmation
        state.mode = ConvMode.CONFIRMATION
        state.mode_entered_at = time.time()
        return state.mode, pending_confirmation

    # 1. New speaker detected → GREETING (takes priority)
    if new_speakers:
        names = " and ".join(new_speakers)
        state.mode = ConvMode.GREETING
        state.mode_entered_at = time.time()
        utterance = f"Hello {names}, welcome. Lab Brain is active and capturing this session."
        state.greeted_speakers.update(new_speakers)
        # Immediately transition to MEETING_CAPTURE after greeting
        state.mode = ConvMode.MEETING_CAPTURE
        return state.mode, utterance

    # 2. Active transcript and question detected → QA
    if transcript and _looks_like_question(transcript):
        state.mode = ConvMode.QA
        state.mode_entered_at = time.time()
        return state.mode, None

    # 3. Active transcript (non-question) → MEETING_CAPTURE
    if transcript and len(transcript.split()) >= 3:
        if state.mode in (ConvMode.AMBIENT, ConvMode.QA, ConvMode.CONFIRMATION):
            state.mode = ConvMode.MEETING_CAPTURE
            state.mode_entered_at = time.time()
        return state.mode, None

    # 4. No speakers present → AMBIENT
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
    """
    Generate an agent reply for the current mode.
    Returns None when the mode does not require a spoken reply (MEETING_CAPTURE,
    AMBIENT) so the ASR path stays quiet.
    """
    if state.mode == ConvMode.MEETING_CAPTURE:
        # Silently capture — no spoken reply needed
        return None

    if state.mode == ConvMode.AMBIENT:
        return None

    # Month 3: CONFIRMATION mode — the utterance was already set by update_mode;
    # generate_response should return None so the TTS path uses the mode's entry text.
    if state.mode == ConvMode.CONFIRMATION:
        return None

    if state.mode in (ConvMode.QA, ConvMode.GREETING):
        if not LOCAL_LLM_AVAILABLE:
            return "[Local LLM not available — install the openai package and configure local_llm in config.json]"

        # Build grounded prompt
        context_lines = state.context_block()
        lkc_section = (
            f"\n\nRelevant LKC knowledge:\n{lkc_context}"
            if lkc_context.strip()
            else ""
        )
        user_message = (
            f"Recent conversation:\n{context_lines}"
            f"{lkc_section}\n\n"
            f"The speaker just said: \"{transcript}\"\n"
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
            # Keep assistant turn in history for multi-turn context
            history.append({"role": "assistant", "content": reply})
            # Trim history to avoid unbounded growth (keep last 2*CONTEXT_WINDOW turns)
            max_turns = state.CONTEXT_WINDOW * 2
            if len(history) > max_turns:
                state._chat_history = history[-max_turns:]
            return reply
        except Exception as exc:
            log.warning(f"[dialogue:{state.session_id}] Local LLM error: {exc}")
            return None

    return None

# ── Month 3: End-of-session summary ──────────────────────────────────────────
async def generate_summary(
    state: DialogueState,
    session_tags: dict,
) -> str:
    """
    Generate a structured end-of-meeting summary using the local LLM.

    session_tags is the dict returned by capture.get_session_tags() and should
    contain keys: action_items, decisions, deadlines, entities.

    Returns a markdown-formatted summary string, or a plain fallback if the
    LLM is unavailable.
    """
    if not LOCAL_LLM_AVAILABLE:
        return (
            "## Session Summary (stub)\n"
            "Local LLM unavailable — install the openai package and configure local_llm.\n"
        )

    transcript_excerpt = state.context_block()
    action_block  = "\n".join(f"- {a}" for a in session_tags.get("action_items", [])) or "None detected"
    decision_block = "\n".join(f"- {d}" for d in session_tags.get("decisions", [])) or "None detected"
    deadline_block = "\n".join(f"- {d}" for d in session_tags.get("deadlines", [])) or "None detected"
    entity_block   = ", ".join(session_tags.get("entities", [])) or "None detected"

    user_message = (
        f"You are Lab Brain summarising a research meeting.\n\n"
        f"Recent transcript (last {state.CONTEXT_WINDOW} turns):\n{transcript_excerpt}\n\n"
        f"Captured tags:\n"
        f"Action items:\n{action_block}\n\n"
        f"Decisions:\n{decision_block}\n\n"
        f"Deadlines:\n{deadline_block}\n\n"
        f"Key entities/people: {entity_block}\n\n"
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
        log.warning(f"[dialogue:{state.session_id}] summary generation failed: {exc}")
        return (
            f"## Session Summary (fallback)\n"
            f"**Action items:**\n{action_block}\n\n"
            f"**Decisions:**\n{decision_block}\n\n"
            f"**Deadlines:**\n{deadline_block}\n"
        )