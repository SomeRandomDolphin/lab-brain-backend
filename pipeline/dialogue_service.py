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
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from core.config import cfg

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
        # pyannote.audio 4.x's Pipeline.from_pretrained() takes `token=`
        # (use_auth_token was renamed in the 4.x line, and the requirements.txt
        # pin was bumped to pyannote.audio>=4.0.0 for unrelated huggingface_hub
        # version-conflict reasons — see the comment there).
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

# Dedicated executor for dialogue LLM network calls (generate_response/qa,
# summarize). Same rationale as vision's _vision_executor: these calls used
# run_in_executor(None, ...) — the shared default executor also used by
# diarization and (via vision.py/livekit_rooms.py) ASR-adjacent work. A
# single QA call can legitimately hold a thread for 5-10s+ with a thinking
# model; keeping that off the shared pool means it can't stall diarization
# or frame draining while it's in flight.
from concurrent.futures import ThreadPoolExecutor
_dialogue_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dialogue-llm")

_SYSTEM_PROMPT = (
    "You are Lab Brain, a helpful AI assistant embedded in a research "
    "laboratory meeting room. You are concise (≤2 sentences), professional, "
    "and always grounded in the lab context provided. Never hallucinate "
    "citations or project details. You only respond when directly addressed."
)


def warmup() -> None:
    """
    Send a throwaway completion to the local LLM at startup so Ollama loads
    the model into memory once, at boot, instead of on whoever's first real
    summon.

    Without this, the FIRST chat.completions.create() call of the server's
    life is whatever the first user question happens to be — and that call
    pays the full cost of Ollama reading the model off disk into RAM/VRAM
    before it can generate a single token. On this setup that showed up as
    a 259-second wait for one QA reply, versus ~2s for every call after the
    model was already warm (see the 10:26:42 → 10:31:01 vs. 10:32:43 →
    10:32:45 gap in the logs).

    This is a blocking network call — run it via run_in_executor from the
    FastAPI startup hook, the same way lkc_retrieval.warmup() is called:

        await loop.run_in_executor(None, dialogue_service.warmup)

    Also set OLLAMA_KEEP_ALIVE on the Ollama container to something longer
    than its 5-minute default (see start_services.sh) — otherwise the model
    gets evicted during any quiet stretch of a meeting and the next summon
    pays the cold-start cost all over again, warmup or not.

    NOTE: log the wall time this itself takes. In the 2026-07-14 log this
    call spans ~11:00:48 → 11:03:51 (~3 min) — i.e. warmup() is *absorbing*
    a cold load, not skipping one. That's expected once, at boot. If you
    ever see that multi-minute span recur on a *later* warmup (e.g. after a
    restart triggered mid-session), it means keep_alive lapsed or the
    container was recreated — not a "the model got slower" problem.
    """
    if not LOCAL_LLM_AVAILABLE:
        log.warning("[dialogue] warmup: local LLM client not available — skipping.")
        return
    t0 = time.perf_counter()
    try:
        _dialogue_client.chat.completions.create(
            model=cfg.local_llm.dialogue_model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )
        log.info(
            f"[dialogue] local LLM warmed: {cfg.local_llm.dialogue_model} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    except Exception as exc:
        log.warning(f"[dialogue] warmup failed (non-fatal, first real call will be slow): {exc}")


# Tiny 1x1 transparent PNG, just enough to make Ollama load the vision
# projector/encoder weights — content is irrelevant, this is a throwaway.
_WARMUP_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def warmup_vision() -> None:
    """
    Same rationale as warmup(), for the vision model.

    Ollama keeps every model's weights in its own independent memory slot —
    warming qwen3-vl:4b does NOT also warm the vision model (e.g. qwen3-vl:4b
    from config.json). They're separate `ollama pull`s and separate
    load-into-VRAM events. Whatever code path first sends a frame to vision
    (screenshot description, slide capture, etc.) will otherwise eat the
    exact same multi-minute cold-load penalty warmup() was written to avoid
    for chat — just later, and probably mid-meeting instead of at boot.

    Call this from the same FastAPI startup hook as warmup(), as its own
    run_in_executor call so one failing doesn't block the other:

        await loop.run_in_executor(None, dialogue_service.warmup)
        await loop.run_in_executor(None, dialogue_service.warmup_vision)

    Caveat: OLLAMA_KEEP_ALIVE (see start_services.sh) is a container-wide
    default, so it applies to this model too — but keep_alive only stops
    *time-based* eviction. If VRAM/RAM is too small to hold both the
    dialogue model and the vision model resident at once, loading one can
    still force Ollama to evict the other regardless of keep_alive, and
    you'll pay a cold-load on whichever one got pushed out next time it's
    summoned. `docker exec lab-brain-ollama ollama ps` shows what's
    currently resident if replies start getting slow again after this.
    """
    if not LOCAL_LLM_AVAILABLE:
        log.warning("[dialogue] vision warmup: local LLM client not available — skipping.")
        return
    t0 = time.perf_counter()
    try:
        _dialogue_client.chat.completions.create(
            model=cfg.local_llm.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_WARMUP_IMAGE_B64}"},
                    },
                ],
            }],
            max_tokens=1,
        )
        log.info(
            f"[dialogue] local vision model warmed: {cfg.local_llm.vision_model} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    except Exception as exc:
        log.warning(f"[dialogue] vision warmup failed (non-fatal, first real call will be slow): {exc}")


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

# How long after a QA reply a follow-up question is still routed to QA
# without needing to re-summon (wake word / button click) first. See rule
# 2b in update_mode() and the re-arming in session_pipeline.py's
# _handle_qa_sse.
QA_FOLLOW_UP_WINDOW_SECONDS = 15.0


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
    _speaker_map:         dict          = field(default_factory=dict, repr=False)
    # Timestamp (time.time()) until which a non-summoned utterance that
    # looks like a question should still be routed to QA. 0 = no active
    # follow-up window. Re-armed by session_pipeline.py after each QA
    # reply; consumed (reset to 0) the moment it's used or the session is
    # manually dismissed. See QA_FOLLOW_UP_WINDOW_SECONDS.
    qa_follow_up_until:   float         = 0.0

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
        state.qa_follow_up_until = 0.0  # consumed; re-armed after the reply
        return state.mode, None

    # 2b. Follow-up window → QA, without needing the wake word again.
    # Only for something that reads like an actual question/command --
    # ordinary conversation during this window should still fall through
    # to rule 3 as normal meeting capture, so two people continuing to
    # chat right after a reply doesn't get silently routed to the agent.
    if (
        transcript
        and not summoned
        and state.qa_follow_up_until
        and time.time() < state.qa_follow_up_until
        and _looks_like_question(transcript)
    ):
        state.mode = ConvMode.QA
        state.mode_entered_at = time.time()
        state.qa_follow_up_until = 0.0  # consumed; re-armed after the reply
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


def force_exit_qa(session_id: str) -> Optional[ConvMode]:
    """
    Manually force a session out of QA mode. This is the escape hatch behind
    the summon button's "dismiss" action.

    update_mode()'s FSM (and _handle_qa_sse's automatic revert in
    session_pipeline.py) both bring QA back to MEETING_CAPTURE/AMBIENT on
    their own -- but only once a segment or an LLM reply actually completes.
    If the LLM call is still in flight, hung, or a segment never arrives,
    there was previously no way for the user to get back to a listening
    state at all. This lets the "dismiss" click bypass all of that and reset
    the mode immediately, regardless of what the pipeline is doing.

    Returns the new mode, or None if there's no active dialogue state for
    this session_id (e.g. it was never summoned, or the session already
    ended).
    """
    state = _dialogue_states.get(session_id)
    if state is None:
        return None
    if state.mode == ConvMode.QA:
        state.mode = ConvMode.MEETING_CAPTURE
        state.mode_entered_at = time.time()
        # A manual dismiss means "I'm done talking to the agent" -- don't
        # leave the follow-up window armed, or the very next sentence in
        # the room could get silently routed back to QA.
        state.qa_follow_up_until = 0.0
        # Import here (not at module top) since livekit_rooms.py is one
        # layer up the pipeline stack and doesn't otherwise need to be a
        # dependency of dialogue_service.py just for this one broadcast.
        from pipeline.livekit_rooms import broadcast
        broadcast(session_id, {"type": "mode_change", "mode": state.mode.value})
    return state.mode


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
            f"/no_think\n"
            f"Recent conversation:\n{state.context_block()}"
            f"{lkc_section}\n\n"
            f'The speaker directly addressed you and said: "{transcript}"\n'
            f"Respond as Lab Brain in ≤2 sentences."
        )

        history = state.get_chat_history()
        history.append({"role": "user", "content": user_message})

        # Explicit dispatch/completion markers with a short call id. The
        # httpx "HTTP Request: POST ... 200 OK" line only tells you a
        # completion happened, not which code path issued it or whether the
        # OpenAI SDK's built-in retry (max_retries=2 by default) fired
        # underneath — which is exactly the ambiguity that made an earlier
        # cluster of completions around a summary call impossible to
        # attribute after the fact. This makes call count/origin explicit
        # going forward instead of having to infer it from timestamps.
        call_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_event_loop()
        log.info(f"[dialogue:{state.session_id}] LLM dispatch qa call_id={call_id}")
        t0 = time.perf_counter()
        try:
            response = await loop.run_in_executor(
                _dialogue_executor,
                lambda: _dialogue_client.chat.completions.create(
                    model=cfg.local_llm.dialogue_model,
                    messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + history,
                    # 120 was sized for the visible ≤2-sentence answer only.
                    # Qwen3 thinks by default and was silently spending the
                    # entire budget on the <think> block, leaving content=""
                    # with no error anywhere (200 OK, clean elapsed time) —
                    # the same failure mode vision.py had. Give it real
                    # headroom for thinking + the short answer.
                    max_tokens=4096,
                    temperature=0.4,
                    timeout=30,  # fail fast instead of silently retrying/hanging
                )
            )
            # elapsed is the actual network+inference time for THIS call. If
            # elapsed is small (a few seconds) but the gap between this log
            # line's timestamp and the dispatch line above is large, the
            # model/network wasn't the bottleneck — something else was
            # hogging the event loop thread (or the default executor's
            # thread pool) and delayed this coroutine from resuming. That's
            # the distinction "warmed but still slow" needs to separate.
            elapsed = time.perf_counter() - t0
            finish_reason = response.choices[0].finish_reason
            log.info(
                f"[dialogue:{state.session_id}] LLM complete qa call_id={call_id} "
                f"({elapsed:.1f}s) finish_reason={finish_reason}"
            )
            raw = response.choices[0].message.content.strip()
            # Defensive strip in case /no_think doesn't fully suppress
            # thinking on this template/version — don't let a leaked
            # <think> block get spoken or broadcast to the user.
            reply = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not reply:
                # This used to fail silently: `if reply:` in
                # _handle_qa_sse skips broadcast/speak/persist on an empty
                # string with no error anywhere, which is exactly what made
                # this bug invisible — clean 200 OK, clean elapsed time,
                # user just never got an answer. Surface it loudly now.
                log.warning(
                    f"[dialogue:{state.session_id}] qa call_id={call_id} returned EMPTY "
                    f"content (finish_reason={finish_reason}, raw={raw[:200]!r}) — "
                    f"model likely exhausted max_tokens on thinking; returning fallback"
                )
                return "Sorry, I'm having trouble forming a reply right now — could you ask again?"
            history.append({"role": "assistant", "content": reply})
            max_turns = state.CONTEXT_WINDOW * 2
            if len(history) > max_turns:
                state._chat_history = history[-max_turns:]
            return reply
        except Exception as exc:
            log.warning(f"[dialogue:{state.session_id}] LLM error call_id={call_id}: {exc}")
            return None

    return None


# ── Summary generator ─────────────────────────────────────────────────────────

async def generate_summary(session_id: str, transcript_text: str, session_tags: dict) -> str:
    """
    session_id/transcript_text replace the old `state: DialogueState` param.
    Summary generation used to read state.context_block() — the in-memory
    DialogueState's own transcript_context list — but DELETE /livekit/room
    calls clear_dialogue(session_id) as part of its teardown, which pops
    that DialogueState out of _dialogue_states entirely. If the frontend
    then calls POST /summary/{session_id} (a completely normal teardown
    order — end the room, then fetch the recap), get_dialogue() silently
    handed back a brand-new, empty DialogueState instead of raising, so
    this function saw "0 words" and returned the "not enough was captured"
    stub even when a real conversation had just happened and was already
    durably persisted via Supabase/the LKC graph. The caller (sessions.py's
    post_summary) now builds transcript_text from those durable records
    instead, so this function no longer depends on in-memory state that
    another endpoint may have already torn down by the time it runs.
    """
    if not LOCAL_LLM_AVAILABLE:
        return "## Session Summary (stub)\nLocal LLM unavailable — configure local_llm.\n"

    context = transcript_text

    # Hallucination guard. The prompt below demands 4 populated markdown
    # sections no matter what — that's fine for a real meeting, but for a
    # session with only a couple of test utterances (e.g. "This is a test
    # from front end to back end.") there is nothing to summarize, and a
    # small local model (qwen3:4b) will not say "not enough content" on
    # its own. It will pattern-match "research laboratory meeting" from the
    # system prompt and invent a plausible-sounding agenda — fake findings,
    # a fake grant request, a fake conference — to fill the sections it was
    # told to produce. That's exactly what happened in the 2026-07-14 run.
    # Short-circuit before the LLM call entirely rather than trying to
    # prompt our way out of it once the transcript is already too thin.
    MIN_SUMMARY_WORDS = 25
    if len(context.split()) < MIN_SUMMARY_WORDS:
        log.info(
            f"[dialogue:{session_id}] summary skipped — transcript too short "
            f"({len(context.split())} words < {MIN_SUMMARY_WORDS}); returning stub instead of risking hallucination"
        )
        return (
            "## Summary\n"
            "Not enough conversation was captured in this session to generate a meaningful summary.\n\n"
            "## Decisions\nNone\n\n## Action Items\nNone\n\n## Open Questions\nNone\n"
        )

    action_block   = "\n".join(f"- {a}" for a in session_tags.get("action_items", [])) or "None"
    decision_block = "\n".join(f"- {d}" for d in session_tags.get("decisions",    [])) or "None"
    deadline_block = "\n".join(f"- {d}" for d in session_tags.get("deadlines",    [])) or "None"
    entity_block   = ", ".join(session_tags.get("entities", [])) or "None"

    user_message = (
        f"/no_think\n"
        f"You are Lab Brain summarising a research meeting.\n\n"
        f"The transcript below is the ONLY source of truth. Do not invent, "
        f"assume, or infer any finding, decision, action item, deadline, or "
        f"topic that is not explicitly present in it — even if it would be "
        f"plausible for a research lab meeting. If a section has no "
        f"supporting content in the transcript, write exactly 'None' for "
        f"that section instead of fabricating something to fill it.\n\n"
        f"Session transcript:\n{context}\n\n"
        f"Action items:\n{action_block}\n\nDecisions:\n{decision_block}\n\n"
        f"Deadlines:\n{deadline_block}\n\nKey entities: {entity_block}\n\n"
        f"Produce a concise meeting summary in markdown: "
        f"## Summary, ## Decisions, ## Action Items, ## Open Questions. "
        f"≤4 bullet points per section, each bullet traceable to the transcript above."
    )

    call_id = uuid.uuid4().hex[:8]
    loop = asyncio.get_event_loop()
    log.info(f"[dialogue:{session_id}] LLM dispatch summary call_id={call_id}")
    t0 = time.perf_counter()
    try:
        response = await loop.run_in_executor(
            _dialogue_executor,
            lambda: _dialogue_client.chat.completions.create(
                model=cfg.local_llm.dialogue_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=16384,  # more thinking-budget headroom compared to generate_response()
                temperature=0.1,  # was 0.3 — summarization should be low-creativity
                timeout=30,
            )
        )
        elapsed = time.perf_counter() - t0
        finish_reason = response.choices[0].finish_reason
        log.info(
            f"[dialogue:{session_id}] LLM complete summary call_id={call_id} "
            f"({elapsed:.1f}s) finish_reason={finish_reason}"
        )
        raw = response.choices[0].message.content.strip()
        summary = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not summary:
            log.warning(
                f"[dialogue:{session_id}] summary call_id={call_id} returned EMPTY "
                f"content (finish_reason={finish_reason}) — falling back to tag-based summary"
            )
            raise ValueError("empty summary content")
        return summary
    except Exception as exc:
        log.warning(f"[dialogue:{session_id}] summary failed call_id={call_id}: {exc}")
        return (
            f"## Session Summary (fallback)\n"
            f"**Action items:**\n{action_block}\n\n"
            f"**Decisions:**\n{decision_block}\n\n"
            f"**Deadlines:**\n{deadline_block}\n"
        )