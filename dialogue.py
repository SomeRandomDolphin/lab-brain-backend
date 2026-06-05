"""
dialogue.py — Conversation Mode FSM + Agent Replies (Module 5, Month 2)

Manages the four conversation modes defined in the research plan:
  GREETING      → triggered when a new face is detected or session starts
  MEETING_CAPTURE → default active-session mode; logs everything to LKC
  QA            → triggered when a question is detected in the transcript
  AMBIENT       → low-activity background mode (no active speaker detected)

Each mode has:
  - an entry action (what the agent says when the mode is entered)
  - a transition rule (what switches us to another mode)
  - a response generator (how the agent replies to speech segments)

The agent's spoken replies are sent back to the browser as text; the browser
uses the Web Speech API (SpeechSynthesis) to vocalise them — no server-side
TTS dependency needed for Month 2 PoC.

LKC Retrieval Integration
--------------------------
In QA mode the agent queries lkc_retrieval.py to answer questions grounded
in prior session knowledge before falling back to Gemini.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)

# ── Gemini for dialogue ───────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    if cfg.gemini.available:
        genai.configure(api_key=cfg.gemini.api_key)
        _chat_model = genai.GenerativeModel(
            cfg.gemini.dialogue_model,
            system_instruction=(
                "You are Lab Brain, a helpful AI assistant embedded in a research "
                "laboratory meeting room. You are concise (≤2 sentences), professional, "
                "and always grounded in the lab context provided. Never hallucinate "
                "citations or project details."
            ),
        )
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False


# ── Mode enum ─────────────────────────────────────────────────────────────────
class ConvMode(str, Enum):
    GREETING        = "greeting"
    MEETING_CAPTURE = "meeting_capture"
    QA              = "qa"
    AMBIENT         = "ambient"


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
    # Gemini multi-turn chat handle (one per session)
    _chat: object = field(default=None, repr=False)
    # Rolling context window sent to Gemini (last N transcript lines)
    transcript_context: list[str] = field(default_factory=list)
    CONTEXT_WINDOW: int = field(default_factory=lambda: cfg.dialogue.context_window)

    def push_context(self, speaker: str, text: str) -> None:
        self.transcript_context.append(f"{speaker}: {text}")
        if len(self.transcript_context) > self.CONTEXT_WINDOW:
            self.transcript_context.pop(0)

    def context_block(self) -> str:
        return "\n".join(self.transcript_context)

    def get_chat(self):
        if self._chat is None and GEMINI_AVAILABLE:
            self._chat = _chat_model.start_chat(history=[])
        return self._chat


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
) -> tuple[ConvMode, Optional[str]]:
    """
    Evaluate mode transitions. Returns (new_mode, entry_utterance | None).
    entry_utterance is the text Lab Brain should speak on mode entry.
    """
    previous = state.mode
    utterance: Optional[str] = None

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
        if state.mode in (ConvMode.AMBIENT, ConvMode.QA):
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

    if state.mode in (ConvMode.QA, ConvMode.GREETING):
        if not GEMINI_AVAILABLE:
            return "[Gemini not configured — set api_key in config.json for agent replies]"

        # Build grounded prompt
        context_lines = state.context_block()
        lkc_section = (
            f"\n\nRelevant LKC knowledge:\n{lkc_context}"
            if lkc_context.strip()
            else ""
        )
        prompt = (
            f"Recent conversation:\n{context_lines}"
            f"{lkc_section}\n\n"
            f"The speaker just said: \"{transcript}\"\n"
            f"Respond as Lab Brain in ≤2 sentences."
        )

        loop = asyncio.get_event_loop()
        try:
            chat = state.get_chat()
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(prompt)
            )
            return response.text.strip()
        except Exception as exc:
            log.warning(f"[dialogue:{state.session_id}] Gemini error: {exc}")
            return None

    return None