"""
vision.py — VLM Perception Layer (Module 5, Month 2)

Receives JPEG frames from the browser camera WebSocket, sends them to
Gemini Flash for structured scene understanding, and returns:
  - present_speakers: list of detected person labels (face-ID placeholder)
  - engagement_cues:  per-speaker attention/engagement estimate
  - scene_summary:    one-sentence description for the LKC

Design notes
------------
* We use Gemini 2.0 Flash (gemini-2.0-flash) via the google-generativeai SDK.
  It accepts inline base64 image parts so no file upload is needed.
* Frame analysis runs at ~1 FPS (controlled by the caller); results are cached
  so the ASR path can read the latest perception state without waiting.
* Face recognition is deliberately kept as a label ("Person A/B/C") rather
  than biometric identification — privacy gating is a Month 3 concern.
* The Gemini API key is read from config.json (gemini.api_key).
  Without it, vision runs in stub mode — no export or env var needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)

# ── Gemini SDK ────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    if cfg.gemini.available:
        genai.configure(api_key=cfg.gemini.api_key)
        _vision_model = genai.GenerativeModel(cfg.gemini.vision_model)
        GEMINI_AVAILABLE = True
        log.info(f"Gemini vision model ready ({cfg.gemini.vision_model}).")
    else:
        GEMINI_AVAILABLE = False
        log.warning("Gemini API key not configured — vision stub mode active.")
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not installed — vision stub mode active.")

# ── Perception state shared across the session ───────────────────────────────
@dataclass
class PerceptionState:
    present_speakers: list[str] = field(default_factory=list)
    engagement_cues: dict[str, str] = field(default_factory=dict)
    scene_summary: str = ""
    last_updated: float = 0.0
    frame_count: int = 0
    error_count: int = 0

# One global state per session (keyed by session_id)
_states: dict[str, PerceptionState] = {}

def get_state(session_id: str) -> PerceptionState:
    if session_id not in _states:
        _states[session_id] = PerceptionState()
    return _states[session_id]

def clear_state(session_id: str) -> None:
    _states.pop(session_id, None)


# ── Prompt ────────────────────────────────────────────────────────────────────
_VISION_PROMPT = """
You are a meeting-room perception agent. Analyse this camera frame and respond
with ONLY a JSON object — no prose, no markdown fences — with exactly these keys:

{
  "present_speakers": ["Person A", "Person B"],   // list of visible people; use
                                                   // consistent labels across frames
  "engagement_cues": {                             // one of: focused | distracted |
    "Person A": "focused",                         //         away | unknown
    "Person B": "distracted"
  },
  "scene_summary": "Two people at a whiteboard, one writing."  // ≤15 words
}

Rules:
- If no people are visible, return empty lists/dict and summary "Empty room."
- Use the same person labels (Person A, B, …) consistently within a session.
- Do NOT include names, biometric data, or emotion diagnoses.
- Respond with valid JSON only.
""".strip()


# ── Core analysis function ────────────────────────────────────────────────────
async def analyse_frame(
    session_id: str,
    jpeg_bytes: bytes,
) -> PerceptionState:
    """
    Send a JPEG frame to Gemini Flash and update the session's PerceptionState.
    Returns the updated state immediately (caller does not need to await result
    separately — the state object is mutated in-place).
    """
    state = get_state(session_id)
    state.frame_count += 1

    if not GEMINI_AVAILABLE:
        # Stub: return a plausible fake so the rest of the system keeps working
        state.present_speakers = ["Person A"]
        state.engagement_cues = {"Person A": "focused"}
        state.scene_summary = "[Vision stub — set api_key in config.json to enable]"
        state.last_updated = time.time()
        return state

    b64 = base64.b64encode(jpeg_bytes).decode()
    image_part = {"mime_type": "image/jpeg", "data": b64}

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _vision_model.generate_content(
                [_VISION_PROMPT, image_part],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
        )
        raw = response.text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        state.present_speakers = parsed.get("present_speakers", [])
        state.engagement_cues  = parsed.get("engagement_cues", {})
        state.scene_summary    = parsed.get("scene_summary", "")
        state.last_updated     = time.time()
        log.debug(f"[vision:{session_id}] {state.scene_summary} | speakers={state.present_speakers}")

    except Exception as exc:
        state.error_count += 1
        log.warning(f"[vision:{session_id}] frame analysis failed: {exc}")

    return state