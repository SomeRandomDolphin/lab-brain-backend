"""
vision.py — VLM Perception Layer (Module 5, Month 2)

Receives JPEG frames from the browser camera WebSocket, sends them to
a locally hosted vision model via an OpenAI-compatible API, and returns:
  - present_speakers: list of detected person labels (face-ID placeholder)
  - engagement_cues:  per-speaker attention/engagement estimate
  - scene_summary:    one-sentence description for the LKC

Design notes
------------
* Uses the openai Python package pointed at a local server (e.g. Ollama,
  LM Studio, llama.cpp server, vLLM). Configure base_url and vision_model
  in config.json under the "local_llm" section.
* Frame analysis runs at ~1 FPS (controlled by the caller); results are cached
  so the ASR path can read the latest perception state without waiting.
* Face recognition is deliberately kept as a label ("Person A/B/C") rather
  than biometric identification — privacy gating is a Month 3 concern.
* Without a reachable local server, vision runs in stub mode automatically.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)

# ── OpenAI-compatible local client ────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI(
        base_url=cfg.local_llm.base_url,
        api_key=cfg.local_llm.api_key,
    )
    LOCAL_LLM_AVAILABLE = True
    log.info(f"Local vision model ready ({cfg.local_llm.vision_model} @ {cfg.local_llm.base_url}).")
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    log.warning("openai package not installed — vision stub mode active.")
except Exception as exc:
    LOCAL_LLM_AVAILABLE = False
    log.warning(f"Could not initialise local LLM client: {exc} — vision stub mode active.")

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

# Backwards-compatible alias used by server.py
GEMINI_AVAILABLE = LOCAL_LLM_AVAILABLE


# ── JSON extraction (robust, handles common LLM formatting quirks) ────────────
def _extract_json(session_id: str, raw: str) -> dict:
    """
    Try increasingly lenient strategies to pull a valid JSON object out of
    whatever the vision model returned. Strategies in order:

    1. Direct parse — model followed instructions perfectly.
    2. Strip markdown fences (```json ... ``` or ``` ... ```).
    3. Regex: grab the first {...} block anywhere in the text.
    4. Fallback: return a safe empty-room default and log the raw text.
    """
    # Strategy 1 — direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 — extract first {...} block (handles preamble/postamble prose)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4 — give up gracefully
    log.warning(
        f"[vision:{session_id}] could not parse model output as JSON; "
        f"raw={raw[:200]!r}"
    )
    return {
        "present_speakers": [],
        "engagement_cues": {},
        "scene_summary": "Unknown — could not parse model response.",
    }


# ── Prompt ────────────────────────────────────────────────────────────────────
_VISION_PROMPT = """
Analyse this camera frame. Reply with ONLY a raw JSON object — no markdown, no code fences, no comments, no trailing commas, no explanation.

Example of the exact format required:
{"present_speakers":["Person A"],"engagement_cues":{"Person A":"focused"},"scene_summary":"One person at a desk looking at a monitor."}

Rules:
- present_speakers: array of visible person labels. Use "Person A", "Person B", etc. Empty array if nobody visible.
- engagement_cues: object mapping each label to one of: focused | distracted | away | unknown
- scene_summary: one plain sentence, 15 words max. Use "Empty room." if nobody is present.
- No names, no biometric data, no emotion diagnoses.
- Output ONLY the JSON. Nothing before or after it.
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

    if not LOCAL_LLM_AVAILABLE:
        # Stub: return a plausible fake so the rest of the system keeps working
        state.present_speakers = ["Person A"]
        state.engagement_cues = {"Person A": "focused"}
        state.scene_summary = "[Vision stub — install openai package and configure local_llm in config.json]"
        state.last_updated = time.time()
        return state

    b64 = base64.b64encode(jpeg_bytes).decode()

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _client.chat.completions.create(
                model=cfg.local_llm.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                },
                            },
                            {"type": "text", "text": _VISION_PROMPT},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.0,
            )
        )
        raw = response.choices[0].message.content.strip()
        parsed = _extract_json(session_id, raw)

        state.present_speakers = parsed.get("present_speakers", [])
        state.engagement_cues  = parsed.get("engagement_cues", {})
        state.scene_summary    = parsed.get("scene_summary", "")
        state.last_updated     = time.time()
        log.debug(f"[vision:{session_id}] {state.scene_summary} | speakers={state.present_speakers}")

    except Exception as exc:
        state.error_count += 1
        log.warning(f"[vision:{session_id}] frame analysis failed: {exc}")

    return state