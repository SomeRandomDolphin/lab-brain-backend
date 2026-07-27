"""
app/services/vision.py — VLM Perception Layer.

Receives JPEG frames, sends to a local VLM via OpenAI-compatible API,
returns speaker/engagement/environment analysis. Privacy-gated.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import cfg
from app.services import privacy as _privacy

log = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _client = OpenAI(
        base_url=cfg.local_llm.base_url,
        api_key=cfg.local_llm.api_key,
    )
    LOCAL_LLM_AVAILABLE = True
    log.info(f"Vision model ready ({cfg.local_llm.vision_model} @ {cfg.local_llm.base_url}).")
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    log.warning("openai package not installed — vision stub mode active.")
except Exception as exc:
    LOCAL_LLM_AVAILABLE = False
    log.warning(f"Could not initialise local LLM client: {exc} — vision stub mode.")

# Backward-compat alias used by pipeline
GEMINI_AVAILABLE = LOCAL_LLM_AVAILABLE


@dataclass
class PerceptionState:
    present_speakers: list[str] = field(default_factory=list)
    engagement_cues:  dict[str, str] = field(default_factory=dict)
    scene_summary:    str = ""
    last_updated:     float = 0.0
    frame_count:      int = 0
    error_count:      int = 0
    environment_state: dict = field(default_factory=lambda: {
        "objects": [], "layout": "unknown",
        "lighting": "unknown", "ambient": "unknown",
    })


_states: dict[str, PerceptionState] = {}


def get_state(session_id: str) -> PerceptionState:
    if session_id not in _states:
        _states[session_id] = PerceptionState()
    return _states[session_id]


def clear_state(session_id: str) -> None:
    _states.pop(session_id, None)


_VISION_PROMPT = """
Analyse this camera frame. Reply with ONLY a raw JSON object — no markdown, no code fences.

Example:
{
  "present_speakers": ["Person A"],
  "engagement_cues": {"Person A": "focused"},
  "scene_summary": "One person at a desk.",
  "environment_state": {
    "objects": ["laptop", "whiteboard"],
    "layout": "huddle",
    "lighting": "bright",
    "ambient": "quiet"
  }
}

Rules:
- present_speakers: array of visible person labels ("Person A", "Person B", …). Empty if none.
- engagement_cues: map each label to: focused | distracted | away | unknown
- scene_summary: one sentence, 15 words max. "Empty room." if nobody visible.
- environment_state.layout: classroom | huddle | open | home | unknown
- environment_state.lighting: bright | dim | mixed | unknown
- environment_state.ambient: quiet | noisy | unknown
- No personal names, no biometric data, no emotion diagnoses.
- Output ONLY the JSON.
""".strip()


def _extract_json(session_id: str, raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    log.warning(f"[vision:{session_id}] could not parse model output; raw={raw[:200]!r}")
    return {
        "present_speakers": [],
        "engagement_cues": {},
        "scene_summary": "Unknown — could not parse model response.",
    }


async def analyse_frame(session_id: str, jpeg_bytes: bytes, known_identity: Optional[str] = None) -> PerceptionState:
    state = get_state(session_id)
    state.frame_count += 1

    if not LOCAL_LLM_AVAILABLE:
        state.present_speakers = ["Person A"]
        state.engagement_cues  = {"Person A": "focused"}
        state.scene_summary    = "[Vision stub — configure local_llm in config.json]"
        state.last_updated     = time.time()
        return state

    b64  = base64.b64encode(jpeg_bytes).decode()
    loop = asyncio.get_event_loop()

    # Same instrumentation pattern as dialogue_service.py's qa/summary calls.
    # Without this, a vision call hitting the same Ollama instance shows up
    # in the httpx logger as an indistinguishable "200 OK" with no way to
    # tell it apart from a dialogue call — which is exactly what made an
    # earlier latency investigation (147s "warm" QA reply) take a file read
    # to resolve instead of a log read. call_id + explicit elapsed time
    # means the next one won't need that.
    call_id = uuid.uuid4().hex[:8]
    log.info(f"[vision:{session_id}] LLM dispatch frame call_id={call_id} model={cfg.local_llm.vision_model}")
    t0 = time.perf_counter()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _client.chat.completions.create(
                model=cfg.local_llm.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }],
                max_tokens=300,
                temperature=0.0,
                timeout=30,  # fail fast instead of silently retrying/hanging
            )
        )
        elapsed = time.perf_counter() - t0
        log.info(f"[vision:{session_id}] LLM complete frame call_id={call_id} ({elapsed:.1f}s)")
        raw    = response.choices[0].message.content.strip()
        parsed = _extract_json(session_id, raw)

        raw_speakers = parsed.get("present_speakers", [])
        raw_cues     = parsed.get("engagement_cues", {})

        # Privacy gate
        gated_speakers: list[str] = []
        gated_cues: dict[str, str] = {}
        # The VLM has no notion of real identity — "Person A"/"Person B" are
        # arbitrary per-frame labels, not a persistent ID. The one case where
        # we CAN safely attach a real name is a single detected person in a
        # session where we know who actually joined the room (the logged-in
        # account, via LiveKit's participant name) — there's no ambiguity
        # about who that is, and it's their own account recording their own
        # session, not a third party being identified without consent.
        # Multiple simultaneous faces still go through the normal
        # consent-registry gate below, since we can't tell which VLM label
        # corresponds to which real person.
        if known_identity and len(raw_speakers) == 1:
            sp = raw_speakers[0]
            gated_speakers.append(known_identity)
            gated_cues[known_identity] = raw_cues.get(sp, "unknown")
        else:
            for sp in raw_speakers:
                if _privacy.should_identify_face(sp):
                    gated_speakers.append(sp)
                    gated_cues[sp] = raw_cues.get(sp, "unknown")
                else:
                    anon = "Person (anon)"
                    gated_speakers.append(anon)
                    gated_cues[anon] = "unknown"

        state.present_speakers = gated_speakers
        state.engagement_cues  = gated_cues
        state.scene_summary    = parsed.get("scene_summary", "")

        env = parsed.get("environment_state", {})
        if env:
            state.environment_state = {
                "objects":  env.get("objects",  []),
                "layout":   env.get("layout",   "unknown"),
                "lighting": env.get("lighting", "unknown"),
                "ambient":  env.get("ambient",  "unknown"),
            }

        state.last_updated = time.time()
    except Exception as exc:
        state.error_count += 1
        log.warning(f"[vision:{session_id}] frame analysis failed call_id={call_id}: {exc}")

    return state