"""
config.py — Central configuration loader for Module 5.

Reads config.json from the same directory as this file.
All other modules import from here instead of reading env vars or
hardcoding constants directly.

Usage:
    from config import cfg

    cfg.local_llm.base_url
    cfg.local_llm.vision_model
    cfg.vad.silence_threshold
    ...
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.json"

# ── Typed sub-configs ─────────────────────────────────────────────────────────

@dataclass
class LocalLLMConfig:
    base_url:       str = "http://localhost:11434/v1"
    api_key:        str = "ollama"          # Ollama ignores this; other servers may need it
    vision_model:   str = "llava:7b"        # must support image inputs
    dialogue_model: str = "llama3.2:3b"    # text-only chat model
    hf_token:       str = ""                # Hugging Face token for gated models (e.g. pyannote diarization)

    @property
    def available(self) -> bool:
        """Always true — no remote API key required for a local server."""
        return True


@dataclass
class WhisperConfig:
    model_size:   str           = "small"  # tiny, base, small, medium, turbo
    device:       str           = "cpu"
    compute_type: str           = "int8"
    beam_size:    int           = 1       # 1=greedy (3-4x faster, ~same WER for conversational speech)
    language:     Optional[str] = None   # None = auto-detect
    cpu_threads:  int           = 8      # set to your physical core count
    num_workers:  int           = 2      # allows overlap between segments


@dataclass
class VadConfig:
    sample_rate:        int   = 16000
    silence_threshold:  float = 0.01
    silence_chunks:     int   = 4
    max_segment_chunks: int   = 30


@dataclass
class VisionConfig:
    frame_interval: int   = 5
    camera_fps:     int   = 5
    camera_quality: float = 0.6


@dataclass
class LkcConfig:
    log_file:       str = "lkc_stream.jsonl"
    retrieval_top_k: int = 4


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class DialogueConfig:
    context_window:   int = 12
    tts_auto_hide_ms: int = 8000


@dataclass
class Config:
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    whisper:  WhisperConfig   = field(default_factory=WhisperConfig)
    vad:      VadConfig      = field(default_factory=VadConfig)
    vision:   VisionConfig   = field(default_factory=VisionConfig)
    lkc:      LkcConfig      = field(default_factory=LkcConfig)
    server:   ServerConfig   = field(default_factory=ServerConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _apply(dataclass_obj, raw: dict) -> None:
    """Recursively set fields on a dataclass from a dict, ignoring unknown keys."""
    for key, val in raw.items():
        if key.startswith("_"):
            continue  # skip comment keys
        if hasattr(dataclass_obj, key):
            setattr(dataclass_obj, key, val)


def load(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        log.warning(f"config.json not found at {path} — using defaults.")
        return Config()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.error(f"config.json parse error: {e}")
        sys.exit(1)

    c = Config()
    section_map = {
        "local_llm": c.local_llm,
        "whisper":  c.whisper,
        "vad":      c.vad,
        "vision":   c.vision,
        "lkc":      c.lkc,
        "server":   c.server,
        "dialogue": c.dialogue,
    }
    for section, obj in section_map.items():
        if section in raw:
            _apply(obj, raw[section])

    if not c.local_llm.available:
        log.warning("Local LLM base_url not set — vision and dialogue will run in stub mode.")
    else:
        log.info(f"Config loaded. Local LLM endpoint: {c.local_llm.base_url}")

    return c


# ── Module-level singleton ────────────────────────────────────────────────────
cfg: Config = load()