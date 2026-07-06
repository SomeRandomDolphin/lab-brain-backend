"""
app/core/config.py — Central configuration loader for Lab Brain.

Reads config.json (repo root) and env-var overrides.
Exposes a single `cfg` singleton used throughout the app.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# config.json lives two levels above this file (project root)
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


# ── Typed sub-configs ──────────────────────────────────────────────────────────

@dataclass
class LocalLLMConfig:
    base_url:       str = "http://localhost:11434/v1"
    api_key:        str = "ollama"
    vision_model:   str = "llava:7b"
    dialogue_model: str = "llama3.2:3b"
    hf_token:       str = ""

    @property
    def available(self) -> bool:
        return True


@dataclass
class WhisperConfig:
    model_size:   str           = "small"
    device:       str           = "cpu"
    compute_type: str           = "int8"
    beam_size:    int           = 1
    language:     Optional[str] = None
    cpu_threads:  int           = 8
    num_workers:  int           = 2


@dataclass
class VadConfig:
    sample_rate:        int   = 16000
    silence_threshold:  float = 0.03   # float32 norm audio; 0.01 was too low and classified speech as silence
    silence_chunks:     int   = 4
    max_segment_chunks: int   = 30


@dataclass
class VisionConfig:
    frame_interval: int   = 5
    camera_fps:     int   = 5
    camera_quality: float = 0.6


@dataclass
class LkcConfig:
    log_file:        str = "lkc_stream.jsonl"
    db_file:         str = "lkc_graph.db"
    retrieval_top_k: int = 4


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class DialogueConfig:
    context_window:   int = 12
    tts_auto_hide_ms: int = 8000


@dataclass
class SummonConfig:
    phrases: List[str] = field(default_factory=lambda: [
        "lab brain", "hey brain", "hey lab brain",
        "@lab", "brain,", "brain?",
    ])
    require_summon: bool = True


@dataclass
class SpacyConfig:
    model:        str       = "en_core_web_sm"
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "PRODUCT", "GPE", "DATE", "EVENT", "WORK_OF_ART"
    ])


@dataclass
class LiveKitConfig:
    """
    LiveKit SFU configuration.

    Quick local dev:
        docker run --rm -p 7880:7880 livekit/livekit-server --dev
    Then set api_key="devkey", api_secret="devsecret".
    """
    url:        str = "ws://localhost:7880"
    api_key:    str = "devkey"
    api_secret: str = "devsecret"


@dataclass
class SupabaseConfig:
    """
    Supabase persistence layer config.
    url / key are always overridden by SUPABASE_URL / SUPABASE_KEY env vars.
    Use the service-role key (not anon) so the backend can bypass RLS.
    """
    url:          str  = ""
    key:          str  = ""
    store_audio:  bool = False
    store_vision: bool = True


@dataclass
class Config:
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    whisper:   WhisperConfig  = field(default_factory=WhisperConfig)
    vad:       VadConfig      = field(default_factory=VadConfig)
    vision:    VisionConfig   = field(default_factory=VisionConfig)
    lkc:       LkcConfig      = field(default_factory=LkcConfig)
    server:    ServerConfig   = field(default_factory=ServerConfig)
    dialogue:  DialogueConfig = field(default_factory=DialogueConfig)
    summon:    SummonConfig   = field(default_factory=SummonConfig)
    spacy:     SpacyConfig    = field(default_factory=SpacyConfig)
    livekit:   LiveKitConfig  = field(default_factory=LiveKitConfig)
    supabase:  SupabaseConfig = field(default_factory=SupabaseConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────

def _apply(dataclass_obj, raw: dict) -> None:
    for key, val in raw.items():
        if key.startswith("_"):
            continue
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
        "whisper":   c.whisper,
        "vad":       c.vad,
        "vision":    c.vision,
        "lkc":       c.lkc,
        "server":    c.server,
        "dialogue":  c.dialogue,
        "summon":    c.summon,
        "spacy":     c.spacy,
        "livekit":   c.livekit,
        "supabase":  c.supabase,
    }
    for section, obj in section_map.items():
        if section in raw:
            _apply(obj, raw[section])

    # Env vars always take precedence over config.json values for secrets
    if os.environ.get("SUPABASE_URL"):
        c.supabase.url = os.environ["SUPABASE_URL"]
    if os.environ.get("SUPABASE_SERVICE_KEY"):
        c.supabase.key = os.environ["SUPABASE_SERVICE_KEY"]
    # Push resolved values back into env so the supabase client picks them up
    if c.supabase.url:
        os.environ.setdefault("SUPABASE_URL", c.supabase.url)
    if c.supabase.key:
        os.environ.setdefault("SUPABASE_SERVICE_KEY", c.supabase.key)

    log.info(
        f"Config loaded. LLM: {c.local_llm.base_url} | "
        f"summon_required={c.summon.require_summon} | "
        f"NER={c.spacy.model} | graph={c.lkc.db_file} | "
        f"livekit={c.livekit.url} | "
        f"supabase={'configured' if c.supabase.url else 'disabled'}"
    )
    return c


# Module-level singleton
cfg: Config = load()