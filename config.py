"""
config.py — Central configuration loader for Module 5 (Month 6).

Month 6 additions
-----------------
* LiveKitConfig — SFU URL, API key/secret for token signing and room management
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.json"


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
    log_file:        str = "lkc_stream.jsonl"
    db_file:         str = "lkc_graph.db"
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
class SummonConfig:
    phrases: List[str] = field(default_factory=lambda: [
        "lab brain",
        "hey brain",
        "hey lab brain",
        "@lab",
        "brain,",
        "brain?",
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
    Month 6 — LiveKit SFU configuration.

    url        : WebSocket URL of the LiveKit server (self-hosted or Cloud).
    api_key    : LiveKit API key (used to sign tokens and manage rooms).
    api_secret : LiveKit API secret (never sent to the browser).

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
    Month 7 — Supabase persistence layer.

    url  : Supabase project URL (https://<ref>.supabase.co).
           Overridden by SUPABASE_URL env var when set.
    key  : Supabase service-role secret key.
           Overridden by SUPABASE_KEY env var when set.
           Use the service-role key (not anon) so the backend can bypass RLS.

    store_audio  : upload raw PCM segments to the audio-segments Storage bucket.
                   Disable to save quota during development.
    store_vision : persist vision frame rows to Supabase.
                   Generates ~1 row/s; disable if not needed.

    NOTE: env vars always take precedence over values in config.json.
    """
    url:          str  = ""
    key:          str  = ""
    store_audio:  bool = False   # off by default — enable once bucket is created
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
    livekit:   LiveKitConfig  = field(default_factory=LiveKitConfig)   # Month 6
    supabase:  SupabaseConfig = field(default_factory=SupabaseConfig)  # Month 7


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
        "livekit":   c.livekit,   # Month 6
        "supabase":  c.supabase,  # Month 7
    }
    for section, obj in section_map.items():
        if section in raw:
            _apply(obj, raw[section])

    # Env vars override config.json values for secrets
    import os
    if os.environ.get("SUPABASE_URL"):
        c.supabase.url = os.environ["SUPABASE_URL"]
    if os.environ.get("SUPABASE_KEY"):
        c.supabase.key = os.environ["SUPABASE_KEY"]
    # Push resolved values back into env so supabase_store._get_client() picks them up
    if c.supabase.url:
        os.environ.setdefault("SUPABASE_URL", c.supabase.url)
    if c.supabase.key:
        os.environ.setdefault("SUPABASE_KEY", c.supabase.key)

    log.info(f"Config loaded. LLM: {c.local_llm.base_url} | "
             f"summon_required={c.summon.require_summon} | "
             f"NER={c.spacy.model} | "
             f"graph={c.lkc.db_file} | "
             f"livekit={c.livekit.url} | "
             f"supabase={'configured' if c.supabase.url else 'disabled'}")
    return c


cfg: Config = load()