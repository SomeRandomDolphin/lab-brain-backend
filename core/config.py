"""
app/core/config.py — Central configuration loader for Lab Brain.

Reads config.json (repo root) and env-var overrides.
Exposes a single `cfg` singleton used throughout the 
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
CONFIG_PATH = Path(__file__).parent.parent / "config.json"


# ── Typed sub-configs ──────────────────────────────────────────────────────────

@dataclass
class LocalLLMConfig:
    base_url:       str = "http://host.docker.internal:11434/v1"
    api_key:        str = "ollama"
    vision_model:   str = "qwen3-vl:4b"
    dialogue_model: str = "qwen3:4b"
    hf_token:       str = ""

    @property
    def available(self) -> bool:
        return True


@dataclass
class WhisperConfig:
    model_size:   str           = "large-v3-turbo"
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
    silence_chunks:     int   = 40
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
    url:            str  = "ws://host.docker.internal:7880"
    # public_url is what gets handed to the *browser* in join-room API
    # responses (create_room / get_token / client_config). It has to be
    # reachable from wherever the browser actually sits — a Tailscale IP,
    # a public hostname, etc — never host.docker.internal, which only
    # resolves inside a container. Falls back to `url` below if unset, so
    # existing deployments where the two happen to coincide keep working
    # with no config change required.
    public_url:     str  = ""
    api_key:        str  = "devkey"
    api_secret:     str  = "devsecret"
    egress_enabled: bool = False


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
    # S3-compatible endpoint for LiveKit Egress uploads (Supabase Storage
    # speaks S3). s3_endpoint is left blank here on purpose — if it's not
    # set explicitly (via config.json or S3_ENDPOINT env var), load() below
    # derives it from `url` as f"{url}/storage/v1/s3", which is correct for
    # both self-hosted and cloud Supabase projects and means you don't need
    # to set it separately from SUPABASE_URL in the common case.
    s3_endpoint:    str = ""
    s3_bucket:      str = ""
    s3_region:      str = "us-east-1"  # placeholder; self-hosted Storage doesn't enforce this
    s3_access_key:  str = ""
    s3_secret_key:  str = ""


@dataclass
class KgAgentConfig:
    """
    Client config for the shared, read-only kg-agent service on citi-condor
    — Neo4j-backed QA over a fixed 8-paper corpus (servitization, ISO 9001,
    supply chain integration, the Hawthorne effect). This is NOT session-
    transcript retrieval — that's LkcConfig.retrieval_top_k / lkc_retrieval.py,
    a separate system this doesn't replace. session_pipeline.py races the
    two and falls back to the transcript when a question isn't in-corpus.

    See app/services/kg_agent_client.py for the full contract notes
    (agentic must stay False, `passed` isn't a quality signal, don't lower
    request_timeout_seconds below the server's own 300s ceiling).
    """
    enabled:                          bool  = True
    base_url:                         str   = "http://100.122.56.39:8003"
    request_timeout_seconds:          float = 300.0  # server-side ceiling — do not lower
    soft_deadline_seconds:            float = 6.0    # UX budget for the hybrid race in live QA
    faithfulness_threshold:           float = 0.7     # documented gate; passed:false is expected/ignored
    circuit_breaker_cooldown_seconds: float = 30.0


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
    kg_agent:  KgAgentConfig  = field(default_factory=KgAgentConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────

def _apply(dataclass_obj, raw: dict) -> None:
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if hasattr(dataclass_obj, key):
            setattr(dataclass_obj, key, val)


def _looks_like_jwt(token: str) -> bool:
    """
    Cheap sanity check, not a real JWT validator: a Supabase service/anon
    key is a JWT — three '.'-separated segments, each valid base64url.
    This exists to catch one specific failure mode at startup instead of
    at the first storage upload: a stale/corrupted SUPABASE_SERVICE_KEY
    (truncated during copy-paste, a stray trailing newline/space, an old
    rotated key still sitting in .env) currently fails silently here and
    only surfaces later as a cryptic 403 "Failed to base64url decode the
    signature" the first time something tries to use it for a Storage
    call.
    """
    import base64
    parts = token.split(".")
    if len(parts) != 3:
        return False
    for part in parts:
        padded = part + "=" * (-len(part) % 4)
        try:
            base64.urlsafe_b64decode(padded)
        except Exception:
            return False
    return True


def load(path: Path = CONFIG_PATH) -> Config:
    # Don't rely on main.py having already called load_env() before this
    # module gets imported. `cfg = load()` below runs exactly once, at
    # whatever moment ANYTHING in the process first imports core.config
    # — if that happens to occur before main.py's load_env() call (e.g. via
    # some other import chain, a package __init__.py, alembic's own env.py,
    # a test importing this module directly, etc.), this singleton would
    # otherwise lock in a stale, empty environment permanently, with no
    # error — just a fallback to config.json that looks like ".env isn't
    # being read" when it's really an import-order race. load_env() is
    # documented as idempotent/cheap to call repeatedly, so calling it here
    # too removes the dependency on being imported after main.py's call.
    #
    # override=True: .env is now the source of truth, full stop — it always
    # wins, even over a real OS-level env var of the same name. This is a
    # deliberate change from env.py's documented default (real env wins),
    # made specifically because a stale SUPABASE_SERVICE_KEY sitting in the
    # real Windows environment was silently shadowing a correct, freshly
    # edited .env value and causing storage uploads to fail with a cryptic
    # 403 "Failed to base64url decode the signature". Editing .env should
    # always be enough to change what the app sees, with no separate step
    # to also clear/update a shell-level variable.
    #
    # Tradeoff to know about: if this app is ever deployed somewhere that
    # intentionally injects secrets as real environment variables (Docker,
    # Render, Railway, etc. — the exact scenario env.py's default was
    # designed for) AND a .env file happens to exist in that environment
    # too, .env would now win there as well. That's fine as long as no
    # .env file ships to production; if it ever might, this should go back
    # to the default override=False (or override just the Supabase keys
    # specifically) before deploying.
    from core.env import load_env
    load_env(override=True)

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
        "kg_agent":  c.kg_agent,
    }
    for section, obj in section_map.items():
        if section in raw:
            _apply(obj, raw[section])

    # Env vars always take precedence over config.json values for secrets.
    # Blank/whitespace-only values are treated as "not set" so a stray empty
    # env var doesn't shadow a real one from .env or config.json.
    env_url = (os.environ.get("SUPABASE_URL") or "").strip()
    env_key = (os.environ.get("SUPABASE_SERVICE_KEY") or "").strip()

    url_source = "env"
    key_source = "env"
    if env_url:
        c.supabase.url = env_url
    elif c.supabase.url:
        url_source = "config.json"
    if env_key:
        c.supabase.key = env_key
    elif c.supabase.key:
        key_source = "config.json"

    # Push the *resolved* values back into os.environ so every module that
    # reads os.environ directly (supabase_client.py, etc.) sees exactly what
    # cfg resolved to — using setdefault() here would silently keep a blank
    # pre-existing env var instead of the correct fallback.
    if c.supabase.url:
        os.environ["SUPABASE_URL"] = c.supabase.url
    if c.supabase.key:
        os.environ["SUPABASE_SERVICE_KEY"] = c.supabase.key

    if c.supabase.url or c.supabase.key:
        log.info(
            f"[config] supabase.url from {url_source}, supabase.key from {key_source}"
        )
    if key_source == "config.json" and c.supabase.key:
        log.warning(
            "[config] SUPABASE_SERVICE_KEY not found in the real environment or .env — "
            "falling back to the key hardcoded in config.json. Set SUPABASE_SERVICE_KEY "
            "in your .env instead; config.json is not meant to hold secrets long-term."
        )
    if c.supabase.key and not _looks_like_jwt(c.supabase.key):
        log.warning(
            f"[config] SUPABASE_SERVICE_KEY (resolved from {key_source}) doesn't look "
            "like a valid JWT (expected 3 '.'-separated base64url segments). Storage "
            "uploads will likely fail with a cryptic 403 'Failed to base64url decode "
            "the signature' error the first time they run. Double-check the value in "
            "your .env file — it's now always the source of truth for this key."
        )

    # ── S3-compatible credentials (LiveKit Egress -> Supabase Storage) ────────
    # Same env-wins-over-config.json precedence as the Supabase URL/key above,
    # for the same reason: these are secrets and .env is the source of truth.
    env_s3_access_key = (os.environ.get("S3_PROTOCOL_ACCESS_KEY_ID") or "").strip()
    env_s3_secret_key = (os.environ.get("S3_PROTOCOL_ACCESS_KEY_SECRET") or "").strip()
    env_s3_bucket     = (os.environ.get("S3_BUCKET") or "").strip()
    env_s3_region     = (os.environ.get("S3_REGION") or "").strip()
    env_s3_endpoint   = (os.environ.get("S3_ENDPOINT") or "").strip()

    if env_s3_access_key:
        c.supabase.s3_access_key = env_s3_access_key
    if env_s3_secret_key:
        c.supabase.s3_secret_key = env_s3_secret_key
    if env_s3_bucket:
        c.supabase.s3_bucket = env_s3_bucket
    if env_s3_region:
        c.supabase.s3_region = env_s3_region

    if env_s3_endpoint:
        c.supabase.s3_endpoint = env_s3_endpoint
    elif not c.supabase.s3_endpoint and c.supabase.url:
        # Derive from SUPABASE_URL rather than requiring a separate var in
        # the common case — this is correct for both self-hosted Supabase
        # (http://host:port/storage/v1/s3) and cloud projects
        # (https://<project>.supabase.co/storage/v1/s3).
        c.supabase.s3_endpoint = f"{c.supabase.url.rstrip('/')}/storage/v1/s3"

    env_egress_enabled = (os.environ.get("LIVEKIT_EGRESS_ENABLED") or "").strip().lower()
    if env_egress_enabled in ("1", "true", "yes"):
        c.livekit.egress_enabled = True
    elif env_egress_enabled in ("0", "false", "no"):
        c.livekit.egress_enabled = False

    if c.livekit.egress_enabled and not (c.supabase.s3_access_key and c.supabase.s3_secret_key and c.supabase.s3_bucket):
        log.warning(
            "[config] livekit.egress_enabled is True but S3 credentials/bucket are "
            "incomplete (need S3_PROTOCOL_ACCESS_KEY_ID, S3_PROTOCOL_ACCESS_KEY_SECRET, "
            "and S3_BUCKET) — recordings will fail to start until these are set."
        )

    # ── LiveKit connection (env-wins, same reasoning as Supabase above) ───────
    # api_secret in particular is a real credential and previously had no env
    # override at all — only config.json, which isn't meant to hold secrets
    # long-term (self-hosted LiveKit's api_secret is exactly as sensitive as
    # the Supabase service key above).
    env_lk_url        = (os.environ.get("LIVEKIT_URL") or "").strip()
    env_lk_public_url = (os.environ.get("LIVEKIT_PUBLIC_URL") or "").strip()
    env_lk_api_key    = (os.environ.get("LIVEKIT_API_KEY") or "").strip()
    env_lk_api_secret = (os.environ.get("LIVEKIT_API_SECRET") or "").strip()
    if env_lk_url:
        c.livekit.url = env_lk_url
    if env_lk_public_url:
        c.livekit.public_url = env_lk_public_url
    if env_lk_api_key:
        c.livekit.api_key = env_lk_api_key
    if env_lk_api_secret:
        c.livekit.api_secret = env_lk_api_secret

    # If nothing set public_url explicitly (neither config.json nor
    # LIVEKIT_PUBLIC_URL), fall back to `url` so this doesn't silently
    # break deployments that never knew about the split. This intentionally
    # runs *after* the LIVEKIT_URL env override above, so on this box it
    # would fall back to host.docker.internal too — LIVEKIT_PUBLIC_URL (or
    # config.json's livekit.public_url) must be set explicitly to fix the
    # browser-facing address; there is no way to derive a real reachable
    # address automatically.
    if not c.livekit.public_url:
        c.livekit.public_url = c.livekit.url

    # ── kg-agent (env override, same env-wins reasoning as above) ─────────────
    # Not a secret — the service has no auth, Tailscale is the access
    # boundary — but still overridable so a dev box can point at a
    # different kg-agent deployment without editing config.json.
    env_kg_base_url = (os.environ.get("KG_AGENT_BASE_URL") or "").strip()
    env_kg_enabled  = (os.environ.get("KG_AGENT_ENABLED") or "").strip().lower()
    if env_kg_base_url:
        c.kg_agent.base_url = env_kg_base_url
    if env_kg_enabled in ("1", "true", "yes"):
        c.kg_agent.enabled = True
    elif env_kg_enabled in ("0", "false", "no"):
        c.kg_agent.enabled = False

    log.info(
        f"Config loaded. LLM: {c.local_llm.base_url} | "
        f"summon_required={c.summon.require_summon} | "
        f"NER={c.spacy.model} | graph={c.lkc.db_file} | "
        f"livekit={c.livekit.url} public={c.livekit.public_url} "
        f"(egress={'on' if c.livekit.egress_enabled else 'off'}) | "
        f"supabase={'configured' if c.supabase.url else 'disabled'} | "
        f"kg_agent={c.kg_agent.base_url if c.kg_agent.enabled else 'disabled'}"
    )
    return c


# Module-level singleton
cfg: Config = load()