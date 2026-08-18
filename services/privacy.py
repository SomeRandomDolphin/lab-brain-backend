"""
app/services/privacy.py — Privacy & Consent Layer.

1. Consent registry  — persisted to consent.json across restarts.
2. PII redaction     — two-layer scrubber applied to transcript text before
                        LKC writes: a fast regex pass for structured,
                        locale-specific formats (Indonesian NIK, +62 phone,
                        card numbers) the ML model below wasn't trained on,
                        then openai/privacy-filter for unstructured PII
                        (names, addresses, emails, URLs, dates, secrets)
                        that regex can't reliably catch.
3. Blur signal       — tells vision to skip face labels for non-consenting
                        speakers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONSENT_FILE = Path("consent.json")

# ── PII redaction patterns (layer 1 — regex, fast + deterministic) ────────────
REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),    "[EMAIL]"),
    (re.compile(r"\b(?:\+62|0)[\d\s\-]{8,14}\b"),          "[PHONE]"),
    (re.compile(r"\b\d{16}\b"),                             "[NIK]"),
    (re.compile(r"\b(?:\d[ \-]?){13,19}\b"),               "[CARD]"),
]


def redact(text: str) -> str:
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── PII redaction (layer 2 — ML, unstructured PII) ─────────────────────────────
# Dedicated single-worker executor, same reasoning as vision.py's
# _vision_executor: this must never share the default executor with
# WhisperX/diarization, or a slow inference pass here would stall audio
# processing the same way unthrottled vision calls used to.
_pii_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-filter")
_pii_pipe = None
PII_FILTER_AVAILABLE = False


def _load_pii_filter() -> None:
    global _pii_pipe, PII_FILTER_AVAILABLE
    try:
        from transformers import pipeline
        _pii_pipe = pipeline(
            task="token-classification",
            model="openai/privacy-filter",
            aggregation_strategy="simple",
        )
        PII_FILTER_AVAILABLE = True
        log.info("[privacy] openai/privacy-filter loaded.")
    except ImportError:
        PII_FILTER_AVAILABLE = False
        log.warning(
            "[privacy] transformers not installed — ML PII redaction disabled, "
            "regex-only redaction active. Run: pip install transformers torch "
            "--break-system-packages"
        )
    except Exception as exc:
        PII_FILTER_AVAILABLE = False
        log.warning(f"[privacy] could not load openai/privacy-filter, regex-only redaction: {exc}")


_load_pii_filter()

# openai/privacy-filter's 8-category taxonomy -> our redaction tokens.
# Note: it's primarily English-trained (per its model card), so it's a
# second layer on TOP of the regex patterns above, not a replacement —
# it may under-detect Indonesian names or address formats.
_ML_LABEL_MAP = {
    "private_person":  "[NAME]",
    "private_address": "[ADDRESS]",
    "private_email":   "[EMAIL]",
    "private_phone":   "[PHONE]",
    "account_number":  "[ACCOUNT]",
    "private_url":     "[URL]",
    "private_date":    "[DATE]",
    "secret":          "[SECRET]",
}


def _ml_redact_sync(text: str) -> str:
    if not PII_FILTER_AVAILABLE or not text.strip():
        return text
    try:
        spans = _pii_pipe(text)
    except Exception as exc:
        log.warning(f"[privacy] ML redaction pass failed, keeping regex-only result: {exc}")
        return text
    out = text
    # Replace back-to-front so earlier span offsets stay valid as the string shrinks/grows.
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        label = _ML_LABEL_MAP.get(span["entity_group"], "[PII]")
        out = out[: span["start"]] + label + out[span["end"] :]
    return out


async def redact_async(text: str) -> str:
    """Two-layer redaction: regex first (cheap, deterministic, catches
    locale-specific structured formats), then openai/privacy-filter for
    unstructured PII. Falls back to regex-only if the model isn't loaded."""
    text = redact(text)
    if not PII_FILTER_AVAILABLE:
        return text
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pii_executor, _ml_redact_sync, text)


# ── Consent registry ───────────────────────────────────────────────────────────
_registry: dict[str, dict] = {}

# Conservative default: nobody is recorded unless they opt in.
DEFAULT_CONSENT: bool = False


def _load() -> None:
    global _registry
    if CONSENT_FILE.exists():
        try:
            _registry = json.loads(CONSENT_FILE.read_text(encoding="utf-8"))
            log.info(f"[privacy] consent registry loaded ({len(_registry)} entries).")
        except Exception as exc:
            log.warning(f"[privacy] could not load consent.json: {exc}")


def _save() -> None:
    try:
        CONSENT_FILE.write_text(
            json.dumps(_registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        log.warning(f"[privacy] could not save consent.json: {exc}")


_load()


def check_consent(speaker_label: str) -> bool:
    entry = _registry.get(speaker_label)
    if entry is None:
        return DEFAULT_CONSENT
    return bool(entry.get("consented", False))


def register_consent(
    speaker_label: str,
    consented: bool,
    real_name: Optional[str] = None,
) -> dict:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _registry[speaker_label] = {
        "consented": consented,
        "ts":        ts,
        "name":      real_name,
    }
    _save()
    log.info(f"[privacy] {speaker_label} consent={'yes' if consented else 'no'} name={real_name}")
    return _registry[speaker_label]


def all_consents() -> dict:
    return dict(_registry)


def revoke_consent(speaker_label: str) -> bool:
    removed = _registry.pop(speaker_label, None)
    if removed:
        _save()
    return removed is not None


def should_identify_face(speaker_label: str) -> bool:
    return check_consent(speaker_label)