"""
app/services/privacy.py — Privacy & Consent Layer.

1. Consent registry  — persisted to consent.json across restarts.
2. PII redaction     — regex scrubber applied to transcript text before LKC writes.
3. Blur signal       — tells vision to skip face labels for non-consenting speakers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONSENT_FILE = Path("consent.json")

# ── PII redaction patterns ─────────────────────────────────────────────────────
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
