"""
privacy.py — Privacy & Consent Layer (Module 5, Month 3)

Implements the privacy-gating requirements from the Month 3 roadmap:

  1. Consent registry   — tracks who has opted in/out of recording.
                          Persisted to consent.json so consent survives
                          server restarts.

  2. PII redaction      — lightweight regex scrubber applied to transcript
                          text before LKC writes.  Masks email addresses,
                          phone numbers, and Indonesian NIK (16-digit ID)
                          by default.  Extend REDACT_PATTERNS as needed.

  3. Blur signal        — tells vision.py to skip face labels for speakers
                          who have not consented to identification.

  4. Consent API hooks  — FastAPI router mounted in server.py so the browser
                          can show a consent banner and register opt-in/out
                          without restarting the session.

Usage (server.py):
    from privacy import router as privacy_router, check_consent, redact
    app.include_router(privacy_router, prefix="/privacy")

    # Before writing a transcript segment to the LKC:
    if check_consent(speaker):
        text = redact(text)
        write_to_lkc(...)
    else:
        # Skip — speaker has not consented or has opted out
        pass
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger(__name__)

CONSENT_FILE = Path("consent.json")

# ── PII redaction patterns ─────────────────────────────────────────────────────
REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), "[EMAIL]"),
    # Indonesian phone numbers (08xx / +628xx, 10-13 digits)
    (re.compile(r"\b(?:\+62|0)[\d\s\-]{8,14}\b"), "[PHONE]"),
    # NIK — 16-digit Indonesian national ID
    (re.compile(r"\b\d{16}\b"), "[NIK]"),
    # Credit / debit card numbers (13-19 digits, optional dashes/spaces)
    (re.compile(r"\b(?:\d[ \-]?){13,19}\b"), "[CARD]"),
]


def redact(text: str) -> str:
    """Apply all PII redaction patterns to a transcript string."""
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Consent registry ──────────────────────────────────────────────────────────
# Internal store: {speaker_label: {"consented": bool, "ts": iso_str, "name": str|None}}
_registry: dict[str, dict] = {}


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


# Load once at import time
_load()

# Default policy when a speaker is not in the registry.
# Set to True for opt-out (record everyone unless they say no).
# Set to False for opt-in (record nobody unless they say yes).
DEFAULT_CONSENT: bool = False  # conservative default for research setting


def check_consent(speaker_label: str) -> bool:
    """Return True if the speaker is allowed to be recorded."""
    entry = _registry.get(speaker_label)
    if entry is None:
        return DEFAULT_CONSENT
    return bool(entry.get("consented", False))


def register_consent(speaker_label: str, consented: bool, real_name: Optional[str] = None) -> dict:
    """Update consent for a speaker label. Returns the new registry entry."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _registry[speaker_label] = {
        "consented": consented,
        "ts": ts,
        "name": real_name,
    }
    _save()
    log.info(f"[privacy] {speaker_label} consent={'yes' if consented else 'no'} name={real_name}")
    return _registry[speaker_label]


def all_consents() -> dict:
    return dict(_registry)


def should_identify_face(speaker_label: str) -> bool:
    """
    Returns True if the vision module is allowed to show a face label for
    this speaker in the UI / LKC.  False → vision.py returns "Person (anon)".
    """
    return check_consent(speaker_label)


# ── FastAPI router ─────────────────────────────────────────────────────────────
router = APIRouter(tags=["privacy"])


class ConsentRequest(BaseModel):
    speaker: str
    consented: bool
    real_name: Optional[str] = None


@router.get("/status")
async def privacy_status():
    """Return all consent registrations and current default policy."""
    return {
        "default_consent": DEFAULT_CONSENT,
        "registry": all_consents(),
    }


@router.post("/consent")
async def post_consent(req: ConsentRequest):
    """Register or update consent for a speaker."""
    entry = register_consent(req.speaker, req.consented, req.real_name)
    return {"speaker": req.speaker, **entry}


@router.delete("/consent/{speaker}")
async def delete_consent(speaker: str):
    """Remove a speaker from the registry (reverts to default policy)."""
    removed = _registry.pop(speaker, None)
    if removed:
        _save()
    return {"speaker": speaker, "removed": removed is not None}