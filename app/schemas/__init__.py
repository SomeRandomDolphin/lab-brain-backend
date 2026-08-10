"""
app/schemas/ — Pydantic v2 request/response models.
Grouped by domain for clarity.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

# Auth schemas re-exported for convenience
from app.schemas.auth import (  # noqa: F401
    UserOut,
    AuthResponse,
    RegisterRequest,
    LoginRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    OkResponse,
    TosConsentRequest,
)


# ── LiveKit / Room ─────────────────────────────────────────────────────────────

class RoomCreateRequest(BaseModel):
    display_name: str = "browser-user"


class RoomCreateResponse(BaseModel):
    session_id: str
    token:      str
    lk_url:     str


# ── Privacy / Consent ─────────────────────────────────────────────────────────

class ConsentRequest(BaseModel):
    speaker:    str
    consented:  bool
    real_name:  Optional[str] = None


class ConsentSyncRequest(BaseModel):
    """
    Dual-write consent: local registry + Supabase.

    session_id is required as of migration 0009 — consent_registry is now
    scoped per session (speaker_label alone collided across sessions).
    """
    session_id: str
    speaker:    str
    consented:  bool
    real_name:  Optional[str] = None


# ── Evaluation ────────────────────────────────────────────────────────────────

class WerRequest(BaseModel):
    session_id: str
    reference:  str
    hypothesis: str


# ── Rifqi Module 2 ingest ─────────────────────────────────────────────────────

class RifqiSegment(BaseModel):
    session_id: str
    speaker:    str
    text:       str
    timestamp:  Optional[str] = None
    source:     str = "module2"
    mode:       str = "meeting_capture"
    language:   str = "id"
    extra:      Optional[dict] = None


# ── Migration ─────────────────────────────────────────────────────────────────

class MigrationRunResponse(BaseModel):
    applied:  list[str]
    skipped:  list[str]
    errors:   list[str]
    dry_run:  bool