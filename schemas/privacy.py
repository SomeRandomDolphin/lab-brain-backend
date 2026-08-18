"""
app/schemas/privacy.py — Privacy / consent request models.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


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