"""
app/schemas/livekit.py — LiveKit / room request & response models.
"""

from __future__ import annotations
from pydantic import BaseModel


class RoomCreateRequest(BaseModel):
    display_name: str = "browser-user"


class RoomCreateResponse(BaseModel):
    session_id: str
    token:      str
    lk_url:     str