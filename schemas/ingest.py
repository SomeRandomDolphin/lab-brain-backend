"""
app/schemas/ingest.py — External segment ingest models (Module 2 capture).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class RifqiSegment(BaseModel):
    session_id: str
    speaker:    str
    text:       str
    timestamp:  Optional[str] = None
    source:     str = "module2"
    mode:       str = "meeting_capture"
    language:   str = "id"
    extra:      Optional[dict] = None