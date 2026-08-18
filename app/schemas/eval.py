"""
app/schemas/eval.py — Evaluation metric request models.
"""

from __future__ import annotations
from pydantic import BaseModel


class WerRequest(BaseModel):
    session_id: str
    reference:  str
    hypothesis: str