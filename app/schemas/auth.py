"""
app/schemas/auth.py — Auth request/response Pydantic models.

These match the User interface expected by the frontend (src/types/index.ts):
  { id, name, email, avatarUrl, createdAt, isAdmin }

Changes from the SQLite version:
  • AuthResponse now includes refreshToken (Supabase JWTs are short-lived;
    the frontend needs the refresh token to silently renew access).
  • RefreshRequest added for POST /auth/refresh.
  • UserOut now includes isAdmin (from Supabase user_metadata.role — see
    supabase_auth.py), so the frontend can conditionally show operator-only
    UI (migrations panel, full-graph viewer) without a separate whoami call.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── User response shape (matches frontend User interface) ─────────────────────

class UserOut(BaseModel):
    id:         str
    name:       str
    email:      str
    avatarUrl:  Optional[str] = None
    createdAt:  str           # ISO 8601
    isAdmin:    bool = False


# ── Register ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name:     str = Field(..., min_length=1, max_length=100)
    email:    str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("name")
    @classmethod
    def name_strip(cls, v: str) -> str:
        return v.strip()


class AuthResponse(BaseModel):
    user:         UserOut
    token:        str           # Supabase access_token (JWT)
    refreshToken: Optional[str] = None  # Supabase refresh_token (opaque)


# ── Login ─────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        return v.strip().lower()


# ── Token refresh ─────────────────────────────────────────────────────────────

class RefreshRequest(BaseModel):
    refreshToken: str = Field(..., min_length=1)


# ── Forgot / Reset password ───────────────────────────────────────────────────

class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        return v.strip().lower()


class ResetPasswordRequest(BaseModel):
    token:    str = Field(..., min_length=1)
    password: str = Field(..., min_length=8, max_length=128)


# ── Generic ok ────────────────────────────────────────────────────────────────

class OkResponse(BaseModel):
    ok: bool = True