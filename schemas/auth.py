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
    # None = the account-level privacy-screen decision hasn't been made yet;
    # the dashboard shows the first-login ToS modal until this is set.
    tosAccepted:   Optional[bool] = None
    tosAcceptedAt: Optional[str]  = None


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


# ── Privacy screen / ToS consent ───────────────────────────────────────────────
# NOTE: this is used by app/api/v1/endpoints/privacy.py's POST /tos-consent.
# It's defined here (alongside UserOut, since it's an account-level field)
# rather than in whatever schemas module ConsentRequest/ConsentSyncRequest
# live in — make sure it's re-exported from app/schemas/__init__.py the same
# way those are, or the `from schemas import ... TosConsentRequest`
# import in the endpoints file will fail.

class TosConsentRequest(BaseModel):
    accepted: bool


# ── Profile update ──────────────────────────────────────────────────────────
# Used by PATCH /auth/me. All fields optional — only whatever's present in
# the request body gets changed (see UpdateProfileRequest.model_dump's use
# of exclude_unset in the endpoint). email going through this path (rather
# than a dedicated /auth/change-email + verification flow) means a changed
# email takes effect immediately with no re-confirmation step — see the
# email_confirm=True passed alongside it in supabase_auth.update_profile.

class UpdateProfileRequest(BaseModel):
    name:      Optional[str] = Field(default=None, min_length=1, max_length=100)
    email:     Optional[str] = Field(default=None, min_length=3, max_length=254)
    avatarUrl: Optional[str] = Field(default=None, max_length=2048)

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower() if v is not None else v

    @field_validator("name")
    @classmethod
    def name_strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v is not None else v