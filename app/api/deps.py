"""
app/api/deps.py — FastAPI dependency functions.

get_current_user()
    Extracts `Authorization: Bearer <token>` from the request,
    validates the Supabase JWT via supabase_auth.verify_session_token(),
    and returns the user dict.  Raises HTTP 401 on any failure.

get_optional_user()
    Same but returns None instead of raising — for endpoints that
    behave differently for authenticated vs anonymous users.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db import supabase_auth

_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Require a valid Supabase JWT. Raises 401 if missing or invalid."""
    if credentials is None:
        raise _401
    user = supabase_auth.verify_session_token(credentials.credentials)
    if user is None:
        raise _401
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Return user dict or None — does not raise on missing/invalid token."""
    if credentials is None:
        return None
    return supabase_auth.verify_session_token(credentials.credentials)