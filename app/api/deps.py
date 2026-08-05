"""
app/api/deps.py — FastAPI dependency functions.

get_current_user()
    Extracts `Authorization: Bearer <token>` from the request (or, as a
    fallback used only by the SSE route, a `?token=` query param — see the
    note below), validates the Supabase JWT via
    supabase_auth.verify_session_token(), and returns the user dict.
    Raises HTTP 401 on any failure.

get_optional_user()
    Same but returns None instead of raising — for endpoints that
    behave differently for authenticated vs anonymous users.

require_session_access(session_id)
    Authenticated user must be the session's owner OR a recorded
    participant. Use for view/join-level actions (transcripts, summary,
    joining via token, etc). Returns 404 (not 403) on mismatch — this
    deliberately doesn't confirm the session exists to someone who
    isn't allowed to see it.

require_session_owner(session_id)
    Authenticated user must be the session's owner. Use for destructive/
    management actions (deleting a room, wiping session records).

require_admin
    Authenticated user must have isAdmin=True (Supabase user_metadata
    role="admin" — see supabase_auth.py). Use for operator-only routes
    (migrations, full-graph dumps/wipes) that aren't scoped to any one
    session.

Query-param token fallback (get_current_user)
    Browser `EventSource` (used by GET /events/{session_id} for the SSE
    stream) cannot set custom headers, so it can't send
    `Authorization: Bearer <token>` like every other request. get_current_user
    accepts the access token via `?token=` as a fallback specifically for
    that case. Every other route continues to authenticate via the header —
    the frontend only ever sends ?token= to the SSE endpoint (see
    lib/api.ts::openSSE). This does mean the SSE URL carries a short-lived
    access token as a query param (visible in server logs / browser history)
    rather than a header; acceptable given Supabase access tokens are
    short-lived, but worth knowing if that ever changes.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db import supabase_auth, supabase_client

_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
)

_404_SESSION = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Session not found.",
)

_403_ADMIN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Admin access required.",
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(
        default=None,
        description="Access token fallback for clients that can't set headers (SSE/EventSource only).",
    ),
) -> dict:
    """
    Require a valid Supabase JWT. Raises 401 if missing or invalid.

    Prefers the Authorization header; falls back to ?token= only when no
    header is present (see module docstring — this exists for the SSE
    route). Every non-SSE call site continues to use the header as before.
    """
    raw = credentials.credentials if credentials is not None else token
    if raw is None:
        raise _401
    user = supabase_auth.verify_session_token(raw)
    if user is None:
        raise _401
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(default=None),
) -> Optional[dict]:
    """Return user dict or None — does not raise on missing/invalid token."""
    raw = credentials.credentials if credentials is not None else token
    if raw is None:
        return None
    return supabase_auth.verify_session_token(raw)


async def require_session_access(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> str:
    """
    Owner OR participant. `session_id` is resolved by FastAPI from whatever
    path/query parameter the endpoint itself declares under that name —
    this dependency doesn't change how the parameter is bound, it just adds
    a check on top of it.
    """
    owner, participants = await supabase_client.get_session_access(session_id)
    if owner is None:
        raise _404_SESSION
    if current_user["id"] != owner and current_user["id"] not in participants:
        raise _404_SESSION
    return session_id


async def require_session_owner(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> str:
    """Owner only — for destructive/management actions."""
    owner, _participants = await supabase_client.get_session_access(session_id)
    if owner is None or current_user["id"] != owner:
        raise _404_SESSION
    return session_id


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Operator-only routes not scoped to a single session."""
    if not current_user.get("isAdmin"):
        raise _403_ADMIN
    return current_user