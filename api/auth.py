"""
app/api/auth.py — Authentication endpoints.

POST   /auth/register         — create account, return user + tokens
POST   /auth/login            — verify credentials, return user + tokens
POST   /auth/logout           — revoke token server-side
POST   /auth/refresh          — exchange refresh_token for new access_token
GET    /auth/me               — validate token, return current user
PATCH  /auth/me               — update current user's name/email/avatarUrl
POST   /auth/forgot-password  — trigger Supabase password-recovery email
POST   /auth/reset-password   — consume reset token, set new password

All session tokens are Supabase JWTs.  The frontend should store both
access_token (short-lived, ~1 hour) and refresh_token (long-lived) and
call POST /auth/refresh before the access_token expires.

get_current_user() / get_optional_user()
    Formerly lived in api/deps.py (now removed — each endpoint file that
    needs an auth dependency defines its own copy locally). get_current_user
    lives here since this is the auth module; get_optional_user isn't
    currently used by any endpoint file but is kept here for the same
    reason. If another endpoint file needs either of these, copy the
    definition into that file rather than importing it from here.

    Query-param token fallback: Browser `EventSource` (used by the SSE
    stream) cannot set custom headers, so it can't send
    `Authorization: Bearer <token>` like every other request.
    get_current_user accepts the access token via `?token=` as a fallback
    specifically for that case. Every other route continues to
    authenticate via the header.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from db import supabase_auth
from schemas.auth import (
    AuthResponse,
    ForgotPasswordRequest,
    LoginRequest,
    OkResponse,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    UpdateProfileRequest,
    UserOut,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
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


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
async def register(req: RegisterRequest):
    """
    Create a new user account via Supabase Auth.

    - **409** if the email is already registered.
    - **400** for validation failures (handled by Pydantic).
    """
    try:
        user = supabase_auth.create_user(
            name=req.name,
            email=req.email,
            password=req.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # Mint a session immediately so the client gets tokens on registration
    try:
        access_token, refresh_token = supabase_auth.create_session_for_user(user["id"])
    except Exception as exc:
        log.warning(f"[auth] session mint failed after register, falling back to sign-in: {exc}")
        result = supabase_auth.authenticate_user(req.email, req.password)
        if result is None:
            raise HTTPException(status_code=500, detail="Could not create session after registration.")
        user, access_token, refresh_token = result

    log.info(f"[auth] register: {user['email']} ({user['id']})")
    return AuthResponse(
        user=UserOut(**user),
        token=access_token,
        refreshToken=refresh_token,
    )


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Sign in with email + password",
)
async def login(req: LoginRequest):
    """
    Authenticate with email and password via Supabase Auth.

    Returns the same **401** for both wrong email and wrong password
    to avoid leaking which emails are registered.

    The response includes both an `access_token` (JWT, ~1 hour TTL) and a
    `refresh_token` (long-lived opaque token).  Store both client-side and
    call POST /auth/refresh when the access token nears expiry.
    """
    result = supabase_auth.authenticate_user(req.email, req.password)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    user, access_token, refresh_token = result
    log.info(f"[auth] login: {user['email']} ({user['id']})")
    return AuthResponse(
        user=UserOut(**user),
        token=access_token,
        refreshToken=refresh_token,
    )


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=OkResponse,
    summary="Invalidate the current session token",
)
async def logout(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """
    Revoke the Supabase session associated with the Bearer token.
    The frontend should also clear stored tokens regardless of this response.
    """
    auth_header = request.headers.get("authorization", "")
    raw_token   = auth_header.removeprefix("Bearer ").strip()
    if raw_token:
        supabase_auth.revoke_session_token(raw_token)
    log.info(f"[auth] logout: {current_user['email']}")
    return OkResponse()


# ── POST /auth/refresh ────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=AuthResponse,
    summary="Exchange a refresh token for a new access token",
)
async def refresh(req: RefreshRequest):
    """
    Use a valid refresh_token to obtain a fresh access_token.

    - **401** if the refresh token is invalid or expired.

    Call this endpoint when the frontend detects (or preemptively expects)
    an expired access token.
    """
    result = supabase_auth.refresh_access_token(req.refreshToken)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired. Please log in again.",
        )

    access_token, refresh_token = result
    user = supabase_auth.verify_session_token(access_token)
    if user is None:
        raise HTTPException(status_code=500, detail="Could not verify refreshed token.")

    return AuthResponse(
        user=UserOut(**user),
        token=access_token,
        refreshToken=refresh_token,
    )


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=dict,   # { "user": UserOut }
    summary="Return the currently authenticated user",
)
async def me(current_user: dict = Depends(get_current_user)):
    """
    Validate the Bearer token and return the current user.
    Called by `AuthProvider` on every page load to rehydrate the session.

    - **401** if the token is missing, expired, or revoked.
    """
    return {"user": UserOut(**current_user)}


# ── PATCH /auth/me ─────────────────────────────────────────────────────────────

@router.patch(
    "/me",
    response_model=dict,   # { "user": UserOut }
    summary="Update the current user's profile",
)
async def update_me(
    req: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Partial update of the current account's profile — name, email, and/or
    avatarUrl. Only fields present in the request body are changed;
    omitted fields are left as-is (exclude_unset, not "falsy value means
    clear it" — sending `{"name": ""}` would fail Pydantic's min_length=1
    rather than silently no-op).

    - **409** if the new email is already registered to a different account.
    """
    patch = req.model_dump(exclude_unset=True)
    if not patch:
        return {"user": UserOut(**current_user)}

    try:
        updated = supabase_auth.update_profile(
            current_user["id"],
            name=patch.get("name"),
            email=patch.get("email"),
            avatar_url=patch.get("avatarUrl"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    log.info(f"[auth] profile updated: {updated['email']} ({updated['id']})")
    return {"user": UserOut(**updated)}


# ── POST /auth/forgot-password ────────────────────────────────────────────────

@router.post(
    "/forgot-password",
    response_model=OkResponse,
    summary="Request a password-reset email",
)
async def forgot_password(req: ForgotPasswordRequest):
    """
    Trigger Supabase's built-in password-recovery email for the given address.

    Always returns **200 OK** — even if the email is not registered —
    to avoid leaking which email addresses have accounts.

    Supabase sends the email directly using your project's email settings.
    To customise the template, edit it in the Supabase dashboard under
    Authentication → Email Templates → Reset Password.
    """
    user = supabase_auth.get_user_by_email(req.email)
    if user:
        try:
            supabase_auth.create_reset_token(user["id"])
        except Exception as exc:
            log.error(f"[auth] forgot-password token generation failed: {exc}")
    else:
        log.debug(f"[auth] forgot-password: unknown email {req.email}")

    return OkResponse()


# ── POST /auth/reset-password ─────────────────────────────────────────────────

@router.post(
    "/reset-password",
    response_model=OkResponse,
    summary="Consume a reset token and set a new password",
)
async def reset_password(req: ResetPasswordRequest):
    """
    Validate the single-use recovery token (from the email link's `?token=` param)
    and update the user's password.

    - **400** if the token is expired, already used, or not found.
    - **400** if the new password is too short (< 8 chars, enforced by Pydantic).

    After a successful reset, **all active sessions for the user are revoked**
    so any other logged-in devices are signed out.
    """
    user_id = supabase_auth.consume_reset_token(req.token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token is invalid, expired, or has already been used.",
        )

    supabase_auth.update_user_password(user_id, req.password)
    supabase_auth.revoke_all_user_tokens(user_id)

    log.info(f"[auth] password reset completed for user {user_id}")
    return OkResponse()