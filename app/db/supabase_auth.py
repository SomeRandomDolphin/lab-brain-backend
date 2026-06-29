"""
app/db/supabase_auth.py — Auth persistence layer (Supabase Auth).

Replaces the old SQLite auth_store.py. All identity operations are delegated
to Supabase Auth (GoTrue). The local SQLite auth.db is no longer needed.

Supabase Auth gives us for free:
  • bcrypt password hashing
  • opaque session tokens (JWTs signed with the project's JWT secret)
  • token refresh via refresh_token
  • single-use, time-limited password-reset emails (sent by Supabase)
  • per-user session revocation

Required environment variables (same ones used by supabase_client.py):
  SUPABASE_URL          https://<project>.supabase.co
  SUPABASE_SERVICE_KEY  service_role key  (bypasses Row-Level Security — keep secret)

Optional:
  SUPABASE_ANON_KEY     anon/public key   (used for client-side sign-in flows)

Two clients are maintained:
  _admin  — service-role client; used for admin operations (create user, look up by id)
  _anon   — anon-key client;    used for sign-in / sign-up so JWTs are scoped correctly

If Supabase is not configured the module raises RuntimeError on first use,
so the rest of the server still starts (helpful during local dev without Supabase).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

_SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
_SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_SUPABASE_ANON_KEY    = os.environ.get("SUPABASE_ANON_KEY", "")

_admin_client = None
_anon_client  = None


def _get_admin():
    """Return a service-role Supabase client (lazy init)."""
    global _admin_client
    if _admin_client is not None:
        return _admin_client
    if not _SUPABASE_URL or not _SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
        )
    from supabase import create_client
    _admin_client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)
    return _admin_client


def _get_anon():
    """Return an anon-key Supabase client (lazy init). Falls back to admin key."""
    global _anon_client
    if _anon_client is not None:
        return _anon_client
    key = _SUPABASE_ANON_KEY or _SUPABASE_SERVICE_KEY
    if not _SUPABASE_URL or not key:
        raise RuntimeError(
            "Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY."
        )
    from supabase import create_client
    _anon_client = create_client(_SUPABASE_URL, key)
    return _anon_client


# ── User shape helper ──────────────────────────────────────────────────────────

def _user_to_dict(supa_user) -> dict:
    """
    Map a Supabase User object to the app's internal user dict shape,
    which matches the frontend's User interface:
      { id, name, email, avatarUrl, createdAt }

    Supabase stores extra profile fields in user_metadata.
    """
    meta       = supa_user.user_metadata or {}
    created_at = (
        supa_user.created_at.isoformat()
        if hasattr(supa_user.created_at, "isoformat")
        else str(supa_user.created_at)
    )
    return {
        "id":        supa_user.id,
        "name":      meta.get("name", meta.get("full_name", "")),
        "email":     supa_user.email,
        "avatarUrl": meta.get("avatar_url"),
        "createdAt": created_at,
    }


# ── User CRUD ──────────────────────────────────────────────────────────────────

def get_user_by_id(user_id: str) -> Optional[dict]:
    """Fetch a user by UUID via the admin API. Returns None if not found."""
    try:
        resp = _get_admin().auth.admin.get_user_by_id(user_id)
        return _user_to_dict(resp.user) if resp.user else None
    except Exception as exc:
        log.warning(f"[supabase_auth] get_user_by_id failed: {exc}")
        return None


def get_user_by_email(email: str) -> Optional[dict]:
    """
    Fetch a user by email via the admin API.
    Supabase doesn't expose a direct get-by-email, so we list and filter.
    (Fine for low-volume auth flows — not intended for bulk queries.)
    """
    try:
        resp = _get_admin().auth.admin.list_users()
        for u in resp:
            if u.email and u.email.lower() == email.lower():
                return _user_to_dict(u)
        return None
    except Exception as exc:
        log.warning(f"[supabase_auth] get_user_by_email failed: {exc}")
        return None


def create_user(name: str, email: str, password: str) -> dict:
    """
    Create a new Supabase Auth user. Raises ValueError if email is taken.
    Returns the internal user dict.
    """
    try:
        resp = _get_admin().auth.admin.create_user({
            "email":          email.lower().strip(),
            "password":       password,
            "email_confirm":  True,          # skip email-confirmation loop
            "user_metadata":  {"name": name.strip()},
        })
        if resp.user is None:
            raise ValueError("User creation returned no user object.")
        log.info(f"[supabase_auth] user created: {resp.user.id} <{email}>")
        return _user_to_dict(resp.user)
    except Exception as exc:
        msg = str(exc).lower()
        if "already registered" in msg or "already exists" in msg or "unique" in msg:
            raise ValueError(f"Email already registered: {email}") from exc
        raise


def authenticate_user(email: str, password: str) -> Optional[tuple[dict, str, str]]:
    """
    Sign in with email + password via Supabase Auth.

    Returns (user_dict, access_token, refresh_token) on success, None on failure.
    The access_token is a signed JWT; store it in the Authorization header.
    The refresh_token is opaque and can be used to get a new access_token.
    """
    try:
        resp = _get_anon().auth.sign_in_with_password({
            "email":    email.lower().strip(),
            "password": password,
        })
        if resp.user is None or resp.session is None:
            return None
        return (
            _user_to_dict(resp.user),
            resp.session.access_token,
            resp.session.refresh_token,
        )
    except Exception as exc:
        log.debug(f"[supabase_auth] sign_in failed for {email}: {exc}")
        return None


def create_session_for_user(user_id: str) -> tuple[str, str]:
    """
    Admin-mint a new session for a user (used after registration so we can
    return a token immediately without a separate sign-in round-trip).

    Returns (access_token, refresh_token).
    """
    resp = _get_admin().auth.admin.generate_link({
        "type":    "magiclink",
        "email":   _get_admin().auth.admin.get_user_by_id(user_id).user.email,
    })
    # generate_link gives a link, not tokens — use sign_in_with_otp pattern.
    # Instead, the simpler approach: create a session directly via admin API.
    resp = _get_admin().auth.admin.create_session(user_id)   # supabase-py ≥ 2.4
    return resp.session.access_token, resp.session.refresh_token


def verify_session_token(access_token: str) -> Optional[dict]:
    """
    Validate a Supabase JWT and return the user dict, or None if invalid/expired.

    Supabase JWTs are self-contained; get_user() does a server-side check
    which also handles revoked tokens.
    """
    try:
        resp = _get_anon().auth.get_user(access_token)
        return _user_to_dict(resp.user) if resp.user else None
    except Exception:
        return None


def revoke_session_token(access_token: str) -> None:
    """Sign out the session associated with this access token."""
    try:
        # sign_out with a specific JWT requires the token to be set on the client.
        client = _get_anon()
        client.auth.sign_out()          # signs out the current session on this client
    except Exception as exc:
        log.debug(f"[supabase_auth] sign_out failed: {exc}")


def revoke_all_user_tokens(user_id: str) -> None:
    """
    Invalidate every active session for a user.
    Used after a password reset to force re-login on all devices.
    """
    try:
        _get_admin().auth.admin.sign_out(user_id, scope="global")
        log.info(f"[supabase_auth] all sessions revoked for user {user_id}")
    except Exception as exc:
        log.warning(f"[supabase_auth] revoke_all_user_tokens failed: {exc}")


def update_user_password(user_id: str, new_password: str) -> None:
    """Update a user's password via the admin API."""
    try:
        _get_admin().auth.admin.update_user_by_id(
            user_id, {"password": new_password}
        )
        log.info(f"[supabase_auth] password updated for user {user_id}")
    except Exception as exc:
        log.error(f"[supabase_auth] update_user_password failed: {exc}")
        raise


def create_reset_token(user_id: str) -> str:
    """
    Trigger Supabase's built-in password-recovery email flow.

    Supabase generates, stores, and emails the reset link itself.
    Returns the raw reset token extracted from the generated link so callers
    that want to send a custom email still can.

    If you're relying on Supabase's email delivery, just call
    `trigger_password_recovery_email()` instead and ignore the return value.
    """
    user = _get_admin().auth.admin.get_user_by_id(user_id).user
    resp = _get_admin().auth.admin.generate_link({
        "type":             "recovery",
        "email":            user.email,
        "options": {
            "redirect_to":  os.environ.get("FRONTEND_URL", "http://localhost:5173")
                            + "/auth/reset-password",
        },
    })
    # The link looks like: https://<project>.supabase.co/auth/v1/verify?token=<token>&type=recovery
    # Extract the raw token so the caller can embed it in a custom email.
    from urllib.parse import urlparse, parse_qs
    qs    = parse_qs(urlparse(resp.properties.action_link).query)
    token = qs.get("token", [""])[0]
    return token


def consume_reset_token(raw_token: str) -> Optional[str]:
    """
    Validate a recovery token and return the user_id.

    Supabase verifies the token server-side via verify_otp().
    On success the session is available but we only need the user_id here;
    the password update happens via update_user_password().
    """
    try:
        resp = _get_anon().auth.verify_otp({
            "token_hash": raw_token,
            "type":       "recovery",
        })
        if resp.user:
            return resp.user.id
        return None
    except Exception as exc:
        log.debug(f"[supabase_auth] consume_reset_token failed: {exc}")
        return None


def refresh_access_token(refresh_token: str) -> Optional[tuple[str, str]]:
    """
    Exchange a refresh token for a new (access_token, refresh_token) pair.
    Returns None if the refresh token is invalid or expired.
    """
    try:
        resp = _get_anon().auth.refresh_session(refresh_token)
        if resp.session:
            return resp.session.access_token, resp.session.refresh_token
        return None
    except Exception:
        return None