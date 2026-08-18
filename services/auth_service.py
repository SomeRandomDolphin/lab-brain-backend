"""
app/services/auth_service.py — Auth business logic helpers.

With Supabase Auth, the platform handles:
  • password hashing (bcrypt)
  • session token issuance and verification (JWTs)
  • password-reset email delivery (via your project's SMTP settings)

This module is kept for:
  1. Custom-email override — if you'd rather send reset emails via your own
     SMTP server instead of Supabase's built-in delivery, call
     send_reset_email() after generating a recovery link with
     supabase_auth.create_reset_token().
  2. A convenient place for future auth-adjacent logic (rate limiting,
     audit logging, post-login hooks, etc.).

Environment variables (only needed if using the custom SMTP override):
  SMTP_HOST     (default: "" — disables custom sending; Supabase sends instead)
  SMTP_PORT     (default: 587)
  SMTP_USER     (optional)
  SMTP_PASSWORD (optional)
  SMTP_FROM     (default: noreply@labbrain.local)
  FRONTEND_URL  (default: http://localhost:5173)
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
SMTP_HOST    = os.environ.get("SMTP_HOST", "")
SMTP_PORT    = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER    = os.environ.get("SMTP_USER", "")
SMTP_PASS    = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM    = os.environ.get("SMTP_FROM", "noreply@labbrain.local")


def build_reset_link(raw_token: str) -> str:
    return f"{FRONTEND_URL}/auth/reset-password?token={raw_token}"


def send_reset_email(to_email: str, to_name: str, raw_token: str) -> bool:
    """
    Send a password-reset email using custom SMTP.

    Only call this if you want to override Supabase's built-in email delivery.
    If SMTP_HOST is not set, this function just logs the link (dev mode) and
    returns True — Supabase will have already sent the email via its own SMTP.

    Returns True on success (or dev-mode skip), False on SMTP error.
    """
    link = build_reset_link(raw_token)

    if not SMTP_HOST:
        log.debug(
            f"[auth_service] Custom SMTP not configured. "
            f"Supabase is handling email delivery for {to_email}. "
            f"Dev reset link: {link}"
        )
        return True

    subject  = "Lab Brain — Password Reset"
    body_txt = (
        f"Hi {to_name},\n\n"
        f"Click the link below to reset your password. "
        f"It expires in 1 hour and can only be used once.\n\n"
        f"  {link}\n\n"
        f"If you did not request a password reset, you can safely ignore this email.\n\n"
        f"— Lab Brain"
    )
    body_html = f"""
<html><body>
<p>Hi {to_name},</p>
<p>Click the button below to reset your Lab Brain password.
   This link expires in <strong>1 hour</strong> and can only be used once.</p>
<p><a href="{link}" style="background:#6366f1;color:#fff;padding:10px 20px;
   border-radius:6px;text-decoration:none;font-weight:600;">Reset Password</a></p>
<p>Or paste this URL into your browser:<br><code>{link}</code></p>
<p>If you didn't request a reset, ignore this email.</p>
<p>— Lab Brain</p>
</body></html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    msg.attach(MIMEText(body_txt,  "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if SMTP_PORT in (587, 2587):
                server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        log.info(f"[auth_service] custom reset email sent to {to_email}")
        return True
    except Exception as exc:
        log.error(f"[auth_service] failed to send reset email to {to_email}: {exc}")
        return False