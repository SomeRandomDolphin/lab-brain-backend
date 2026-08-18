"""
app/api/v1/endpoints/privacy.py — Privacy & Consent endpoints.

GET    /privacy/status              — registry overview + default policy (any authenticated user)
POST   /privacy/consent             — register / update local consent only (any authenticated user)
DELETE /privacy/consent/{speaker}   — revoke consent (any authenticated user)
POST   /privacy/consent/sync        — dual-write: local + Supabase (owner or participant of the session)
POST   /privacy/tos-consent         — account-level privacy-screen decision (any authenticated user)

Confirmed via the consent_registry scoping fix (migration 0009): consent is
now tied to a session_id, since a diarization-assigned label like "Person A"
is not a stable identity across unrelated sessions. sync_consent therefore
requires session access, matching every other per-session route.

The local-only routes (/consent, DELETE /consent/{speaker}) still operate
on the process-wide in-memory registry in app.services.privacy and are not
session-scoped — that's a pre-existing property of that service, unrelated
to this fix. They're gated to "logged in" rather than left fully open, in
keeping with the rest of this migration, but not tied to a specific
session's ownership since they aren't part of the tenancy model.

/tos-consent is a distinct thing from the above: it's the account-level
default (from the dashboard's first-login modal), not a per-session
diarization label. It's keyed by current_user["id"] — the Supabase user
id — rather than a display name, specifically to avoid two people sharing
a display name colliding in _privacy._registry. This is guaranteed to
match participant.identity as seen by session_pipeline.py's
check_consent(identity) call: app.api.v1.endpoints.livekit's create_room
and get_token both pass identity=current_user["id"] to create_token()
(display names go through the separate display_name param instead) — see
the fix there for why this didn't hold before.
"""

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user, require_session_access
from app.schemas.privacy import ConsentRequest, ConsentSyncRequest
from app.schemas.auth import TosConsentRequest
from app.services import privacy as _privacy
from app.db import supabase_client, supabase_auth

router = APIRouter(prefix="/privacy", tags=["privacy"])


@router.get("/status")
async def privacy_status(_current_user: dict = Depends(get_current_user)):
    return {
        "default_consent": _privacy.DEFAULT_CONSENT,
        "registry":        _privacy.all_consents(),
    }


@router.post("/consent")
async def post_consent(req: ConsentRequest, _current_user: dict = Depends(get_current_user)):
    """Register or update consent (local registry only)."""
    entry = _privacy.register_consent(req.speaker, req.consented, req.real_name)
    return {"speaker": req.speaker, **entry}


@router.post("/consent/sync")
async def sync_consent(req: ConsentSyncRequest, _current_user: dict = Depends(get_current_user)):
    """Dual-write: local consent.json + Supabase consent_registry table."""
    # Manual access check (session_id is in the body, not a path param) —
    # same owner-or-participant rule as every other per-session route.
    owner, participants = await supabase_client.get_session_access(req.session_id)
    if owner is None or (_current_user["id"] != owner and _current_user["id"] not in participants):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found.")

    entry = _privacy.register_consent(req.speaker, req.consented, req.real_name)
    await supabase_client.upsert_consent(req.session_id, req.speaker, req.consented, req.real_name)
    return {"speaker": req.speaker, "session_id": req.session_id, **entry}


@router.delete("/consent/{speaker}")
async def delete_consent(speaker: str, _current_user: dict = Depends(get_current_user)):
    removed = _privacy.revoke_consent(speaker)
    return {"speaker": speaker, "removed": removed}


@router.post("/tos-consent")
async def set_tos_consent(req: TosConsentRequest, current_user: dict = Depends(get_current_user)):
    """
    Account-level privacy decision, captured once at first login (see the
    dashboard's PrivacyConsentModal).

    accepted=True  -> identified / not redacted for this account going forward
    accepted=False -> privacy screen (redaction + anonymized face labels)
                       stays on, the safe default

    Writes twice, deliberately: Supabase user_metadata is the source of
    truth for whether the frontend shows the modal again (tosAccepted is
    None until this is called); _privacy._registry is the runtime gate
    session_pipeline.py / websockets.py already read via check_consent().
    Keyed by the same display-name string livekit_rooms.get_known_identity()
    resolves to (set via create_token()'s display_name at room-join time),
    so no session-specific wiring is needed — this applies from the very
    next session onward.
    """
    updated_user = supabase_auth.set_tos_consent(current_user["id"], req.accepted)
    _privacy.register_consent(
        speaker_label=current_user["id"],
        consented=req.accepted,
        real_name=current_user["name"],
    )
    return {"accepted": req.accepted, "tosAcceptedAt": updated_user["tosAcceptedAt"]}