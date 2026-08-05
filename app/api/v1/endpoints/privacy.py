"""
app/api/v1/endpoints/privacy.py — Privacy & Consent endpoints.

GET    /privacy/status              — registry overview + default policy (any authenticated user)
POST   /privacy/consent             — register / update local consent only (any authenticated user)
DELETE /privacy/consent/{speaker}   — revoke consent (any authenticated user)
POST   /privacy/consent/sync        — dual-write: local + Supabase (owner or participant of the session)

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
"""

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user, require_session_access
from app.schemas import ConsentRequest, ConsentSyncRequest
from app.services import privacy as _privacy
from app.db import supabase_client

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