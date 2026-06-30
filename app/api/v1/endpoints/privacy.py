"""
app/api/v1/endpoints/privacy.py — Privacy & Consent endpoints.

GET    /privacy/status              — registry overview + default policy
POST   /privacy/consent             — register / update local consent only
DELETE /privacy/consent/{speaker}   — revoke consent
POST   /privacy/consent/sync        — dual-write: local + Supabase
"""

from fastapi import APIRouter
from app.schemas import ConsentRequest, ConsentSyncRequest
from app.services import privacy as _privacy
from app.db import supabase_client

router = APIRouter(prefix="/privacy", tags=["privacy"])


@router.get("/status")
async def privacy_status():
    return {
        "default_consent": _privacy.DEFAULT_CONSENT,
        "registry":        _privacy.all_consents(),
    }


@router.post("/consent")
async def post_consent(req: ConsentRequest):
    """Register or update consent (local registry only)."""
    entry = _privacy.register_consent(req.speaker, req.consented, req.real_name)
    return {"speaker": req.speaker, **entry}


@router.post("/consent/sync")
async def sync_consent(req: ConsentSyncRequest):
    """Dual-write: local consent.json + Supabase consent_registry table."""
    entry = _privacy.register_consent(req.speaker, req.consented, req.real_name)
    await supabase_client.upsert_consent(req.speaker, req.consented, req.real_name)
    return {"speaker": req.speaker, **entry}


@router.delete("/consent/{speaker}")
async def delete_consent(speaker: str):
    removed = _privacy.revoke_consent(speaker)
    return {"speaker": speaker, "removed": removed}
