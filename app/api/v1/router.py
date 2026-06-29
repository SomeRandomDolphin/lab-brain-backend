"""
app/api/v1/router.py — Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.livekit import router as livekit_router, sse_router
from app.api.v1.endpoints.privacy import router as privacy_router
from app.api.v1.endpoints.lkc import router as lkc_router
from app.api.v1.endpoints.capture import (
    capture_router,
    agent_router,
    ner_router,
    retrieval_router,
)
from app.api.v1.endpoints.sessions import router as sessions_router
from app.api.v1.endpoints.supabase import router as supabase_router
from app.api.v1.endpoints.websockets import router as ws_router

api_router = APIRouter()

# Auth comes first so /auth/... routes take priority
api_router.include_router(auth_router)

api_router.include_router(livekit_router)
api_router.include_router(sse_router)
api_router.include_router(privacy_router)
api_router.include_router(lkc_router)
api_router.include_router(capture_router)
api_router.include_router(agent_router)
api_router.include_router(ner_router)
api_router.include_router(retrieval_router)
api_router.include_router(sessions_router)
api_router.include_router(supabase_router)
api_router.include_router(ws_router)
