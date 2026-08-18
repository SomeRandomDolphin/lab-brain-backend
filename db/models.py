"""
app/db/models.py — SQLAlchemy ORM models for the Lab Brain Supabase schema.

All tables mirror the Postgres schema on your Supabase project.
Import these models from any db module; never define table structure elsewhere.

Engine / session setup lives in supabase_client.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id                = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id        = Column(String, nullable=False, unique=True, index=True)
    # Multi-tenancy: the account that created (owns) this session. NOT NULL —
    # every session is required to be created by an authenticated user as of
    # migration 0007. FK to auth.users(id) ON DELETE CASCADE.
    #
    # PG_UUID(as_uuid=False): the actual Postgres column is `uuid`, not
    # `varchar` — using plain String here made SQLAlchemy bind every
    # parameter as ::VARCHAR, which Postgres refuses to compare/insert
    # against a uuid column (asyncpg.UndefinedFunctionError /
    # DatatypeMismatchError). as_uuid=False keeps the Python-side value a
    # plain str (matching current_user["id"] everywhere it's passed in) —
    # only the wire-level bind type changes, no call sites need to change.
    user_id           = Column(PG_UUID(as_uuid=False), nullable=False, index=True)
    host_identity     = Column(String, nullable=False, default="browser-user")
    started_at        = Column(DateTime(timezone=True), nullable=False)
    ended_at          = Column(DateTime(timezone=True), nullable=True)
    livekit_room_sid  = Column(String, nullable=True)
    metadata_         = Column("metadata", JSON, nullable=False, default=dict, key="metadata_")
    updated_at        = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class SessionParticipant(Base):
    """
    Authenticated users who joined a session via GET /livekit/token but did
    not create it. A row is inserted on every successful join. Used by
    api.deps.require_session_access to grant view/join-level access to
    anyone who is owner OR participant, as distinct from owner-only actions
    (delete/manage) gated by require_session_owner.
    """
    __tablename__ = "session_participants"

    session_id = Column(String, primary_key=True)
    # Same fix as Session.user_id above — real column is uuid, not varchar.
    user_id    = Column(PG_UUID(as_uuid=False), primary_key=True)
    joined_at  = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id       = Column(String, nullable=False, index=True)
    segment_index    = Column(Integer, nullable=False, default=0)
    speaker          = Column(String, nullable=False)
    text             = Column(Text, nullable=False)
    language         = Column(String, nullable=True)
    mode             = Column(String, nullable=True)
    timestamp_iso    = Column(String, nullable=False)
    timestamp_unix   = Column(Float, nullable=False, index=True)
    tags             = Column(JSON, nullable=False, default=dict)
    word_timestamps  = Column(JSON, nullable=False, default=list)
    asr_latency_ms   = Column(Integer, nullable=False, default=0)
    e2e_latency_ms   = Column(Integer, nullable=False, default=0)
    created_at       = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AgentReply(Base):
    __tablename__ = "agent_replies"

    id             = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id     = Column(String, nullable=False, index=True)
    text           = Column(Text, nullable=False)
    mode           = Column(String, nullable=True)
    timestamp_iso  = Column(String, nullable=False)
    timestamp_unix = Column(Float, nullable=False, index=True)
    grounded       = Column(Boolean, nullable=False, default=False)
    lkc_context    = Column(Text, nullable=False, default="")
    created_at     = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class VisionFrame(Base):
    __tablename__ = "vision_frames"

    id                 = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id         = Column(String, nullable=False, index=True)
    timestamp_iso      = Column(String, nullable=False)
    timestamp_unix     = Column(Float, nullable=False, index=True)
    scene_summary      = Column(Text, nullable=False)
    present_speakers   = Column(JSON, nullable=False, default=list)
    engagement_cues    = Column(JSON, nullable=False, default=dict)
    environment_state  = Column(JSON, nullable=False, default=dict)
    latency_ms         = Column(Integer, nullable=False, default=0)
    created_at         = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id  = Column(String, nullable=False, unique=True, index=True)
    summary_md  = Column(Text, nullable=False)
    tags        = Column(JSON, nullable=False, default=dict)
    created_at  = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at  = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class EvalMetrics(Base):
    __tablename__ = "eval_metrics"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id  = Column(String, nullable=False, unique=True, index=True)
    snapshot    = Column(JSON, nullable=False, default=dict)
    snapshot_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at  = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class ConsentRegistry(Base):
    __tablename__ = "consent_registry"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    # Composite PK as of migration 0009 — speaker_label alone collided across
    # unrelated sessions (diarization labels like "Person A" are not globally
    # unique). `id` remains a separate unique surrogate, untouched by this.
    session_id    = Column(String, primary_key=True)
    speaker_label = Column(String, primary_key=True, index=True)
    consented     = Column(Boolean, nullable=False)
    real_name     = Column(String, nullable=True)
    updated_at    = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class LkcRecord(Base):
    """
    Supabase-backed replacement for the old SQLite lkc_records table.
    Stores the full LKC event graph — transcripts, vision frames, agent replies,
    and session summaries as a unified append-only event log.
    """
    __tablename__ = "lkc_records"

    id             = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id     = Column(String, nullable=False, index=True)
    record_type    = Column(String, nullable=False, index=True)
    timestamp_unix = Column(Float, nullable=False, index=True)
    timestamp_iso  = Column(String, nullable=False)
    speaker        = Column(String, nullable=True)
    text           = Column(Text, nullable=True)
    mode           = Column(String, nullable=True)
    language       = Column(String, nullable=True)
    payload        = Column(JSON, nullable=False)