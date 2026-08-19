"""
app/services/kg_agent_client.py — Client for the shared kg-agent service.

kg-agent is a separate, read-only Q&A service running on citi-condor,
backed by a Neo4j knowledge graph over a fixed 8-paper corpus (servitization,
ISO 9001, supply chain integration, the Hawthorne effect). It is NOT a
replacement for lkc_retrieval.py: that module answers "what did we say in
this session" from the session transcript; this module answers "what does
the literature say" from a static academic corpus. session_pipeline.py's
_handle_qa_sse races both and uses whichever one actually answered the
question — see the hybrid logic there.

Contract notes (verified against the service on 2026-08-17):

- Never send {"agentic": true}. The model backing kg-agent (hermes3:3b)
  can't do native tool-calling reliably; that path returns malformed
  pseudo-JSON as plain text with none of the verification fields below,
  and no HTTP error — status stays 200, so this would fail silently if
  it weren't pinned to False here.
- `passed` is *not* a quality signal on this deployment. The trust gate
  reads `source_type`/`confidence_score` node properties that don't exist
  on this graph, so trust_score is a constant 0.2 (permanently below the
  0.4 threshold) regardless of answer quality. `passed` is therefore
  almost always False. Use `faithfulness` (threshold 0.7) instead — it's
  the metric that actually reflects how well-supported the answer is.
- The literal string "The model did not produce an answer for this query
  (empty response)" is an explicit server-side sentinel (model/Ollama
  misconfiguration), not a real answer. Treated here as a failure and
  never handed back to callers.
- Empty `documents_used` means the question fell outside the 8-paper
  corpus — expected behavior, not an error. Used here as the signal that
  this wasn't actually a literature question, so the hybrid QA path
  should fall back to transcript grounding instead.
- Server-side timeout ceiling is 300s (KG_LLM_TIMEOUT=60 x up to 2 LLM
  calls x up to 2 attempts). The HTTP client timeout below matches that
  on purpose — setting it lower doesn't make failures happen faster, it
  just means the client gives up while the server keeps burning a worker
  and an Ollama slot for a request nobody's waiting on anymore. Live QA
  responsiveness is instead handled by soft_deadline_seconds in the
  caller (session_pipeline._handle_qa_sse), which races this call against
  a timer WITHOUT cancelling it — see that function's comments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import os

import httpx

log = logging.getLogger(__name__)

_EMPTY_RESPONSE_SENTINEL = "The model did not produce an answer for this query (empty response)"
_DISCLAIMER_MARKER = "\n\n[!] Unverified:"

KG_AGENT_ENABLED                          = os.environ.get("KG_AGENT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
KG_AGENT_BASE_URL                         = os.environ.get("KG_AGENT_BASE_URL", "http://100.122.56.39:8003")
KG_AGENT_REQUEST_TIMEOUT_SECONDS          = float(os.environ.get("KG_AGENT_REQUEST_TIMEOUT_SECONDS", "300.0"))  # server-side ceiling — do not lower
KG_AGENT_SOFT_DEADLINE_SECONDS            = float(os.environ.get("KG_AGENT_SOFT_DEADLINE_SECONDS", "6.0"))
KG_AGENT_FAITHFULNESS_THRESHOLD           = float(os.environ.get("KG_AGENT_FAITHFULNESS_THRESHOLD", "0.7"))
KG_AGENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(os.environ.get("KG_AGENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30.0"))

_http_client: Optional[httpx.AsyncClient] = None
_health_client: Optional[httpx.AsyncClient] = None

# Circuit breaker: opened on any failed /query or /health call, cleared on
# success. Avoids spending the soft-deadline wait (and a slice of the
# server's 300s budget) probing a service we already know is down — the
# kg-agent doc is explicit about this: on a "degraded" health check,
# report it, don't retry repeatedly.
_last_failure_at: float = 0.0
KG_AGENT_AVAILABLE = True


@dataclass
class KgAgentAnswer:
    answer:                    str                  # disclaimer suffix stripped — safe to speak/display as the reply
    raw_answer:                str                  # exactly as returned by kg-agent, disclaimer included
    faithfulness:               float
    overall_confidence:         float
    temporal_validity_status:   str
    documents_used:             list[dict] = field(default_factory=list)
    sources_used:                list[dict] = field(default_factory=list)
    disclaimer:                  str = ""
    strategy:                    str = ""
    retries:                     int = 0

    @property
    def in_corpus(self) -> bool:
        """False means the question fell outside the 8-paper corpus."""
        return bool(self.documents_used)


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=KG_AGENT_REQUEST_TIMEOUT_SECONDS)
    return _http_client


def _get_health_client() -> httpx.AsyncClient:
    global _health_client
    if _health_client is None:
        _health_client = httpx.AsyncClient(timeout=5.0)
    return _health_client


def _circuit_open() -> bool:
    if _last_failure_at == 0.0:
        return False
    return (time.time() - _last_failure_at) < KG_AGENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS


def _mark_failure(reason: str) -> None:
    global _last_failure_at, KG_AGENT_AVAILABLE
    _last_failure_at = time.time()
    if KG_AGENT_AVAILABLE:
        log.warning(
            f"[kg-agent] marking unavailable ({reason}) — cooling down "
            f"{KG_AGENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS:.0f}s before retrying"
        )
    KG_AGENT_AVAILABLE = False


def _mark_success() -> None:
    global KG_AGENT_AVAILABLE
    if not KG_AGENT_AVAILABLE:
        log.info("[kg-agent] service recovered")
    KG_AGENT_AVAILABLE = True


async def health() -> Optional[dict]:
    """GET /health. Cheap — safe to poll. Returns None on any failure."""
    if not KG_AGENT_ENABLED:
        return None
    try:
        client = _get_health_client()
        resp = await client.get(f"{KG_AGENT_BASE_URL}/health")
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "degraded":
            _mark_failure("neo4j degraded")
        else:
            _mark_success()
        return data
    except Exception as exc:
        _mark_failure(f"health check failed: {exc}")
        return None


async def warmup() -> None:
    """Call once at startup, same spirit as lkc_retrieval.warmup()."""
    if not KG_AGENT_ENABLED:
        log.info("[kg-agent] disabled via config — skipping warmup")
        return
    data = await health()
    if data is None:
        log.warning(
            f"[kg-agent] unreachable at {KG_AGENT_BASE_URL} at startup "
            f"(check Tailscale / citi-condor) — hybrid QA will fall back to "
            f"transcript-only until it recovers"
        )
    elif data.get("status") != "ok":
        log.warning(f"[kg-agent] unhealthy at startup: {data}")
    else:
        log.info(f"[kg-agent] healthy at {KG_AGENT_BASE_URL}")


async def query(question: str) -> Optional[KgAgentAnswer]:
    """
    POST /query. Returns None on any failure, on the explicit empty-answer
    sentinel, or when kg-agent is disabled/circuit-broken. Callers should
    treat None exactly like "no literature answer available" and fall back
    to transcript-grounded QA. Never raises.
    """
    if not KG_AGENT_ENABLED:
        return None
    if _circuit_open():
        return None

    try:
        client = _get_http_client()
        resp = await client.post(
            f"{KG_AGENT_BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            # agentic explicitly pinned False — see module docstring.
            json={"query": question, "agentic": False},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _mark_failure(f"query failed: {exc}")
        return None

    raw_answer = (data.get("answer") or "").strip()
    if not raw_answer or raw_answer == _EMPTY_RESPONSE_SENTINEL:
        log.warning(
            f"[kg-agent] empty-answer sentinel for query {question!r} — "
            f"server-side issue (model/Ollama), treating as no answer"
        )
        _mark_success()  # the service itself responded fine; the model didn't
        return None

    clean_answer = (
        raw_answer.split(_DISCLAIMER_MARKER)[0].rstrip()
        if _DISCLAIMER_MARKER in raw_answer else raw_answer
    )

    _mark_success()
    return KgAgentAnswer(
        answer=clean_answer,
        raw_answer=raw_answer,
        faithfulness=float(data.get("faithfulness", 0.0)),
        overall_confidence=float(data.get("overall_confidence", 0.0)),
        temporal_validity_status=data.get("temporal_validity_status", "VALID"),
        documents_used=data.get("documents_used", []) or [],
        sources_used=data.get("sources_used", []) or [],
        disclaimer=data.get("disclaimer", "") or "",
        strategy=data.get("strategy", ""),
        retries=int(data.get("retries", 0)),
    )


def stats() -> dict:
    return {
        "enabled":            KG_AGENT_ENABLED,
        "base_url":           KG_AGENT_BASE_URL,
        "available":          KG_AGENT_AVAILABLE,
        "circuit_open":       _circuit_open(),
        "faithfulness_threshold": KG_AGENT_FAITHFULNESS_THRESHOLD,
    }