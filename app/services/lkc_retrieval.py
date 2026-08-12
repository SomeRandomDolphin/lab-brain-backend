"""
app/services/lkc_retrieval.py — Dense LKC Retrieval Service.

Sources documents from the SQLite LKC graph (supports cross-session Q&A).
Embeddings are served by the same local Ollama instance the app already
runs for dialogue/vision (see config.json → local_llm), instead of a
separately-downloaded sentence-transformers model. This removes the
`sentence-transformers==3.3.1` pin and the Windows/FFmpeg dependency
issues that pin existed for entirely.

Primary:  Ollama embedding model, via /api/embed
          (config: local_llm.embedding_model, default "qwen3-embedding:0.6b")
Fallback: TF-IDF (scikit-learn), used only if Ollama is unreachable —
          e.g. the container hasn't finished starting, the host is down,
          or embedding_model was never pulled.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from app.db import lkc_graph

log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"
_DEFAULT_EMBED_MODEL     = "qwen3-embedding:0.6b"


def _load_llm_config() -> tuple[str, str]:
    """
    Resolve (ollama_base_url, embedding_model) from config.json, walking
    upward from this file's location rather than assuming a fixed relative
    path — keeps this working if the service module ever moves. Falls back
    to hardcoded defaults on a missing or malformed config.json so a bad
    config degrades gracefully at import time instead of crashing the app.
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "config.json"
        if not candidate.is_file():
            continue
        try:
            cfg = json.loads(candidate.read_text())
            llm_cfg  = cfg.get("local_llm", {})
            base_url = llm_cfg.get("base_url", _DEFAULT_OLLAMA_BASE_URL).rstrip("/")
            # config.json's base_url is the OpenAI-compat path (".../v1")
            # used by the dialogue/vision clients. Ollama's native
            # embeddings endpoint lives one level up, at ".../api/embed" —
            # strip a trailing "/v1" if present rather than requiring a
            # second base_url entry in config.json to stay in sync.
            if base_url.endswith("/v1"):
                base_url = base_url[: -len("/v1")]
            model = llm_cfg.get("embedding_model", _DEFAULT_EMBED_MODEL)
            return base_url, model
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"[retrieval] could not parse {candidate}: {exc} — using defaults.")
            break
    return _DEFAULT_OLLAMA_BASE_URL, _DEFAULT_EMBED_MODEL


_OLLAMA_BASE_URL, _EMBED_MODEL = _load_llm_config()
_OLLAMA_EMBED_URL = f"{_OLLAMA_BASE_URL}/api/embed"

# Optimistic until a real call proves otherwise — unlike the old SBERT
# import check, we can't know Ollama is reachable until we actually ask it.
OLLAMA_AVAILABLE = True

SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    log.warning("[retrieval] scikit-learn not installed — TF-IDF fallback disabled.")

_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Lazily-created, reused client — avoids a new connection per QA turn."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def _ollama_embed(texts: list[str]) -> Optional[np.ndarray]:
    """
    Batch-embed `texts` via Ollama's native /api/embed endpoint, returning
    L2-normalized float32 vectors (shape: len(texts) x dim), or None if
    Ollama is unreachable, errors, or returns something unusable — the
    caller falls back to TF-IDF in that case rather than crashing.

    Normalization is done manually rather than trusted from the API:
    some serving backends behind Ollama (e.g. llama-server) don't apply
    --embd-normalize automatically, so assuming "the API already
    normalizes it" is fragile across backend/version changes.
    """
    global OLLAMA_AVAILABLE
    try:
        client = _get_http_client()
        resp = await client.post(
            _OLLAMA_EMBED_URL,
            json={"model": _EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        raw = np.array(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        if not OLLAMA_AVAILABLE:
            log.info(f"[retrieval] Ollama embedding endpoint recovered ({_EMBED_MODEL}).")
        OLLAMA_AVAILABLE = True
        return raw / norms
    except Exception as exc:
        if OLLAMA_AVAILABLE:  # log once when it goes down, not on every call
            log.warning(
                f"[retrieval] Ollama embed call failed ({exc}) — "
                + ("falling back to TF-IDF until it recovers." if SKLEARN_AVAILABLE
                   else "retrieval disabled until it recovers (sklearn also missing).")
            )
        OLLAMA_AVAILABLE = False
        return None


def _fit_tfidf(corpus: list[str]):
    """CPU-bound; called via run_in_executor from _IndexEntry.build()."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


async def warmup() -> None:
    """
    Fire a trivial embed call at startup so the embedding model is loaded
    and resident on the Ollama side before the first real QA/summon
    request needs it — same intent as start_services.sh's residency check
    for vision_model/dialogue_model, just from the app side.

    NOTE — behavior change from the sentence-transformers version: this is
    now a small async HTTP call, not a ~420MB blocking model load. Call it
    directly with `await warmup()` from the FastAPI startup hook instead of
    wrapping it in run_in_executor; wrapping an async function in
    run_in_executor won't await it correctly.
    """
    embeddings = await _ollama_embed(["warmup"])
    if embeddings is None:
        log.warning(
            f"[retrieval] warmup: Ollama embedding endpoint unreachable at "
            f"{_OLLAMA_EMBED_URL} (model={_EMBED_MODEL}). "
            + ("Falling back to TF-IDF." if SKLEARN_AVAILABLE
               else "Retrieval will be disabled until it recovers.")
        )


class _IndexEntry:
    def __init__(self, records: list[dict]) -> None:
        self.records      = records
        self.record_count = len(records)
        self.corpus: list[str] = []
        self._embeddings = None   # np.ndarray | None — Ollama path
        self._vectorizer  = None  # TfidfVectorizer | None — fallback path
        self._tfidf_mat   = None

    async def build(self) -> None:
        if not self.records:
            return
        self.corpus = [
            f"{r.get('speaker', '')} {r.get('text', '')}".strip()
            for r in self.records
        ]
        embeddings = await _ollama_embed(self.corpus)
        if embeddings is not None:
            self._embeddings = embeddings
        elif SKLEARN_AVAILABLE:
            # TF-IDF fit over the whole corpus is CPU-bound and, on a
            # long session, runs on nearly every QA turn (see the cache
            # note in _get_session_entry below) — keep it off the event
            # loop the same way the old encode() call was.
            loop = asyncio.get_event_loop()
            self._vectorizer, self._tfidf_mat = await loop.run_in_executor(
                None, _fit_tfidf, self.corpus
            )

    async def search(self, question: str, top_k: int) -> str:
        if self._embeddings is not None:
            q_emb = await _ollama_embed([question])
            if q_emb is None:
                # Ollama went down mid-session after a successful build().
                # Nothing sane to return here — the cached dense index
                # can't be queried without an embedding for `question`,
                # and rebuilding as TF-IDF mid-session would silently
                # change result semantics. Surface as empty; caller
                # already treats "" as "no hits".
                return ""
            scores = (self._embeddings @ q_emb[0].T).flatten()
            floor  = 0.20
        elif SKLEARN_AVAILABLE and self._tfidf_mat is not None:
            # Single-string transform is cheap enough to run inline on
            # the event loop, unlike the full-corpus fit in build().
            q_vec  = self._vectorizer.transform([question])
            scores = _sk_cosine(q_vec, self._tfidf_mat).flatten()
            floor  = 0.05
        else:
            return ""

        top_idx = scores.argsort()[::-1][:top_k]
        hits: list[str] = []
        for idx in top_idx:
            if float(scores[idx]) < floor:
                continue
            r  = self.records[idx]
            ts = r.get("timestamp_iso", "")[:19].replace("T", " ")
            hits.append(f"[{ts}] {r.get('speaker','?')}: {r.get('text','')}")
        return "\n".join(hits)


class LKCRetriever:
    def __init__(self) -> None:
        self._sessions: dict[str, _IndexEntry] = {}
        self._global: Optional[_IndexEntry] = None
        self._global_record_count: int = 0

    async def query(
        self,
        question: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
    ) -> str:
        if session_id is not None:
            entry = await self._get_session_entry(session_id)
        else:
            entry = await self._get_global_entry()
        if entry is None or not entry.records:
            return ""
        # entry.search() is now natively async I/O (an HTTP call to
        # Ollama), not a blocking local computation — so it no longer
        # needs the run_in_executor wrapper the sentence-transformers
        # version required to keep encode() off the event loop. The
        # TF-IDF fallback path inside search() is a single-string
        # transform and stays cheap enough to run inline; the expensive
        # full-corpus TF-IDF fit is offloaded inside _IndexEntry.build().
        return await entry.search(question, top_k)

    async def _get_session_entry(self, session_id: str) -> Optional[_IndexEntry]:
        records = await lkc_graph.session_text_corpus(session_id)
        cached  = self._sessions.get(session_id)
        if cached is None or len(records) != cached.record_count:
            # NOTE: this cache check invalidates on ANY change in record
            # count — and a new transcript record lands after nearly every
            # segment, so in practice this re-embeds the ENTIRE session
            # transcript from scratch on almost every single QA call, not
            # just the new records. With embeddings now coming from a
            # network call instead of a local model, this trades local
            # CPU cost for repeated round-trips to Ollama — batched in a
            # single /api/embed call per rebuild, so it's still one
            # request rather than one per record, but it's still O(whole
            # transcript) per QA call. Not rewritten to be incremental
            # here (that's a bigger change); flagging again since the
            # cost profile shifted.
            entry = _IndexEntry(records)
            await entry.build()
            self._sessions[session_id] = entry
        return self._sessions.get(session_id)

    async def _get_global_entry(self) -> Optional[_IndexEntry]:
        all_records = await lkc_graph.read_lkc(record_type="transcript")
        if len(all_records) == self._global_record_count and self._global is not None:
            return self._global
        lightweight = [
            {"timestamp_iso": r.get("timestamp_iso", ""),
             "speaker": r.get("speaker", ""),
             "text": r.get("text", "")}
            for r in all_records
            if r.get("text")
        ]
        self._global = _IndexEntry(lightweight)
        await self._global.build()
        self._global_record_count = len(all_records)
        return self._global

    def invalidate(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._global = None

    def stats(self) -> dict:
        backend = (
            _EMBED_MODEL if OLLAMA_AVAILABLE
            else ("tfidf" if SKLEARN_AVAILABLE else "none")
        )
        return {
            "backend":           backend,
            "ollama_available":  OLLAMA_AVAILABLE,
            "ollama_embed_url":  _OLLAMA_EMBED_URL,
            "sklearn_available": SKLEARN_AVAILABLE,
            "cached_sessions":   len(self._sessions),
            "global_index_size": self._global_record_count,
        }


_retriever: Optional[LKCRetriever] = None


def get_retriever() -> LKCRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LKCRetriever()
    return _retriever