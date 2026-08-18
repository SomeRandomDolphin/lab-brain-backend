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
import re
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

# Matches "last/previous/prior/most recent meeting", "what happened last
# time", "recap the last meeting", etc. Deliberately conservative (doesn't
# try to catch every phrasing) — a false negative just falls back to normal
# semantic ranking, which is the existing behaviour; a false positive would
# override semantic relevance with recency for a query that didn't want
# that, which is the worse failure mode, so this stays narrow rather than
# broad.
_RECENCY_QUERY_RE = re.compile(
    r"\b(last|previous|prior|most recent)\b.{0,15}\bmeeting\b"
    r"|\bwhat happened\b.{0,15}\b(last time|previously)\b",
    re.IGNORECASE,
)


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
        # A session_summary record's raw payload has a `summary` key, not
        # `text` — without this fallback, every summary record embeds as
        # an empty string and is functionally invisible to search
        # regardless of whether it's included in self.records at all.
        self.corpus = [
            f"{r.get('speaker', '')} {r.get('text') or r.get('summary') or ''}".strip()
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

    async def search(
        self, question: str, top_k: int, exclude_session_id: Optional[str] = None
    ) -> str:
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

        top_idx = list(scores.argsort()[::-1][:top_k])

        # Past QA turns get persisted as ordinary transcript records too —
        # so a question like "what's the summary of the last meeting?" can
        # score higher against OTHER PEOPLE'S PAST INSTANCES OF THAT SAME
        # QUESTION than against an actual session_summary record, since a
        # near-duplicate question is closer in embedding space than a long
        # markdown answer is. That crowds every summary out of top_k even
        # when a clearly-relevant one exists (score >= floor) — reproduced
        # directly: a "recap"-style query returned 4/4 hits that were other
        # transcript lines of people asking the same question, zero actual
        # summaries, even though a matching summary was in the corpus and
        # scored above floor. Guarantee it a slot instead of leaving this
        # purely to raw cosine rank.
        summary_positions = [
            i for i, r in enumerate(self.records)
            if r.get("type") == "session_summary"
            and (exclude_session_id is None or r.get("session_id") != exclude_session_id)
        ]
        if summary_positions:
            if _RECENCY_QUERY_RE.search(question):
                # "Last meeting" is a RECENCY request, not a topical one —
                # cosine similarity has no notion of "most recent"; it just
                # picks whichever summary happens to embed closest to the
                # query text, which among a pile of near-identical "not
                # enough content" stub summaries is close to arbitrary
                # (reproduced directly: it picked a random all-None fallback
                # summary from days earlier over the actual last session).
                # timestamp_iso sorts correctly as a plain string (ISO 8601),
                # so pick by max timestamp instead of max score here, and
                # skip the floor check entirely — recency intent means we
                # want the last summary regardless of how it happens to
                # embed against the question text.
                best_summary_idx = max(
                    summary_positions, key=lambda i: self.records[i].get("timestamp_iso", "")
                )
            else:
                best_summary_idx = max(summary_positions, key=lambda i: scores[i])
                if float(scores[best_summary_idx]) < floor:
                    best_summary_idx = None
            if best_summary_idx is not None and best_summary_idx not in top_idx:
                top_idx[-1] = best_summary_idx
                # Keep the guaranteed slot in score order with the rest so
                # a strong summary still reads before weaker general hits.
                top_idx.sort(key=lambda i: -scores[i])

        hits: list[str] = []
        for idx in top_idx:
            r = self.records[idx]
            is_recency_summary = (
                r.get("type") == "session_summary" and _RECENCY_QUERY_RE.search(question)
            )
            if float(scores[idx]) < floor and not is_recency_summary:
                continue
            ts = r.get("timestamp_iso", "")[:19].replace("T", " ")
            body = r.get("text") or r.get("summary") or ""
            label = "summary" if r.get("type") == "session_summary" or not r.get("text") else r.get("speaker", "?")
            hits.append(f"[{ts}] {label}: {body}")
        return "\n".join(hits)


class LKCRetriever:
    def __init__(self) -> None:
        self._sessions: dict[str, _IndexEntry] = {}
        # Keyed by user_id. Scopes retrieval to one user's own accessible
        # sessions (owned + participated-in, per supabase_client.get_sessions)
        # instead of either a single session or every session in the table —
        # this is what lets the agent draw on a user's past meetings, not
        # just the one currently live. See _get_user_entry below.
        self._users: dict[str, _IndexEntry] = {}
        self._global: Optional[_IndexEntry] = None
        self._global_record_count: int = 0

    async def query(
        self,
        question: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_session_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Scoping precedence:
          1. user_id + user_session_ids given → search across all of that
             user's accessible sessions (owned + participated-in). This is
             a superset of any single session of theirs, including the
             current one — the caller is expected to include the live
             session_id in user_session_ids, not pass both scopes at once.
          2. session_id alone → single-session scope (unchanged behaviour
             for any caller that doesn't have a user_id to resolve).
          3. neither → global scope, across every session in the table
             regardless of owner. Unscoped by user; only appropriate for
             an org-wide or admin context, not a normal per-user QA turn.
        """
        if user_id is not None and user_session_ids is not None:
            entry = await self._get_user_entry(user_id, user_session_ids)
            exclude_id = session_id  # don't let "last meeting" pick the still-live current session
        elif session_id is not None:
            entry = await self._get_session_entry(session_id)
            exclude_id = None  # single-session scope — the current session's own summary is all there is
        else:
            entry = await self._get_global_entry()
            exclude_id = session_id
        if entry is None or not entry.records:
            return ""
        # entry.search() is now natively async I/O (an HTTP call to
        # Ollama), not a blocking local computation — so it no longer
        # needs the run_in_executor wrapper the sentence-transformers
        # version required to keep encode() off the event loop. The
        # TF-IDF fallback path inside search() is a single-string
        # transform and stays cheap enough to run inline; the expensive
        # full-corpus TF-IDF fit is offloaded inside _IndexEntry.build().
        return await entry.search(question, top_k, exclude_session_id=exclude_id)

    async def _get_session_entry(self, session_id: str) -> Optional[_IndexEntry]:
        # session_text_corpus() now defaults to ["transcript", "session_summary"]
        # (see lkc_graph.py) — this is the fallback scope used when a
        # session's owner couldn't be resolved (see session_pipeline.py),
        # so it needs the same summary visibility as the user/global scopes.
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

    async def _get_user_entry(self, user_id: str, session_ids: list[str]) -> Optional[_IndexEntry]:
        """
        Same rebuild-on-count-change caching as _get_session_entry, keyed
        on user_id instead of session_id. `session_ids` is recomputed by
        the caller on every call (see session_pipeline.py) — that's a
        cheap indexed query, so it's fine to re-resolve "which sessions
        can this user see" every QA turn even though the embedding index
        itself is only rebuilt when the underlying record count changes.
        This also means a brand-new session this user just started gets
        picked up automatically, without any explicit cache invalidation.
        """
        if not session_ids:
            return None
        records = await lkc_graph.read_lkc(
            session_ids=session_ids, record_type=["transcript", "session_summary"]
        )
        cached = self._users.get(user_id)
        if cached is None or len(records) != cached.record_count:
            # Same cost caveat as _get_session_entry: this re-embeds the
            # user's ENTIRE cross-session corpus from scratch whenever the
            # total record count across their sessions changes, not just
            # the new records. Fine for a single active session's worth of
            # transcript growth; would need to become incremental if a
            # user's total history across many sessions gets large enough
            # that re-embedding on every new segment becomes a real cost.
            entry = _IndexEntry(records)
            await entry.build()
            self._users[user_id] = entry
        return self._users.get(user_id)

    async def _get_global_entry(self) -> Optional[_IndexEntry]:
        all_records = await lkc_graph.read_lkc(record_type=["transcript", "session_summary"])
        if len(all_records) == self._global_record_count and self._global is not None:
            return self._global
        lightweight = [
            {"timestamp_iso": r.get("timestamp_iso", ""),
             "speaker": r.get("speaker", ""),
             "text": r.get("text") or r.get("summary") or "",
             "type": r.get("type")}
            for r in all_records
            if r.get("text") or r.get("summary")
        ]
        self._global = _IndexEntry(lightweight)
        await self._global.build()
        self._global_record_count = len(all_records)
        return self._global

    def invalidate(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._global = None
        # A cheap blanket clear rather than figuring out which cached users'
        # session_ids sets included this session_id — user-scoped indexes
        # rebuild lazily and cheaply from the DB on next query anyway (see
        # _get_user_entry), so there's no real cost to just dropping them
        # all here.
        self._users.clear()

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
            "cached_users":      len(self._users),
            "global_index_size": self._global_record_count,
        }


_retriever: Optional[LKCRetriever] = None


def get_retriever() -> LKCRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LKCRetriever()
    return _retriever