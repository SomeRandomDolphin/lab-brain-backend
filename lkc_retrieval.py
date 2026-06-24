"""
lkc_retrieval.py — Persistent Dense LKC Retrieval (Module 5, Month 5)

Month 5 changes
---------------
* Upgraded embedding model: all-mpnet-base-v2 (~420 MB) replaces
  all-MiniLM-L6-v2 (~22 MB) for higher retrieval accuracy on longer sessions.
  Falls back to MiniLM automatically when mpnet fails to load (resource-
  constrained laptops, Docker containers without enough disk space, etc.).

* Persistent index: the retriever now sources documents from the SQLite
  LKC graph (lkc_graph.py) instead of reading lkc_stream.jsonl each query.
  This means:
    - Retrieval works across all sessions, not just the current one.
    - The index survives server restarts: records already in the DB are
      indexed without re-reading the JSONL file.
    - Cross-session Q&A: "what did we decide in the sprint planning meeting
      last week?" now works because those records are in the graph.

* Lazy per-session sub-index: by default `query()` restricts to the current
  session_id for performance.  Pass session_id=None to search all sessions.

* The module-level TF-IDF fallback is kept unchanged for environments where
  neither torch nor sentence-transformers is available.

API (unchanged from Month 4)
----------------------------
    retriever = get_retriever()                # module-level singleton
    context   = retriever.query("what did we decide?", top_k=4)
    stats     = retriever.stats()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Month 5: Upgraded primary embedding model ─────────────────────────────────
_EMBED_MODEL_PRIMARY  = "sentence-transformers/all-mpnet-base-v2"   # ~420 MB
_EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"    # ~22 MB

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    log.warning("[lkc_retrieval] sentence-transformers not installed — trying TF-IDF fallback.")

SKLEARN_AVAILABLE = False
if not SBERT_AVAILABLE:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
        import numpy as np
        SKLEARN_AVAILABLE = True
    except ImportError:
        log.warning("[lkc_retrieval] scikit-learn also not installed — retrieval disabled.")

# ── Embedding model singleton ─────────────────────────────────────────────────
_embed_model: Optional["SentenceTransformer"] = None
_embed_model_name: str = ""


def _get_embed_model() -> Optional["SentenceTransformer"]:
    """
    Load the best available sentence-transformers model, once per process.
    Tries all-mpnet-base-v2 first; falls back to all-MiniLM-L6-v2 on failure.
    """
    global _embed_model, _embed_model_name
    if _embed_model is not None:
        return _embed_model
    if not SBERT_AVAILABLE:
        return None

    for model_name in (_EMBED_MODEL_PRIMARY, _EMBED_MODEL_FALLBACK):
        try:
            _embed_model = SentenceTransformer(model_name)
            _embed_model_name = model_name
            log.info(f"[lkc_retrieval] embedding model loaded: {model_name}")
            return _embed_model
        except Exception as exc:
            log.warning(f"[lkc_retrieval] could not load {model_name}: {exc}")

    log.warning("[lkc_retrieval] all embedding models failed — falling back to TF-IDF")
    return None


# ── In-memory dense index ─────────────────────────────────────────────────────

class LKCRetriever:
    """
    Dense-embedding retriever over the persistent SQLite LKC graph.

    Month 5 design
    --------------
    * Sources records from lkc_graph.read_lkc() instead of reading JSONL.
    * Index is built once per session_id and rebuilt when new records arrive
      (checked via record count comparison, not mtime).
    * Supports cross-session queries (pass session_id=None to query all).
    * Gracefully degrades to TF-IDF when sentence-transformers is absent.
    """

    def __init__(self) -> None:
        # Per-session index cache: session_id → IndexEntry
        self._sessions: dict[str, "_IndexEntry"] = {}
        # "all sessions" index — rebuilt when total record count changes
        self._global: Optional["_IndexEntry"] = None
        self._global_record_count: int = 0

    def query(
        self,
        question: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Return a formatted context string with the top_k most relevant
        LKC segments for the given question.

        Pass session_id to restrict search to one session (faster, default
        behaviour when called from server.py).
        Pass session_id=None to search across all sessions.
        """
        import lkc_graph  # late import avoids circular deps at module load

        if session_id is not None:
            entry = self._get_session_entry(session_id, lkc_graph)
        else:
            entry = self._get_global_entry(lkc_graph)

        if entry is None or not entry.records:
            return ""

        return entry.search(question, top_k)

    def _get_session_entry(self, session_id: str, lkc_graph) -> Optional["_IndexEntry"]:
        records = lkc_graph.session_text_corpus(session_id)
        cached  = self._sessions.get(session_id)

        if cached is None or len(records) != cached.record_count:
            entry = _IndexEntry(records)
            entry.build()
            self._sessions[session_id] = entry

        return self._sessions.get(session_id)

    def _get_global_entry(self, lkc_graph) -> Optional["_IndexEntry"]:
        # Pull all transcript records across all sessions
        all_records = lkc_graph.read_lkc(record_type="transcript")
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
        self._global.build()
        self._global_record_count = len(all_records)
        return self._global

    def invalidate(self, session_id: str) -> None:
        """Force index rebuild on next query for a session (call after bulk ingest)."""
        self._sessions.pop(session_id, None)
        self._global = None

    def stats(self) -> dict:
        model = _get_embed_model()
        backend = (
            _embed_model_name
            if (SBERT_AVAILABLE and model is not None)
            else ("tfidf" if SKLEARN_AVAILABLE else "none")
        )
        return {
            "backend":             backend,
            "sbert_available":     SBERT_AVAILABLE,
            "sklearn_available":   SKLEARN_AVAILABLE,
            "cached_sessions":     len(self._sessions),
            "global_index_size":   self._global_record_count,
        }


class _IndexEntry:
    """Holds one dense (or TF-IDF) index for a collection of LKC records."""

    def __init__(self, records: list[dict]) -> None:
        self.records      = records
        self.record_count = len(records)
        self.corpus: list[str] = []
        # Dense
        self._embeddings: Optional["np.ndarray"] = None
        # TF-IDF fallback
        self._vectorizer = None
        self._tfidf_mat  = None

    def build(self) -> None:
        if not self.records:
            return

        self.corpus = [
            f"{r.get('speaker', '')} {r.get('text', '')}".strip()
            for r in self.records
        ]

        model = _get_embed_model()
        if model is not None:
            raw   = model.encode(self.corpus, batch_size=64, show_progress_bar=False)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._embeddings = (raw / norms).astype(np.float32)
            log.debug(f"[lkc_retrieval] dense index: {len(self.records)} recs "
                      f"dim={self._embeddings.shape[1]} model={_embed_model_name}")
        elif SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=8000, sublinear_tf=True
            )
            self._tfidf_mat = self._vectorizer.fit_transform(self.corpus)
            log.debug(f"[lkc_retrieval] TF-IDF index: {len(self.records)} recs")

    def search(self, question: str, top_k: int) -> str:
        model = _get_embed_model()

        if model is not None and self._embeddings is not None:
            q_raw   = model.encode([question], show_progress_bar=False)
            q_norm  = q_raw / max(float(np.linalg.norm(q_raw)), 1e-9)
            scores  = (self._embeddings @ q_norm.T).flatten()
            floor   = 0.20   # mpnet scores are generally higher than MiniLM
        elif SKLEARN_AVAILABLE and self._tfidf_mat is not None:
            q_vec   = self._vectorizer.transform([question])
            scores  = _sk_cosine(q_vec, self._tfidf_mat).flatten()
            floor   = 0.05
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


# ── Module-level singleton ─────────────────────────────────────────────────────
_retriever: Optional[LKCRetriever] = None


def get_retriever() -> LKCRetriever:
    """Return (or create) the module-level retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = LKCRetriever()
    return _retriever
