"""
lkc_retrieval.py — In-Session LKC Retrieval (Module 5, Month 4)

Provides a dense-embedding retrieval layer over lkc_stream.jsonl so the
agent can answer questions grounded in what was already said or written in
the LKC during this session.

Month 4 changes
---------------
* Primary retriever: sentence-transformers (all-MiniLM-L6-v2 by default).
  Embeddings are computed once per record and cached in a numpy matrix.
  Cosine similarity is computed via a dot product on L2-normalised vectors,
  giving the same result as sklearn's cosine_similarity but without the
  sklearn dependency at query time.
* Fallback retriever: TF-IDF (sklearn) — activated automatically when
  sentence-transformers is not installed.  This preserves Month 2/3 behaviour
  in environments without a GPU or where the model download is not feasible.
* The index is rebuilt lazily whenever lkc_stream.jsonl is newer than the
  last build timestamp.  Dense encoding costs ~5ms per record on CPU for
  MiniLM; the full rebuild for a 300-segment session takes < 2 s on CPU.
* Returns the top-k most relevant segments as a formatted context string
  ready to inject into the local LLM prompt.

Design notes
------------
* No vector database dependency — in-memory numpy cosine search is sufficient
  for a PoC with O(hundreds) of segments.
* The embedding model is loaded once at module level via `_get_embed_model()`.
* Month 5 will add persistent storage so the LKC graph survives restarts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Month 4: sentence-transformers (primary) ──────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# ── TF-IDF fallback (Month 2/3) ───────────────────────────────────────────────
SKLEARN_AVAILABLE = False
if not SBERT_AVAILABLE:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
        import numpy as np
        SKLEARN_AVAILABLE = True
    except ImportError:
        log.warning("Neither sentence-transformers nor scikit-learn installed — LKC retrieval disabled.")

# ── Embedding model singleton ─────────────────────────────────────────────────
_embed_model: Optional["SentenceTransformer"] = None
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22 MB, fast on CPU

def _get_embed_model() -> Optional["SentenceTransformer"]:
    """Load the sentence-transformer model once per process."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    if not SBERT_AVAILABLE:
        return None
    try:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        log.info(f"[lkc_retrieval] sentence-transformers model loaded: {_EMBED_MODEL_NAME}")
        return _embed_model
    except Exception as exc:
        log.warning(f"[lkc_retrieval] model load failed ({exc}) — falling back to TF-IDF")
        return None


class LKCRetriever:
    """
    Wraps lkc_stream.jsonl with a dense-embedding index for semantic search.

    Month 4: primary path uses sentence-transformers embeddings stored as a
    normalised numpy matrix; TF-IDF is the automatic fallback.

    Usage:
        retriever = LKCRetriever(Path("lkc_stream.jsonl"))
        context   = retriever.query("what did we decide about embeddings?", top_k=4)
    """

    def __init__(self, lkc_path: Path):
        self.lkc_path = lkc_path
        self._records:  list[dict] = []
        self._corpus:   list[str]  = []
        self._last_built: float = 0.0

        # Dense index (sentence-transformers)
        self._embeddings: Optional["np.ndarray"] = None  # shape: (N, D), L2-normalised

        # TF-IDF fallback index
        self._vectorizer: Optional[object] = None
        self._tfidf_matrix = None

    # ── Index management ──────────────────────────────────────────────────────
    def _needs_rebuild(self) -> bool:
        if not self.lkc_path.exists():
            return False
        mtime = self.lkc_path.stat().st_mtime
        return mtime > self._last_built or len(self._records) == 0

    def _build_index(self) -> None:
        if not self.lkc_path.exists():
            return

        lines = self.lkc_path.read_text(encoding="utf-8").strip().splitlines()
        records: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not records:
            return

        self._records = records
        # Corpus: combine speaker + text for richer term matching
        self._corpus = [
            f"{r.get('speaker', '')} {r.get('text', '')}".strip()
            for r in records
        ]

        model = _get_embed_model()
        if model is not None:
            # Dense path: encode all documents, L2-normalise for dot-product cosine
            raw = model.encode(self._corpus, batch_size=64, show_progress_bar=False)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
            self._embeddings   = (raw / norms).astype(np.float32)
            self._vectorizer   = None
            self._tfidf_matrix = None
            log.debug(f"[lkc_retrieval] dense index built: {len(records)} records "
                      f"(dim={self._embeddings.shape[1]})")
        elif SKLEARN_AVAILABLE:
            # TF-IDF fallback
            self._embeddings = None
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
            log.debug(f"[lkc_retrieval] TF-IDF index built: {len(records)} records (fallback)")
        else:
            log.warning("[lkc_retrieval] no retrieval backend available — index not built")
            return

        self._last_built = time.time()

    # ── Public query interface ────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 4) -> str:
        """
        Return a formatted context string with the top_k most relevant LKC
        segments for the given question.

        Uses dense cosine similarity when sentence-transformers is available,
        otherwise falls back to TF-IDF cosine similarity.
        Returns empty string if no index could be built.
        """
        if self._needs_rebuild():
            self._build_index()

        if not self._records:
            return ""

        model = _get_embed_model()

        if model is not None and self._embeddings is not None:
            # ── Dense retrieval ──────────────────────────────────────────────
            q_raw  = model.encode([question], show_progress_bar=False)
            q_norm = q_raw / max(np.linalg.norm(q_raw), 1e-9)
            scores = (self._embeddings @ q_norm.T).flatten()   # cosine similarity
        elif SKLEARN_AVAILABLE and self._tfidf_matrix is not None:
            # ── TF-IDF fallback ──────────────────────────────────────────────
            q_vec  = self._vectorizer.transform([question])
            scores = _sk_cosine(q_vec, self._tfidf_matrix).flatten()
        else:
            return ""

        top_idx = scores.argsort()[::-1][:top_k]

        hits: list[str] = []
        relevance_floor = 0.15 if (model is not None) else 0.05
        for idx in top_idx:
            if float(scores[idx]) < relevance_floor:
                continue
            r  = self._records[idx]
            ts = r.get("timestamp_iso", "")[:19].replace("T", " ")
            hits.append(f"[{ts}] {r.get('speaker','?')}: {r.get('text','')}")

        return "\n".join(hits)

    def stats(self) -> dict:
        backend = (
            "sentence-transformers"
            if (SBERT_AVAILABLE and self._embeddings is not None)
            else ("tfidf" if SKLEARN_AVAILABLE else "none")
        )
        return {
            "records":           len(self._records),
            "index_built":       self._last_built > 0,
            "backend":           backend,
            "sbert_available":   SBERT_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "embed_dim":         (
                int(self._embeddings.shape[1])
                if self._embeddings is not None else None
            ),
        }


# Module-level singleton — shared by all sessions
_retriever: Optional[LKCRetriever] = None

def get_retriever(lkc_path: Path) -> LKCRetriever:
    global _retriever
    if _retriever is None or _retriever.lkc_path != lkc_path:
        _retriever = LKCRetriever(lkc_path)
    return _retriever