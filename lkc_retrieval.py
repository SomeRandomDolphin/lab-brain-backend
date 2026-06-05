"""
lkc_retrieval.py — In-Session LKC Retrieval (Module 5, Month 2)

Provides a lightweight TF-IDF retrieval layer over lkc_stream.jsonl so the
agent can answer questions grounded in what was already said or written in
the LKC during this session.

Design notes
------------
* No vector database dependency — TF-IDF via sklearn is sufficient for a PoC
  with O(hundreds) of segments.
* The index is rebuilt lazily whenever lkc_stream.jsonl is newer than the
  last build timestamp.  This costs ~1ms for typical session sizes.
* Returns the top-k most relevant segments as a formatted context string
  ready to inject into the Gemini prompt.
* Month 3 will replace this with a proper embedding-based retrieval over the
  full persistent LKC graph.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed — LKC retrieval disabled.")


class LKCRetriever:
    """
    Wraps lkc_stream.jsonl with a TF-IDF index for fast semantic search.

    Usage:
        retriever = LKCRetriever(Path("lkc_stream.jsonl"))
        context   = retriever.query("what did we decide about embeddings?", top_k=4)
    """

    def __init__(self, lkc_path: Path):
        self.lkc_path = lkc_path
        self._records: list[dict] = []
        self._corpus:  list[str] = []
        self._vectorizer: Optional[object] = None
        self._matrix = None
        self._last_built: float = 0.0

    # ── Index management ──────────────────────────────────────────────────────
    def _needs_rebuild(self) -> bool:
        if not self.lkc_path.exists():
            return False
        mtime = self.lkc_path.stat().st_mtime
        return mtime > self._last_built or len(self._records) == 0

    def _build_index(self) -> None:
        if not SKLEARN_AVAILABLE:
            return
        if not self.lkc_path.exists():
            return

        lines = self.lkc_path.read_text(encoding="utf-8").strip().splitlines()
        records = []
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

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._corpus)
        self._last_built = time.time()
        log.debug(f"[lkc_retrieval] index built: {len(records)} records.")

    # ── Public query interface ────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 4) -> str:
        """
        Return a formatted context string with the top_k most relevant LKC
        segments for the given question. Returns empty string if no index.
        """
        if not SKLEARN_AVAILABLE:
            return ""

        if self._needs_rebuild():
            self._build_index()

        if self._matrix is None or len(self._records) == 0:
            return ""

        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]

        hits = []
        for idx in top_idx:
            if scores[idx] < 0.05:
                continue  # below relevance floor
            r = self._records[idx]
            ts = r.get("timestamp_iso", "")[:19].replace("T", " ")
            hits.append(f"[{ts}] {r.get('speaker','?')}: {r.get('text','')}")

        return "\n".join(hits)

    def stats(self) -> dict:
        return {
            "records": len(self._records),
            "index_built": self._last_built > 0,
            "sklearn_available": SKLEARN_AVAILABLE,
        }


# Module-level singleton — shared by all sessions
_retriever: Optional[LKCRetriever] = None

def get_retriever(lkc_path: Path) -> LKCRetriever:
    global _retriever
    if _retriever is None or _retriever.lkc_path != lkc_path:
        _retriever = LKCRetriever(lkc_path)
    return _retriever