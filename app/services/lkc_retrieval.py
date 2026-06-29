"""
app/services/lkc_retrieval.py — Dense LKC Retrieval Service.

Sources documents from the SQLite LKC graph (supports cross-session Q&A).
Primary model: all-mpnet-base-v2 (~420 MB).
Fallback: all-MiniLM-L6-v2 (~22 MB) → TF-IDF.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.db import lkc_graph

log = logging.getLogger(__name__)

_EMBED_MODEL_PRIMARY  = "sentence-transformers/all-mpnet-base-v2"
_EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    log.warning("[retrieval] sentence-transformers not installed — trying TF-IDF.")

SKLEARN_AVAILABLE = False
if not SBERT_AVAILABLE:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
        import numpy as np
        SKLEARN_AVAILABLE = True
    except ImportError:
        log.warning("[retrieval] scikit-learn also missing — retrieval disabled.")

_embed_model = None
_embed_model_name: str = ""


def _get_embed_model():
    global _embed_model, _embed_model_name
    if _embed_model is not None:
        return _embed_model
    if not SBERT_AVAILABLE:
        return None
    for model_name in (_EMBED_MODEL_PRIMARY, _EMBED_MODEL_FALLBACK):
        try:
            _embed_model = SentenceTransformer(model_name)
            _embed_model_name = model_name
            log.info(f"[retrieval] embedding model loaded: {model_name}")
            return _embed_model
        except Exception as exc:
            log.warning(f"[retrieval] could not load {model_name}: {exc}")
    return None


class _IndexEntry:
    def __init__(self, records: list[dict]) -> None:
        self.records      = records
        self.record_count = len(records)
        self.corpus: list[str] = []
        self._embeddings = None
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
        elif SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=8000, sublinear_tf=True
            )
            self._tfidf_mat = self._vectorizer.fit_transform(self.corpus)

    def search(self, question: str, top_k: int) -> str:
        model = _get_embed_model()
        if model is not None and self._embeddings is not None:
            q_raw  = model.encode([question], show_progress_bar=False)
            q_norm = q_raw / max(float(np.linalg.norm(q_raw)), 1e-9)
            scores = (self._embeddings @ q_norm.T).flatten()
            floor  = 0.20
        elif SKLEARN_AVAILABLE and self._tfidf_mat is not None:
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

    def query(
        self,
        question: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
    ) -> str:
        if session_id is not None:
            entry = self._get_session_entry(session_id)
        else:
            entry = self._get_global_entry()
        if entry is None or not entry.records:
            return ""
        return entry.search(question, top_k)

    def _get_session_entry(self, session_id: str) -> Optional[_IndexEntry]:
        records = lkc_graph.session_text_corpus(session_id)
        cached  = self._sessions.get(session_id)
        if cached is None or len(records) != cached.record_count:
            entry = _IndexEntry(records)
            entry.build()
            self._sessions[session_id] = entry
        return self._sessions.get(session_id)

    def _get_global_entry(self) -> Optional[_IndexEntry]:
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
            "backend":           backend,
            "sbert_available":   SBERT_AVAILABLE,
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
