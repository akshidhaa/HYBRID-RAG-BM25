"""
embeddings_retrieval.py
------------------------
Phase-2 upgrade: a real neural-embedding semantic index, as a drop-in
replacement for retrieval.SemanticIndex (same interface, so nothing else
in the pipeline needs to change).

Why this matters: TF-IDF/LSA (the original SemanticIndex) only captures
co-occurrence statistics. It can't tell that "revenue" and "sales" are
related if they never appear near each other in this specific corpus.
A sentence-transformer model has that relationship baked in from
pretraining on billions of sentences, so semantic search actually finds
paraphrases and synonyms, not just shared vocabulary.

Setup:
    pip install sentence-transformers

Usage (see pipeline.py / app.py for the wired-up version):
    from embeddings_retrieval import EmbeddingIndex
    from retrieval import HybridRetriever
    retriever = HybridRetriever(chunks, semantic_index_cls=EmbeddingIndex)

Interface contract (must match retrieval.SemanticIndex exactly):
    .search(query, top_k) -> List[Tuple[chunk_id: int, score: float in [0,1]]]
    .top_terms(n) -> List[str]   # used to auto-generate example questions
"""

from typing import List, Tuple
import re
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from chunking import Chunk

_MODEL_CACHE = {}


def _get_model(name: str):
    """Sentence-transformer models are ~80MB and slow to load; cache across
    documents/requests within the same process so re-indexing a new upload
    doesn't reload the model from disk every time."""
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


class EmbeddingIndex:
    # all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, strong general-purpose
    # quality/speed tradeoff. Swap for a larger model (e.g. bge-base-en-v1.5)
    # if you need higher recall and don't mind the extra latency.
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, chunks: List[Chunk], n_components: int = None):
        # n_components kept in the signature only so this class is a drop-in
        # swap for SemanticIndex(chunks, n_components=...); it's unused here.
        self.chunks = chunks
        self.model = _get_model(self.MODEL_NAME)
        texts = [c.text for c in chunks]
        self.doc_vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        # Embeddings are normalized, so cosine similarity is already in
        # [-1, 1]; clip to [0, 1] to match the [0,1] contract every other
        # signal in the abstention layer expects.
        norm_scores = np.clip(sims, 0.0, 1.0)
        order = np.argsort(-norm_scores)[:top_k]
        return [(int(self.chunks[i].id), float(norm_scores[i])) for i in order]

    def top_terms(self, n: int = 8) -> List[str]:
        """Embeddings don't expose a vocabulary the way TF-IDF does, so this
        falls back to simple word-frequency extraction. Only used to
        generate example questions in the UI — not part of retrieval or
        scoring, so a lower-fidelity fallback here is fine."""
        stop = {
            "this", "that", "with", "from", "have", "which", "were",
            "their", "also", "been", "will", "would", "could", "about",
        }
        words = []
        for c in self.chunks:
            words += re.findall(r"[A-Za-z]{4,}", c.text.lower())
        counts = Counter(w for w in words if w not in stop)
        return [w for w, _ in counts.most_common(n)]


if __name__ == "__main__":
    from chunking import chunk_document

    doc = (
        "The company's revenue in Q3 2025 was $42 million, a 12% increase "
        "year over year. Operating expenses rose to $30 million, driven "
        "mostly by engineering hires. Net income for the quarter was $8.4 "
        "million. The board approved a $50 million stock buyback program."
    )
    chunks = chunk_document(doc, target_chunk_chars=150)
    index = EmbeddingIndex(chunks)
    # Note: "earnings" never appears verbatim in the text — TF-IDF would
    # score this poorly, embeddings should still find the revenue/income
    # chunks via semantic similarity.
    for cid, score in index.search("What were the company's earnings?"):
        chunk = next(c for c in chunks if c.id == cid)
        print(f"[{score:.2f}] {chunk.text}")
