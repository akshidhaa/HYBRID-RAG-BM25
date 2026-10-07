"""
retrieval.py
------------
Hybrid retriever combining:
  1. BM25 (keyword/lexical search) via rank_bm25
  2. "Semantic" search via TF-IDF + Truncated SVD (LSA), cosine similarity

Why LSA instead of a downloaded neural embedding model:
  This environment has no access to model hubs (huggingface.co etc.) at
  runtime, so we approximate semantic similarity with a classic, fully
  offline technique (Latent Semantic Analysis). It captures co-occurrence
  structure beyond exact keyword overlap, which is enough to demonstrate
  genuine *disagreement* between lexical and semantic signals -- the thing
  signal #2 in the abstention layer actually needs. In production you'd
  swap SemanticIndex's internals for a real embedding model without
  touching any other module (that's the point of the interface below).

Both indices expose `.search(query, top_k) -> List[(chunk_id, score)]`
with scores normalized to [0, 1] so they can be fused fairly.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from chunking import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _saturating_normalize(scores: np.ndarray, scale: float) -> np.ndarray:
    """
    Maps raw, unbounded scores to [0, 1] using an ABSOLUTE saturating curve
    (score / (score + scale)) rather than min-max across this query's own
    candidates.

    This distinction matters a lot for the abstention layer: min-max
    normalization would make the single best-matching chunk score 1.0 by
    definition, EVEN IF every candidate is a terrible match (e.g. nothing in
    the document is relevant). That would make "retrieval strength" useless
    as a confidence signal -- it would always report high confidence for
    the top-1 result. Saturating normalization instead asks "how good is
    this match in absolute terms," which is what the abstention decision
    actually needs.

    `scale` is the score at which normalized confidence crosses 0.5; tune
    per corpus/score-scale if needed.
    """
    scores = np.maximum(scores, 0)
    return scores / (scores + scale)


class BM25Index:
    # BM25 raw scores are typically single digits for a good keyword match
    # on short-to-medium chunks; 5.0 is a reasonable "half confidence" point.
    # Tune this if your corpus has much longer/shorter chunks.
    SATURATION_SCALE = 5.0

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        tokenized_query = _tokenize(query)
        raw_scores = np.array(self.bm25.get_scores(tokenized_query))
        norm_scores = _saturating_normalize(raw_scores, scale=self.SATURATION_SCALE)
        order = np.argsort(-norm_scores)[:top_k]
        return [(int(self.chunks[i].id), float(norm_scores[i])) for i in order]


class SemanticIndex:
    # Below this many chunks, TruncatedSVD has too few documents to estimate
    # stable latent components -- with e.g. 5 chunks it collapses to ~4
    # dimensions and produces spuriously high similarities for *unrelated*
    # queries (verified empirically: an unrelated query scored 0.92 against
    # an off-topic chunk in a 5-chunk corpus). Below the threshold we fall
    # back to raw TF-IDF cosine similarity, which is less "semantic" but far
    # more numerically stable and still meaningfully different from BM25
    # (different term weighting/normalization), which is what the rank-
    # agreement signal actually needs.
    MIN_CHUNKS_FOR_SVD = 30

    def __init__(self, chunks: List[Chunk], n_components: int = 50):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.tfidf_matrix = tfidf_matrix  # kept for top-term extraction (see top_terms())

        use_svd = len(texts) >= self.MIN_CHUNKS_FOR_SVD
        if use_svd:
            n_components = max(1, min(n_components, tfidf_matrix.shape[1] - 1, len(texts) - 1))
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        else:
            self.svd = None
            self.doc_vectors = tfidf_matrix.toarray()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        if self.svd is not None:
            query_vec = self.svd.transform(query_vec)
        else:
            query_vec = query_vec.toarray()

        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        # Cosine similarity is already bounded; just clip negatives to 0.
        # (No min-max normalization here either -- same reasoning as BM25:
        # we need an absolute sense of "how semantically close is this,"
        # not a relative ranking among this query's own candidates.)
        norm_scores = np.clip(sims, 0.0, 1.0)
        order = np.argsort(-norm_scores)[:top_k]
        return [(int(self.chunks[i].id), float(norm_scores[i])) for i in order]

    def top_terms(self, n: int = 8) -> List[str]:
        """Returns the n highest-weighted vocabulary terms across the whole
        corpus (summed TF-IDF), longest-first among ties. Used to generate
        document-aware example questions instead of hardcoded ones -- a
        question built from the document's own vocabulary is far more
        likely to actually retrieve well (see BM25/TF-IDF matching on
        shared tokens)."""
        scores = np.asarray(self.tfidf_matrix.sum(axis=0)).ravel()
        vocab = self.vectorizer.get_feature_names_out()
        order = np.argsort(-scores)
        terms = []
        for i in order:
            term = vocab[i]
            if len(term) < 4 or term.isdigit():
                continue
            terms.append(term)
            if len(terms) >= n:
                break
        return terms


@dataclass
class RetrievedChunk:
    chunk: Chunk
    bm25_score: float
    semantic_score: float
    fused_score: float


class HybridRetriever:
    """
    Fuses BM25 + semantic results. Also exposes the raw per-signal rankings
    so the abstention layer can compute rank agreement independently of
    the fused score.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        semantic_index_cls=SemanticIndex,
    ):
        """
        semantic_index_cls: any class implementing the SemanticIndex
        interface (.search(query, top_k) -> List[(chunk_id, score)],
        .top_terms(n) -> List[str]). Defaults to the built-in TF-IDF/LSA
        index. Pass embeddings_retrieval.EmbeddingIndex here to use real
        sentence-transformer embeddings instead — see embeddings_retrieval.py.
        """
        self.chunks = chunks
        self.chunk_by_id: Dict[int, Chunk] = {c.id: c for c in chunks}
        self.bm25_index = BM25Index(chunks)
        self.semantic_index = semantic_index_cls(chunks)
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 10):
        bm25_results = dict(self.bm25_index.search(query, top_k=candidate_pool))
        semantic_results = dict(self.semantic_index.search(query, top_k=candidate_pool))

        all_ids = set(bm25_results) | set(semantic_results)
        fused = []
        for cid in all_ids:
            b = bm25_results.get(cid, 0.0)
            s = semantic_results.get(cid, 0.0)
            fused_score = self.bm25_weight * b + self.semantic_weight * s
            fused.append(
                RetrievedChunk(
                    chunk=self.chunk_by_id[cid],
                    bm25_score=b,
                    semantic_score=s,
                    fused_score=fused_score,
                )
            )
        fused.sort(key=lambda r: -r.fused_score)
        top = fused[:top_k]

        # Rankings needed for rank-agreement signal
        bm25_ranking = [cid for cid, _ in sorted(bm25_results.items(), key=lambda kv: -kv[1])]
        semantic_ranking = [cid for cid, _ in sorted(semantic_results.items(), key=lambda kv: -kv[1])]

        return {
            "top_chunks": top,
            "bm25_ranking": bm25_ranking,
            "semantic_ranking": semantic_ranking,
        }

    def suggest_example_questions(self, n: int = 5) -> List[str]:
        """Generates document-aware example questions from the corpus's
        own top TF-IDF terms, e.g. 'What does the document say about
        <term>?'. Replaces hardcoded example questions that only made
        sense for one specific sample document."""
        terms = self.semantic_index.top_terms(n=n)
        return [f"What does the document say about {term}?" for term in terms]


if __name__ == "__main__":
    from chunking import chunk_document

    doc = (
        "The company's revenue in Q3 2025 was $42 million, a 12% increase year over year. "
        "Operating expenses rose to $30 million, driven mostly by engineering hires. "
        "Net income for the quarter was $8.4 million. "
        "The board approved a $50 million stock buyback program. "
        "The CEO stated the company remains focused on international expansion in 2026."
    )
    chunks = chunk_document(doc, target_chunk_chars=150)
    retriever = HybridRetriever(chunks)
    result = retriever.retrieve("What was the net income?")
    for rc in result["top_chunks"]:
        print(f"[fused={rc.fused_score:.2f} bm25={rc.bm25_score:.2f} sem={rc.semantic_score:.2f}] {rc.chunk.text}")
