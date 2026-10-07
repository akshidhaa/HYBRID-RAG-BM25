"""
abstention.py
-------------
The Confidence-Aware Abstention Layer described in the proposal.

Three independent, lightweight signals are combined into a composite score:

  1. Retrieval strength   - how strong is the best fused match? A low top
                             score means nothing in the document closely
                             matches the query.
  2. Rank agreement       - do BM25 and semantic search independently agree
                             on the top-ranked chunk (or overlap heavily in
                             their top-k)? Agreement = corroborating evidence.
                             Disagreement = ambiguous / poorly covered query.
  3. Citation groundedness - after generation, are the factual claims in the
                             answer (numbers, dates, proper nouns/entities)
                             actually traceable to the retrieved chunks? This
                             is the direct hallucination check.

The three signals are combined with configurable weights into a single
composite confidence score in [0, 1]. If it falls below `threshold`, the
system abstains.

Each signal is also returned individually so the evaluation harness and any
UI can show *why* a decision was made, not just the final verdict.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from chunking import Chunk
from generation import Answer


@dataclass
class ConfidenceReport:
    retrieval_strength: float
    rank_agreement: float
    citation_groundedness: float
    composite_score: float
    decision: str  # "answer" or "abstain"
    threshold: float
    details: dict = field(default_factory=dict)


# ---------- Signal 1: Retrieval strength ----------

def retrieval_strength_signal(top_fused_score: float) -> float:
    """The fused top-1 score is already normalized to [0, 1] by the retriever,
    so this signal is a direct pass-through. Kept as its own function so the
    scoring logic (e.g. non-linear penalties for very low scores) can evolve
    independently of the retriever."""
    return max(0.0, min(1.0, top_fused_score))


# ---------- Signal 2: Rank agreement ----------

def rank_agreement_signal(bm25_ranking: List[int], semantic_ranking: List[int], top_k: int = 3) -> float:
    """
    Combines two checks:
      - Exact top-1 match (strongest signal, weighted higher)
      - Jaccard overlap of the top-k sets (softer corroboration signal)
    Returns a score in [0, 1].
    """
    if not bm25_ranking or not semantic_ranking:
        return 0.0

    top1_match = 1.0 if bm25_ranking[0] == semantic_ranking[0] else 0.0

    set_a = set(bm25_ranking[:top_k])
    set_b = set(semantic_ranking[:top_k])
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 0.0

    return 0.6 * top1_match + 0.4 * jaccard


# ---------- Signal 3: Citation groundedness ----------

_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def _extract_claims(text: str) -> List[str]:
    """Extracts checkable atomic claims: numbers/amounts and proper nouns.
    This mirrors the proposal's definition of groundedness ('numbers, names,
    dates'). Deliberately simple/lexical rather than a full NLI model, in
    keeping with the 'lightweight signals' design goal."""
    numbers = _NUMBER_RE.findall(text)
    proper_nouns = _PROPER_NOUN_RE.findall(text)
    # de-duplicate while preserving order
    seen = set()
    claims = []
    for c in numbers + proper_nouns:
        if c not in seen:
            seen.add(c)
            claims.append(c)
    return claims


def citation_groundedness_signal(answer_text: str, source_chunks: List[Chunk]) -> float:
    """
    Fraction of extracted claims in the answer that also appear verbatim in
    at least one source chunk. An answer with no checkable claims (e.g. a
    short qualitative statement) is treated as fully grounded (score 1.0)
    since there's nothing to contradict -- but note this is a known
    limitation (see README) and a real deployment would add an NLI-based
    entailment check for non-numeric claims too.
    """
    claims = _extract_claims(answer_text)
    if not claims:
        return 1.0

    combined_source_text = " ".join(c.text for c in source_chunks)
    grounded = sum(1 for claim in claims if claim in combined_source_text)
    return grounded / len(claims)


# ---------- Composite ----------

class ConfidenceAwareAbstention:
    def __init__(
        self,
        threshold: float = 0.55,
        w_retrieval: float = 0.35,
        w_rank_agreement: float = 0.25,
        w_groundedness: float = 0.40,
    ):
        self.threshold = threshold
        self.w_retrieval = w_retrieval
        self.w_rank_agreement = w_rank_agreement
        self.w_groundedness = w_groundedness

    def evaluate(
        self,
        top_fused_score: float,
        bm25_ranking: List[int],
        semantic_ranking: List[int],
        answer: Optional[Answer],
        source_chunks: List[Chunk],
    ) -> ConfidenceReport:
        s1 = retrieval_strength_signal(top_fused_score)
        s2 = rank_agreement_signal(bm25_ranking, semantic_ranking)
        s3 = citation_groundedness_signal(answer.text, source_chunks) if answer else 0.0

        composite = self.w_retrieval * s1 + self.w_rank_agreement * s2 + self.w_groundedness * s3
        decision = "answer" if composite >= self.threshold else "abstain"

        return ConfidenceReport(
            retrieval_strength=s1,
            rank_agreement=s2,
            citation_groundedness=s3,
            composite_score=composite,
            decision=decision,
            threshold=self.threshold,
            details={
                "weights": {
                    "retrieval": self.w_retrieval,
                    "rank_agreement": self.w_rank_agreement,
                    "groundedness": self.w_groundedness,
                }
            },
        )


if __name__ == "__main__":
    # Quick sanity check: strong agreement + grounded answer -> should answer
    chunks = [Chunk(id=0, text="Net income for the quarter was $8.4 million.", start_char=0, end_char=10)]
    ans = Answer(text="Net income was $8.4 million.", source_chunk_ids=[0])
    layer = ConfidenceAwareAbstention()
    report = layer.evaluate(
        top_fused_score=0.95,
        bm25_ranking=[0, 1],
        semantic_ranking=[0, 2],
        answer=ans,
        source_chunks=chunks,
    )
    print(report)

    # Weak retrieval + no grounding -> should abstain
    ans2 = Answer(text="The company was founded in 1998 and made $999 million.", source_chunk_ids=[0])
    report2 = layer.evaluate(
        top_fused_score=0.1,
        bm25_ranking=[0, 1],
        semantic_ranking=[3, 4],
        answer=ans2,
        source_chunks=chunks,
    )
    print(report2)
