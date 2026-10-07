"""
pipeline.py
-----------
End-to-end Confidence-Aware Hybrid RAG pipeline, matching the architecture
diagram in the proposal:

    User Query
        |
        v
    Hybrid Retrieval (BM25 + Semantic)
        |
        v
    Re-ranking (fused top-k)
        |
        v
    Generate candidate answer
        |
        v
    Confidence-Aware Abstention Layer
        |
        +-- score >= threshold --> return answer + citations
        |
        +-- score <  threshold --> abstain ("insufficient information")
"""

from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()  # picks up ANTHROPIC_API_KEY / RAG_USE_EMBEDDINGS from .env if present

from chunking import chunk_document, Chunk
from retrieval import HybridRetriever, SemanticIndex
from generation import get_default_generator, Answer
from abstention import ConfidenceAwareAbstention, ConfidenceReport

ABSTAIN_MESSAGE = "Insufficient information found in the document to answer this confidently."


@dataclass
class PipelineResult:
    query: str
    decision: str  # "answer" | "abstain"
    answer_text: str
    cited_chunk_ids: List[int]
    confidence: ConfidenceReport


class ConfidenceAwareRAGPipeline:
    def __init__(
        self,
        document_text: str,
        threshold: float = 0.55,
        top_k: int = 3,
        generator=None,
        chunk_chars: int = 300,
        overlap_sentences: int = 0,
        semantic_index_cls=SemanticIndex,
    ):
        self.chunks: List[Chunk] = chunk_document(
            document_text, target_chunk_chars=chunk_chars, overlap_sentences=overlap_sentences
        )
        self.retriever = HybridRetriever(self.chunks, semantic_index_cls=semantic_index_cls)
        self.generator = generator or get_default_generator()
        self.abstention_layer = ConfidenceAwareAbstention(threshold=threshold)
        self.top_k = top_k

    def answer(self, query: str) -> PipelineResult:
        retrieval = self.retriever.retrieve(query, top_k=self.top_k)
        top_chunks = [rc.chunk for rc in retrieval["top_chunks"]]
        top_fused_score = retrieval["top_chunks"][0].fused_score if retrieval["top_chunks"] else 0.0

        candidate_answer: Optional[Answer] = None
        if top_chunks:
            candidate_answer = self.generator.generate(query, top_chunks)

        confidence = self.abstention_layer.evaluate(
            top_fused_score=top_fused_score,
            bm25_ranking=retrieval["bm25_ranking"],
            semantic_ranking=retrieval["semantic_ranking"],
            answer=candidate_answer,
            source_chunks=top_chunks,
        )

        if confidence.decision == "answer" and candidate_answer:
            return PipelineResult(
                query=query,
                decision="answer",
                answer_text=candidate_answer.text,
                cited_chunk_ids=candidate_answer.source_chunk_ids,
                confidence=confidence,
            )
        else:
            return PipelineResult(
                query=query,
                decision="abstain",
                answer_text=ABSTAIN_MESSAGE,
                cited_chunk_ids=[],
                confidence=confidence,
            )


if __name__ == "__main__":
    # To use real embeddings instead of TF-IDF/LSA (Phase 2 upgrade):
    #   pip install sentence-transformers
    #   from embeddings_retrieval import EmbeddingIndex
    #   pipeline = ConfidenceAwareRAGPipeline(doc, semantic_index_cls=EmbeddingIndex)
    doc = (
        "The company's revenue in Q3 2025 was $42 million, a 12% increase year over year. "
        "Operating expenses rose to $30 million, driven mostly by engineering hires. "
        "Net income for the quarter was $8.4 million. "
        "The board approved a $50 million stock buyback program. "
        "The CEO stated the company remains focused on international expansion in 2026."
    )
    pipeline = ConfidenceAwareRAGPipeline(doc)

    for q in [
        "What was the net income for the quarter?",
        "What is the CEO's stance on remote work policy?",  # unanswerable
    ]:
        result = pipeline.answer(q)
        print(f"\nQ: {q}")
        print(f"Decision: {result.decision}  (composite={result.confidence.composite_score:.2f})")
        print(f"Answer: {result.answer_text}")
        print(f"Signals: retrieval={result.confidence.retrieval_strength:.2f} "
              f"rank_agreement={result.confidence.rank_agreement:.2f} "
              f"groundedness={result.confidence.citation_groundedness:.2f}")
