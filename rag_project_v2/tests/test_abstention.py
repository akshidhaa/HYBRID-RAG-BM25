"""
Basic regression tests. Run with:
    pytest
or in VS Code: open the Testing sidebar (flask icon) and click Run All.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chunking import Chunk
from generation import Answer
from abstention import ConfidenceAwareAbstention
from pipeline import ConfidenceAwareRAGPipeline


def test_strong_match_answers():
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
    assert report.decision == "answer"
    assert report.composite_score > layer.threshold


def test_weak_match_abstains():
    chunks = [Chunk(id=0, text="Net income for the quarter was $8.4 million.", start_char=0, end_char=10)]
    ans = Answer(text="The company was founded in 1998 and made $999 million.", source_chunk_ids=[0])
    layer = ConfidenceAwareAbstention()
    report = layer.evaluate(
        top_fused_score=0.1,
        bm25_ranking=[0, 1],
        semantic_ranking=[3, 4],
        answer=ans,
        source_chunks=chunks,
    )
    assert report.decision == "abstain"


def test_pipeline_end_to_end_on_answerable_question():
    doc = (
        "The company's revenue in Q3 2025 was $42 million, a 12% increase "
        "year over year. Net income for the quarter was $8.4 million."
    )
    pipeline = ConfidenceAwareRAGPipeline(doc, threshold=0.4)
    result = pipeline.answer("What was the net income for the quarter?")
    assert result.decision in ("answer", "abstain")  # smoke test: no crash
    assert "8.4" in result.answer_text or result.decision == "abstain"


def test_pipeline_abstains_on_unrelated_question():
    doc = (
        "The company's revenue in Q3 2025 was $42 million. "
        "Net income for the quarter was $8.4 million."
    )
    pipeline = ConfidenceAwareRAGPipeline(doc, threshold=0.74)
    result = pipeline.answer("What is the capital of France?")
    assert result.decision == "abstain"
