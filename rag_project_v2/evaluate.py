"""
evaluate.py
-----------
Implements the evaluation strategy from the proposal (Section 5):

  - Runs the pipeline over a labeled question set containing both
    answerable and deliberately unanswerable questions.
  - Scores the system's abstain-vs-answer decisions against ground truth
    using precision, recall, and F1 -- the "abstention accuracy" metric.
  - Also reports a simple answer-correctness check (substring match against
    a gold snippet) for questions the system chose to answer, so we can see
    the correctness/abstention trade-off, not abstention in isolation.
  - Additionally simulates a "standard RAG" baseline (threshold = 0, i.e.
    never abstains) run over the same data, to make the comparison called
    for in the proposal ("Standard RAG vs. Confidence-aware RAG") concrete.
"""

import json
import argparse
from dataclasses import dataclass
from typing import List

from pipeline import ConfidenceAwareRAGPipeline


@dataclass
class EvalRow:
    question: str
    should_answer: bool
    decision: str
    composite_score: float
    answer_text: str
    correct: bool  # only meaningful when decision == "answer" and should_answer == True


def load_questions(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def run_eval(document_path: str, questions_path: str, threshold: float, verbose: bool = True):
    doc = open(document_path).read()
    pipeline = ConfidenceAwareRAGPipeline(doc, threshold=threshold)
    questions = load_questions(questions_path)

    rows: List[EvalRow] = []
    for q in questions:
        result = pipeline.answer(q["question"])
        correct = False
        if result.decision == "answer" and q["should_answer"] and q.get("gold_answer_contains"):
            correct = q["gold_answer_contains"].lower() in result.answer_text.lower()
        rows.append(EvalRow(
            question=q["question"],
            should_answer=q["should_answer"],
            decision=result.decision,
            composite_score=result.confidence.composite_score,
            answer_text=result.answer_text,
            correct=correct,
        ))

    # --- Abstention accuracy (precision/recall/F1) ---
    # Define the positive class as "system should answer" (should_answer=True).
    tp = sum(1 for r in rows if r.should_answer and r.decision == "answer")
    fp = sum(1 for r in rows if not r.should_answer and r.decision == "answer")
    fn = sum(1 for r in rows if r.should_answer and r.decision == "abstain")
    tn = sum(1 for r in rows if not r.should_answer and r.decision == "abstain")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # --- Answer correctness among questions that were genuinely answerable ---
    answerable_rows = [r for r in rows if r.should_answer]
    answered_and_correct = sum(1 for r in answerable_rows if r.decision == "answer" and r.correct)
    correctness_rate = answered_and_correct / len(answerable_rows) if answerable_rows else 0.0

    # --- Hallucination rate: unanswerable questions the system answered anyway ---
    unanswerable_rows = [r for r in rows if not r.should_answer]
    hallucination_rate = fp / len(unanswerable_rows) if unanswerable_rows else 0.0

    if verbose:
        print(f"\n{'='*90}\nThreshold = {threshold}\n{'='*90}")
        print(f"{'Question':<55} {'Gold':<6} {'Decision':<9} {'Score':<6} {'Correct'}")
        print("-" * 90)
        for r in rows:
            gold = "ANS" if r.should_answer else "ABST"
            correct_str = "-" if not r.should_answer else ("Y" if r.correct else "N")
            print(f"{r.question[:53]:<55} {gold:<6} {r.decision:<9} {r.composite_score:<6.2f} {correct_str}")

        print("-" * 90)
        print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"Abstention precision: {precision:.2f}  recall: {recall:.2f}  F1: {f1:.2f}")
        print(f"Answer correctness (of genuinely answerable Qs): {correctness_rate:.2f}")
        print(f"Hallucination rate (answered when it shouldn't have): {hallucination_rate:.2f}")

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correctness_rate": correctness_rate,
        "hallucination_rate": hallucination_rate,
        "rows": rows,
    }


def compare_standard_vs_confidence_aware(document_path: str, questions_path: str, confidence_threshold: float):
    print("\n" + "#" * 90)
    print("# BASELINE: Standard RAG (threshold=0.0 -> effectively never abstains)")
    print("#" * 90)
    baseline = run_eval(document_path, questions_path, threshold=0.0)

    print("\n" + "#" * 90)
    print(f"# CONFIDENCE-AWARE RAG (threshold={confidence_threshold})")
    print("#" * 90)
    confidence_aware = run_eval(document_path, questions_path, threshold=confidence_threshold)

    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<35}{'Standard RAG':<20}{'Confidence-Aware RAG'}")
    print(f"{'Hallucination rate':<35}{baseline['hallucination_rate']:<20.2f}{confidence_aware['hallucination_rate']:.2f}")
    print(f"{'Abstention F1':<35}{baseline['f1']:<20.2f}{confidence_aware['f1']:.2f}")
    print(f"{'Answer correctness rate':<35}{baseline['correctness_rate']:<20.2f}{confidence_aware['correctness_rate']:.2f}")


def threshold_sweep(document_path: str, questions_path: str):
    print("\n" + "#" * 90)
    print("# THRESHOLD SWEEP (for calibrating the abstention threshold)")
    print("#" * 90)
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Correctness':<14}{'Halluc. rate'}")
    for t in [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8]:
        res = run_eval(document_path, questions_path, threshold=t, verbose=False)
        print(f"{t:<12}{res['precision']:<12.2f}{res['recall']:<12.2f}{res['f1']:<12.2f}"
              f"{res['correctness_rate']:<14.2f}{res['hallucination_rate']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="sample_data/document.txt")
    parser.add_argument("--questions", default="sample_data/questions.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", choices=["single", "compare", "sweep"], default="compare")
    args = parser.parse_args()

    if args.mode == "single":
        run_eval(args.doc, args.questions, args.threshold)
    elif args.mode == "sweep":
        threshold_sweep(args.doc, args.questions)
    else:
        threshold_sweep(args.doc, args.questions)
        compare_standard_vs_confidence_aware(args.doc, args.questions, args.threshold)
