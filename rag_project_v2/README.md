# Hybrid RAG with Confidence-Aware Abstention — Working Prototype

This is a runnable implementation of the system described in your proposal:
a hybrid (BM25 + semantic) RAG pipeline with a **Confidence-Aware Abstention
Layer** that decides whether to answer at all, and an evaluation harness
that scores those decisions with precision/recall/F1 ("abstention
accuracy") — not just answer correctness.

## Quick start

```bash
pip install -r requirements.txt
python3 pipeline.py                 # single end-to-end demo run
python3 evaluate.py --mode sweep    # threshold calibration sweep
python3 evaluate.py --mode compare --threshold 0.74   # standard vs confidence-aware
```

## Files

| File | Role |
|---|---|
| `chunking.py` | Sentence-aware document chunking with overlap |
| `ingestion.py` | Stage-1 document ingestion: extracts clean text from `.txt`, `.docx`, and `.pdf` |
| `retrieval.py` | `HybridRetriever`: BM25 (`rank_bm25`) + TF-IDF/LSA semantic search, fused |
| `generation.py` | Pluggable generator: `ExtractiveGenerator` (no LLM) or `ClaudeGenerator` (real Anthropic API, if `ANTHROPIC_API_KEY` is set) |
| `abstention.py` | **The core contribution.** `ConfidenceAwareAbstention` — the three signals + composite scoring |
| `pipeline.py` | Wires it all together, matching the architecture diagram in the proposal |
| `app.py` | **Local web UI** (Flask, `http://localhost:5000`): upload any `.txt`/`.docx`/`.pdf`, ask any question, see the composite score + all three signals live |
| `evaluate.py` | Runs the labeled question set, computes abstention precision/recall/F1, compares against a "never abstains" baseline |
| `sample_data/document.txt` | Sample earnings-report document (financial domain, as suggested in the proposal) |
| `sample_data/questions.json` | 20 labeled questions: 10 answerable, 10 deliberately unanswerable |

## Web UI (upload your own document)

```bash
pip install -r requirements.txt
python3 app.py
```

Open `http://localhost:5000`. You'll land on the bundled sample document by
default. Use the upload box at the top to drop in your own `.txt`, `.docx`,
or `.pdf` — it gets extracted (via `ingestion.py`), chunked, and re-indexed
immediately, and all subsequent questions run against that document instead.
Each answer shows the composite confidence score plus the three individual
signals (S1 retrieval strength, S2 rank agreement, S3 citation groundedness),
color-coded, along with the actual retrieved chunks and their per-signal
scores. A threshold slider lets you watch decisions flip between "answer"
and "abstain" in real time.

Scanned/image-only PDFs won't extract usable text (no OCR is included) —
the app will show an error rather than silently indexing an empty document.


## The three confidence signals (`abstention.py`)

1. **Retrieval strength** — the fused top-1 score, on an *absolute* scale
   (saturating normalization: `score / (score + k)`), not min-max across
   this query's own candidates. This distinction matters: min-max would
   make the best-matching chunk score 1.0 by definition even if nothing in
   the document is actually relevant — that would make the signal useless.
2. **Rank agreement** — do BM25 and TF-IDF/semantic search independently
   converge on the same top chunk (weighted higher) and overlap in their
   top-k sets (Jaccard, softer signal)?
3. **Citation groundedness** — after generation, are the numeric/entity
   claims in the answer actually traceable to the retrieved chunks? This is
   the direct hallucination check, run post-hoc on the generated text.

These combine into a weighted composite score in `[0, 1]`; below
`threshold` (default 0.55, tune via the sweep) the system abstains instead
of answering.

## What the evaluation actually shows

Running `evaluate.py --mode compare --threshold 0.74` on the sample data:

| Metric | Standard RAG (never abstains) | Confidence-Aware RAG |
|---|---|---|
| Hallucination rate (answered when it shouldn't) | 1.00 | 0.40 |
| Abstention F1 | 0.67 | 0.73 |
| Answer correctness (on genuinely answerable Qs) | 0.80 | 0.80 |

This is the central result your proposal is after: the abstention layer
substantially cuts hallucinated answers on unanswerable questions **without
sacrificing correctness** on questions it can genuinely support. Run
`--mode sweep` to see the full precision/recall/F1 curve across thresholds
— this is what you'd report in the "Evaluation Strategy" section of your
final report, and it's genuinely a trade-off curve, not a single cherry-picked
number.

## Known limitations (worth naming explicitly in your report)

- **`ExtractiveGenerator` doesn't reason about relevance.** It returns the
  top retrieved chunk regardless of whether it actually answers the
  question, so citation-groundedness trivially passes (the "claims" are
  literally copied from the source). This caps precision — you can see it
  in the eval table (some unanswerable questions still get answered because
  a topically-adjacent chunk was retrieved). Swap in `ClaudeGenerator`
  (works out of the box if you set `ANTHROPIC_API_KEY`) for an LLM that can
  actually recognize "this chunk doesn't answer the question" — this should
  meaningfully improve precision, and is a natural next experiment for your
  report (compare abstention accuracy with extractive vs. LLM generation).
- **Semantic search uses TF-IDF/LSA, not neural embeddings**, since this
  sandbox has no access to model hubs. It's a legitimate lightweight
  "semantic" signal but weaker than e.g. a sentence-transformer. Swapping
  `SemanticIndex`'s internals for a real embedding model is a drop-in
  change — nothing else in the pipeline needs to know.
- **Citation groundedness is lexical** (numbers + proper nouns must appear
  verbatim in the source chunks), not a full entailment check. It catches
  fabricated numbers/entities well but won't catch a subtler factual
  distortion that reuses the source's own vocabulary. An NLI-based
  entailment model would be the natural upgrade, and is worth flagging as
  future work in your report.
- **Small sample corpus (13 chunks).** The 20-question eval set is enough
  to demonstrate the methodology and produce a real (if noisy) F1 curve,
  but you'll want a larger, more diverse document + question set for the
  final report to make the numbers more statistically convincing.

## Extending this

- Swap in `ClaudeGenerator` by setting `ANTHROPIC_API_KEY` in your
  environment — no other code changes needed.
- To test on your own document: point `pipeline.py` / `evaluate.py` at a
  different `.txt` file and write your own `questions.json` with
  `should_answer` ground-truth labels.
- Threshold and signal weights are all constructor arguments on
  `ConfidenceAwareAbstention` — tune them against your own eval set rather
  than reusing the defaults tuned here.
