"""
app.py
------
Local web UI for the Confidence-Aware Hybrid RAG pipeline, matching the
full architecture diagram:

  1. Document Ingestion (txt / docx / pdf upload)         -> ingestion.py
  2. Preprocessing & Chunking                              -> chunking.py
  3. Indexing (BM25 keyword index + semantic index)         -> retrieval.py
  4. Hybrid Retrieval (parallel BM25 + semantic, fused)      -> retrieval.py
  5. Confidence-Aware Abstention Layer (3 signals)           -> abstention.py
  6. Context-aware generation                               -> generation.py
  7. Answer + citations, OR abstain response

Run:
    pip install -r requirements.txt
    python3 app.py

Then open:
    http://localhost:5000

Upload your own .txt / .docx / .pdf in the browser, or start with the
bundled sample document. Ask any question, drag the threshold slider, and
see the composite score plus all three individual signals live.
"""

import os
import uuid
import tempfile

from dotenv import load_dotenv
load_dotenv()  # loads ANTHROPIC_API_KEY / RAG_USE_EMBEDDINGS from .env if present

from flask import Flask, request, render_template_string, redirect, url_for, session

from ingestion import extract_text, UnsupportedFileType
from chunking import chunk_document
from retrieval import HybridRetriever, SemanticIndex
from generation import get_default_generator
from abstention import ConfidenceAwareAbstention

DEFAULT_DOC_PATH = os.environ.get("RAG_DOC_PATH", "sample_data/document.txt")

# Phase 2 upgrade toggle: set RAG_USE_EMBEDDINGS=1 in .env (after
# `pip install sentence-transformers`) to use real neural embeddings for
# semantic search instead of TF-IDF/LSA. Falls back safely if the package
# isn't installed, so this is always safe to leave on.
if os.environ.get("RAG_USE_EMBEDDINGS", "0") == "1":
    try:
        from embeddings_retrieval import EmbeddingIndex as _SemanticIndexCls
        print("[app.py] Using EmbeddingIndex (sentence-transformers) for semantic search")
    except ImportError:
        print("[app.py] RAG_USE_EMBEDDINGS=1 but sentence-transformers isn't installed; "
              "falling back to TF-IDF/LSA. Run: pip install sentence-transformers")
        _SemanticIndexCls = SemanticIndex
else:
    _SemanticIndexCls = SemanticIndex
DEFAULT_THRESHOLD = 0.74
UPLOAD_DIR = tempfile.mkdtemp(prefix="rag_uploads_")
ALLOWED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", uuid.uuid4().hex)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_generator = get_default_generator()
_generator_name = type(_generator).__name__

# In-memory store of built indices per session document, keyed by a doc id.
# Single-user local app, so a simple dict is fine (no external DB needed).
_document_store = {}


def _build_index(doc_id: str, text: str, source_name: str):
    chunks = chunk_document(text, target_chunk_chars=300, overlap_sentences=0)
    retriever = HybridRetriever(chunks, semantic_index_cls=_SemanticIndexCls)
    _document_store[doc_id] = {
        "chunks": chunks,
        "retriever": retriever,
        "source_name": source_name,
        "example_questions": retriever.suggest_example_questions(n=5),
        # Start new documents at a more permissive threshold than the one
        # tuned for the bundled finance sample doc. Rank agreement between
        # BM25 and semantic search depends heavily on document structure
        # (continuous prose vs. scattered bullet points), so a threshold
        # calibrated on one document can be too strict for another. This is
        # just a friendlier starting point -- still adjust the slider and
        # watch the composite score to find what's right for YOUR document.
        "suggested_threshold": 0.5,
    }


def _ensure_default_document():
    if "default" not in _document_store:
        text = extract_text(DEFAULT_DOC_PATH)
        _build_index("default", text, os.path.basename(DEFAULT_DOC_PATH))
        # This is the document the threshold was actually tuned against
        # (see evaluate.py sweep) -- keep the calibrated value here.
        _document_store["default"]["suggested_threshold"] = DEFAULT_THRESHOLD


_ensure_default_document()


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Confidence-Aware RAG</title>
<style>
  :root {
    --bg: #0f1117; --panel: #171a23; --border: #262b3a; --text: #e7e9ee;
    --muted: #9096a8; --accent: #7c9eff; --green: #3ddc97; --amber: #f2b84b;
    --red: #ef6a6a;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 40px 20px; line-height: 1.5;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 18px;
  }
  .doc-panel { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .doc-info { font-size: 13.5px; color: var(--muted); }
  .doc-info b { color: var(--text); }
  .upload-form { display: flex; align-items: center; gap: 10px; }
  input[type=file] { font-size: 12.5px; color: var(--muted); max-width: 220px; }
  .btn-secondary {
    background: transparent; color: var(--accent); border: 1px solid var(--accent);
    padding: 7px 14px; border-radius: 8px; font-size: 13px; cursor: pointer; font-weight: 600;
  }
  .btn-secondary:hover { background: rgba(124,158,255,0.1); }
  .flash { color: var(--red); font-size: 13px; margin-top: 8px; }
  input[type=text] {
    width: 100%; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: #0d0f16; color: var(--text); font-size: 15px; margin-bottom: 14px;
  }
  .row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .row label { color: var(--muted); font-size: 13px; white-space: nowrap; }
  input[type=range] { flex: 1; }
  button.primary {
    background: var(--accent); color: #0d0f16; border: none; padding: 10px 20px;
    border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
  }
  button.primary:hover { opacity: 0.9; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 22px; margin-bottom: 16px;
  }
  .decision {
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
    font-size: 13px; letter-spacing: 0.3px; margin-bottom: 14px;
  }
  .decision.answer { background: rgba(61,220,151,0.15); color: var(--green); }
  .decision.abstain { background: rgba(239,106,106,0.15); color: var(--red); }
  .answer-text { font-size: 15.5px; margin-bottom: 18px; }
  .score-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 6px;
  }
  .score-box { background: #0d0f16; border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
  .score-label { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 6px; }
  .score-value { font-size: 20px; font-weight: 700; }
  .score-value.good { color: var(--green); }
  .score-value.mid { color: var(--amber); }
  .score-value.low { color: var(--red); }
  .bar-track { height: 6px; background: #262b3a; border-radius: 4px; margin-top: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .chunks { font-size: 13px; color: var(--muted); }
  .chunk { background: #0d0f16; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-top: 8px; }
  .chunk .meta { color: var(--accent); font-size: 11.5px; margin-bottom: 4px; }
  .meta-row { color: var(--muted); font-size: 12.5px; margin-top: 10px; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .example {
    background: var(--panel); border: 1px solid var(--border); color: var(--muted);
    padding: 6px 12px; border-radius: 999px; font-size: 12.5px; cursor: pointer;
  }
  .example:hover { border-color: var(--accent); color: var(--text); }
  .pipeline-steps { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; color: var(--muted); margin-bottom: 20px; }
  .pipeline-steps span { background: var(--panel); border: 1px solid var(--border); padding: 4px 10px; border-radius: 999px; }
  .pipeline-steps span.active { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Confidence-Aware Hybrid RAG</h1>
  <div class="sub">BM25 + Semantic Retrieval &rarr; Confidence-Aware Abstention Layer &rarr; Answer or Abstain</div>

  <div class="pipeline-steps">
    <span>1. Ingestion</span><span>2. Chunking</span><span>3. Indexing (BM25 + Semantic)</span>
    <span>4. Hybrid Retrieval</span><span class="active">5. Abstention Layer</span><span>6. Generation</span><span>7. Answer / Abstain</span>
  </div>

  <div class="panel doc-panel">
    <div class="doc-info">
      Current document: <b>{{ source_name }}</b> &middot; {{ n_chunks }} chunks &middot; generator: {{ generator_name }}
    </div>
    <form class="upload-form" method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
      <input type="file" name="document" accept=".txt,.md,.docx,.pdf" required>
      <button type="submit" class="btn-secondary">Upload &amp; Re-index</button>
    </form>
  </div>
  {% if upload_error %}<div class="flash">{{ upload_error }}</div>{% endif %}

  <div class="examples">
    {% for ex in example_questions %}
      <span class="example" onclick="document.getElementById('q').value=this.textContent; document.getElementById('f').submit();">{{ ex }}</span>
    {% endfor %}
  </div>

  <form method="post" action="{{ url_for('index') }}" id="f">
    <input type="text" name="question" id="q" placeholder="Ask anything about the document..." value="{{ question or '' }}" autofocus>
    <div class="row">
      <label>Abstention threshold: {{ "%.2f"|format(threshold) }}</label>
      <input type="range" name="threshold" min="0" max="1" step="0.01" value="{{ threshold }}"
             oninput="this.previousElementSibling.textContent='Abstention threshold: ' + parseFloat(this.value).toFixed(2)">
    </div>
    <button type="submit" class="primary">Ask</button>
  </form>

  {% if result %}
  <div class="card">
    <span class="decision {{ result.decision }}">{{ "ANSWERED" if result.decision == "answer" else "ABSTAINED" }}</span>
    <div class="answer-text">{{ result.answer_text }}</div>

    <div class="score-grid">
      <div class="score-box">
        <div class="score-label">Composite</div>
        <div class="score-value {{ result.composite_class }}">{{ "%.2f"|format(result.composite) }}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{{ result.composite*100 }}%; background:{{ result.composite_color }};"></div></div>
      </div>
      <div class="score-box">
        <div class="score-label">Retrieval (S1)</div>
        <div class="score-value {{ result.retrieval_class }}">{{ "%.2f"|format(result.retrieval) }}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{{ result.retrieval*100 }}%; background:{{ result.retrieval_color }};"></div></div>
      </div>
      <div class="score-box">
        <div class="score-label">Rank Agree (S2)</div>
        <div class="score-value {{ result.rank_class }}">{{ "%.2f"|format(result.rank) }}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{{ result.rank*100 }}%; background:{{ result.rank_color }};"></div></div>
      </div>
      <div class="score-box">
        <div class="score-label">Grounded (S3)</div>
        <div class="score-value {{ result.ground_class }}">{{ "%.2f"|format(result.ground) }}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{{ result.ground*100 }}%; background:{{ result.ground_color }};"></div></div>
      </div>
    </div>

    <div class="meta-row">Threshold used: {{ "%.2f"|format(result.threshold) }} &middot; decision = composite {{ "&ge;" | safe if result.decision == "answer" else "&lt;" | safe }} threshold</div>

    <div class="chunks">
      <div class="meta-row" style="margin-top:16px;">Retrieved chunks used (with per-signal scores):</div>
      {% for c in result.chunks %}
        <div class="chunk">
          <div class="meta">chunk #{{ c.id }} &middot; fused={{ "%.2f"|format(c.fused) }} bm25={{ "%.2f"|format(c.bm25) }} sem={{ "%.2f"|format(c.sem) }}</div>
          {{ c.text }}
        </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div>
</body>
</html>
"""

EXAMPLE_QUESTIONS_FALLBACK = ["What is this document about?"]


def _score_class(score: float) -> str:
    if score >= 0.7:
        return "good"
    if score >= 0.4:
        return "mid"
    return "low"


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "#3ddc97"
    if score >= 0.4:
        return "#f2b84b"
    return "#ef6a6a"


def _current_doc_id():
    return session.get("doc_id", "default")


def run_query(doc_id: str, question: str, threshold: float, top_k: int = 3):
    entry = _document_store.get(doc_id) or _document_store["default"]
    retriever: HybridRetriever = entry["retriever"]

    retrieval = retriever.retrieve(question, top_k=top_k)
    top_chunks = [rc.chunk for rc in retrieval["top_chunks"]]
    top_fused_score = retrieval["top_chunks"][0].fused_score if retrieval["top_chunks"] else 0.0

    candidate_answer = _generator.generate(question, top_chunks) if top_chunks else None

    layer = ConfidenceAwareAbstention(threshold=threshold)
    confidence = layer.evaluate(
        top_fused_score=top_fused_score,
        bm25_ranking=retrieval["bm25_ranking"],
        semantic_ranking=retrieval["semantic_ranking"],
        answer=candidate_answer,
        source_chunks=top_chunks,
    )

    decision = confidence.decision
    answer_text = (
        candidate_answer.text if (decision == "answer" and candidate_answer)
        else "Insufficient information found in the document to answer this confidently."
    )

    return {
        "decision": decision,
        "answer_text": answer_text,
        "composite": confidence.composite_score,
        "composite_class": _score_class(confidence.composite_score),
        "composite_color": _score_color(confidence.composite_score),
        "retrieval": confidence.retrieval_strength,
        "retrieval_class": _score_class(confidence.retrieval_strength),
        "retrieval_color": _score_color(confidence.retrieval_strength),
        "rank": confidence.rank_agreement,
        "rank_class": _score_class(confidence.rank_agreement),
        "rank_color": _score_color(confidence.rank_agreement),
        "ground": confidence.citation_groundedness,
        "ground_class": _score_class(confidence.citation_groundedness),
        "ground_color": _score_color(confidence.citation_groundedness),
        "threshold": threshold,
        "chunks": [
            {"id": rc.chunk.id, "fused": rc.fused_score, "bm25": rc.bm25_score, "sem": rc.semantic_score, "text": rc.chunk.text}
            for rc in retrieval["top_chunks"]
        ],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    doc_id = _current_doc_id()
    entry = _document_store.get(doc_id) or _document_store["default"]

    question = None
    threshold = entry.get("suggested_threshold", DEFAULT_THRESHOLD)
    result = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        threshold = float(request.form.get("threshold", threshold))
        if question:
            result = run_query(doc_id, question, threshold)

    return render_template_string(
        PAGE,
        source_name=entry["source_name"],
        n_chunks=len(entry["chunks"]),
        generator_name=_generator_name,
        question=question,
        threshold=threshold,
        result=result,
        example_questions=entry.get("example_questions", EXAMPLE_QUESTIONS_FALLBACK),
        upload_error=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("document")
    entry = _document_store.get(_current_doc_id()) or _document_store["default"]

    if not file or file.filename == "":
        return render_template_string(
            PAGE, source_name=entry["source_name"], n_chunks=len(entry["chunks"]),
            generator_name=_generator_name, question=None, threshold=entry.get("suggested_threshold", DEFAULT_THRESHOLD),
            result=None, example_questions=entry.get("example_questions", EXAMPLE_QUESTIONS_FALLBACK), upload_error="No file selected.",
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template_string(
            PAGE, source_name=entry["source_name"], n_chunks=len(entry["chunks"]),
            generator_name=_generator_name, question=None, threshold=entry.get("suggested_threshold", DEFAULT_THRESHOLD),
            result=None, example_questions=entry.get("example_questions", EXAMPLE_QUESTIONS_FALLBACK),
            upload_error=f"Unsupported file type '{ext}'. Use .txt, .docx, or .pdf.",
        )

    doc_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")
    file.save(saved_path)

    try:
        text = extract_text(saved_path)
        if not text.strip():
            raise ValueError("No extractable text found in this file (it may be a scanned/image-only PDF).")
        _build_index(doc_id, text, file.filename)
    except (UnsupportedFileType, ValueError) as e:
        return render_template_string(
            PAGE, source_name=entry["source_name"], n_chunks=len(entry["chunks"]),
            generator_name=_generator_name, question=None, threshold=entry.get("suggested_threshold", DEFAULT_THRESHOLD),
            result=None, example_questions=entry.get("example_questions", EXAMPLE_QUESTIONS_FALLBACK), upload_error=str(e),
        )
    finally:
        try:
            os.remove(saved_path)
        except OSError:
            pass

    session["doc_id"] = doc_id
    return redirect(url_for("index"))


if __name__ == "__main__":
    print(f"\nLoaded default document: {DEFAULT_DOC_PATH}")
    print(f"Generator: {_generator_name}")
    print("Open http://localhost:5000 in your browser")
    print("Upload your own .txt / .docx / .pdf from the page to re-index on the fly.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
