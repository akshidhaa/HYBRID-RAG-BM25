"""
chunking.py
-----------
Splits raw document text into overlapping chunks suitable for retrieval.

Design notes:
- Sentence-aware chunking (doesn't cut mid-sentence where possible).
- Overlap between consecutive chunks preserves context across boundaries,
  which matters for the "citation groundedness" check later on: we don't
  want a fact split awkwardly across two chunks with no shared context.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: int
    text: str
    start_char: int
    end_char: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    text: str,
    target_chunk_chars: int = 800,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    """
    Groups sentences into chunks of roughly `target_chunk_chars` characters,
    carrying `overlap_sentences` sentences forward into the next chunk.
    """
    sentences = split_into_sentences(text)
    chunks: List[Chunk] = []

    current_sentences: List[str] = []
    current_len = 0
    cursor = 0  # approximate char offset tracker

    def flush(next_start_sentences):
        nonlocal current_sentences, current_len
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences)
        start = cursor
        end = start + len(chunk_text)
        chunks.append(Chunk(id=len(chunks), text=chunk_text, start_char=start, end_char=end))
        # carry overlap forward
        current_sentences = next_start_sentences
        current_len = sum(len(s) for s in current_sentences)

    for sent in sentences:
        if current_len + len(sent) > target_chunk_chars and current_sentences:
            overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            flush(overlap)
        current_sentences.append(sent)
        current_len += len(sent)

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(id=len(chunks), text=chunk_text, start_char=0, end_char=len(chunk_text)))

    return chunks


if __name__ == "__main__":
    sample = (
        "The company's revenue in Q3 2025 was $42 million. This was a 12% increase "
        "year over year. Operating expenses rose to $30 million, driven mostly by "
        "hiring in the engineering department. Net income for the quarter was $8.4 million. "
        "The board approved a new stock buyback program worth $50 million."
    )
    for c in chunk_document(sample, target_chunk_chars=120, overlap_sentences=1):
        print(c.id, "|", c.text)
