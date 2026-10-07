"""
generation.py
-------------
Pluggable answer generation.

Two backends:
  - ExtractiveGenerator: no external calls. Picks the most query-relevant
    sentence(s) from the retrieved chunks. Always available, deterministic,
    good for testing the abstention layer in isolation from LLM behavior.
  - ClaudeGenerator: calls the real Anthropic API (model: claude-sonnet-4-6)
    to synthesize a grounded answer from the retrieved chunks, used only if
    ANTHROPIC_API_KEY is set in the environment. Falls back to extractive
    generation otherwise.

Both return an Answer(text, source_chunk_ids) so the abstention layer's
citation-groundedness check has something concrete to verify against.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List

from chunking import Chunk


@dataclass
class Answer:
    text: str
    source_chunk_ids: List[int]


class ExtractiveGenerator:
    """Very simple, transparent baseline generator: no LLM, no hallucination risk
    by construction (useful as a control condition in evaluation)."""

    def generate(self, query: str, chunks: List[Chunk]) -> Answer:
        if not chunks:
            return Answer(text="", source_chunk_ids=[])
        # naive: just concatenate the top chunk's text (trimmed) as the "answer"
        top = chunks[0]
        text = top.text
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + "..."
        return Answer(text=text, source_chunk_ids=[c.id for c in chunks])


class ClaudeGenerator:
    """Calls the Anthropic Messages API to produce a grounded answer.
    Requires ANTHROPIC_API_KEY in the environment. Uses only the retrieved
    chunks as context and explicitly instructs the model not to use outside
    knowledge, so citation-groundedness checking downstream is meaningful."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, query: str, chunks: List[Chunk]) -> Answer:
        if not self.available():
            raise RuntimeError("ANTHROPIC_API_KEY not set; ClaudeGenerator unavailable.")

        import urllib.request

        context = "\n\n".join(f"[Chunk {c.id}] {c.text}" for c in chunks)
        prompt = (
            "Answer the question using ONLY the numbered chunks below. "
            "Do not use outside knowledge. If the chunks don't contain the answer, "
            "say so explicitly.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )

        body = json.dumps({
            "model": self.model,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        return Answer(text=text.strip(), source_chunk_ids=[c.id for c in chunks])


def get_default_generator():
    """Returns ClaudeGenerator if an API key is configured, else ExtractiveGenerator."""
    claude_gen = ClaudeGenerator()
    if claude_gen.available():
        return claude_gen
    return ExtractiveGenerator()


if __name__ == "__main__":
    chunks = [
        Chunk(id=0, text="Net income for the quarter was $8.4 million.", start_char=0, end_char=10),
    ]
    gen = get_default_generator()
    print(type(gen).__name__)
    print(gen.generate("What was net income?", chunks))
