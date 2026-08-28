"""Simple chunking kept intentionally visible for learning."""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into fixed word windows with overlap.

    Production chunkers often understand headings, sentences, tables, and code.
    This first version uses word windows so the boundary behaviour is obvious.
    """

    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = re.findall(r"\S+", text)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words

    return chunks
