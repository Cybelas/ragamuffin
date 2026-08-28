"""The only module that talks to the OpenAI API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .store import SearchResult


class OpenAIProvider:
    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        chat_model: str | None = None,
    ) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your shell environment; "
                "never put it in source code."
            )

        from openai import OpenAI

        self.client = OpenAI()
        self.embedding_model = embedding_model or os.environ.get(
            "KB_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.chat_model = chat_model or os.environ.get(
            "KB_CHAT_MODEL", "gpt-5.4-mini"
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text in batches while preserving input order."""

        if not texts:
            return []

        embeddings: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in ordered)
        return embeddings

    def answer(self, question: str, results: Sequence[SearchResult]) -> str:
        context_parts = []
        for number, result in enumerate(results, start=1):
            context_parts.append(
                f"[S{number}] source={Path(result.source).name} "
                f"chunk={result.chunk_index}\n{result.text}"
            )
        context = "\n\n".join(context_parts)

        response = self.client.responses.create(
            model=self.chat_model,
            instructions=(
                "Answer using only the supplied sources. Cite factual claims "
                "with [S1], [S2], and so on. If the sources do not contain "
                "enough evidence, say so plainly. Do not use unstated memory."
            ),
            input=f"Question:\n{question}\n\nSources:\n{context}",
            max_output_tokens=600,
            store=False,
        )
        return response.output_text
