"""SQLite persistence and deliberately simple vector search."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Sequence


@dataclass(frozen=True)
class SearchResult:
    source: str
    chunk_index: int
    text: str
    score: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two vectors."""

    if len(left) != len(right):
        raise ValueError("vectors must have the same number of dimensions")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length == 0 or right_length == 0:
        return 0.0
    return dot_product / (left_length * right_length)


class KnowledgeStore:
    """Store documents and scan their embeddings for nearest neighbours.

    A full scan is intentionally fine for a learning-sized corpus. A vector
    index becomes useful when measurement shows that this is too slow.
    """

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def document_is_current(
        self,
        source: str,
        content_hash: str,
        embedding_model: str,
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM documents
            WHERE source = ? AND content_hash = ? AND embedding_model = ?
            """,
            (source, content_hash, embedding_model),
        ).fetchone()
        return row is not None

    def replace_document(
        self,
        *,
        source: str,
        content_hash: str,
        embedding_model: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("every chunk must have one embedding")

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO documents (source, content_hash, embedding_model)
                VALUES (?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (source, content_hash, embedding_model),
            )
            document_id = self.connection.execute(
                "SELECT id FROM documents WHERE source = ?", (source,)
            ).fetchone()["id"]
            self.connection.execute(
                "DELETE FROM chunks WHERE document_id = ?", (document_id,)
            )
            self.connection.executemany(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, text, embedding_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        index,
                        text,
                        json.dumps(list(embedding), separators=(",", ":")),
                    )
                    for index, (text, embedding) in enumerate(
                        zip(chunks, embeddings, strict=True)
                    )
                ],
            )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        top_k: int = 5,
	min_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        rows = self.connection.execute(
            """
            SELECT d.source, c.chunk_index, c.text, c.embedding_json
            FROM chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE d.embedding_model = ?
            """,
            (embedding_model,),
        ).fetchall()

        results = [
            SearchResult(
                source=row["source"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                score=cosine_similarity(
                    query_embedding, json.loads(row["embedding_json"])
                ),
            )
            for row in rows
        ]
        results.sort(key=lambda result: result.score, reverse=True)

        if min_score is not None:
            results = [
                result for result in results if result.score >= min_score
            ]

        return results[:top_k]

    def stats(self) -> tuple[int, int]:
        document_count = self.connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        chunk_count = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        return document_count, chunk_count
