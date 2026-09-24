"""Command-line entry point for the knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

from .chunking import chunk_text
from .openai_provider import OpenAIProvider
from .store import KnowledgeStore, SearchResult


SUPPORTED_SUFFIXES = {".md", ".txt"}


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not path.is_dir():
        return []

    files: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative_parts = candidate.relative_to(path).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        files.append(candidate)
    return sorted(files)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_label(result: SearchResult) -> str:
    return f"{Path(result.source).name}#chunk-{result.chunk_index}"


def print_results(results: Iterable[SearchResult]) -> None:
    for number, result in enumerate(results, start=1):
        preview = result.text.replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:237] + "..."
        print(
            f"{number}. score={result.score:.4f} "
            f"source={source_label(result)}\n   {preview}"
        )


def no_results_message(min_score: float | None) -> str:
    if min_score is None:
        return "No indexed chunks found. Run `kb ingest PATH` first."

    return (
        f"No chunks met --min-score {min_score:.4f}. "
        "Check the index or try a lower threshold."
    )


def embedding_model_from_environment() -> str:
    return os.environ.get("KB_EMBEDDING_MODEL", "text-embedding-3-small")


def run_ingest(args: argparse.Namespace, store: KnowledgeStore) -> int:
    path = Path(args.path)
    files = discover_files(path)
    if not files:
        print(f"No .md or .txt files found at {path}", file=sys.stderr)
        return 1

    embedding_model = embedding_model_from_environment()
    pending: list[tuple[Path, str, str, list[str]]] = []
    skipped = 0
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        digest = content_hash(text)
        source = str(file_path.resolve())
        if store.document_is_current(source, digest, embedding_model):
            skipped += 1
            continue
        chunks = chunk_text(
            text,
            max_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        pending.append((file_path, source, digest, chunks))

    if not pending:
        print(f"Nothing to update ({skipped} unchanged file(s)).")
        return 0

    provider = OpenAIProvider(embedding_model=embedding_model)
    indexed_chunks = 0
    for file_path, source, digest, chunks in pending:
        embeddings = provider.embed(chunks)
        store.replace_document(
            source=source,
            content_hash=digest,
            embedding_model=provider.embedding_model,
            chunks=chunks,
            embeddings=embeddings,
        )
        indexed_chunks += len(chunks)
        print(f"Indexed {file_path} ({len(chunks)} chunk(s))")

    print(
        f"Done: {len(pending)} file(s), {indexed_chunks} chunk(s), "
        f"{skipped} unchanged."
    )
    return 0


def retrieve(
    query: str,
    top_k: int,
    store: KnowledgeStore,
    *,
    min_score: float | None = None,
) -> tuple[OpenAIProvider, list[SearchResult]]:
    provider = OpenAIProvider()
    query_embedding = provider.embed([query])[0]
    results = store.search(
        query_embedding,
        embedding_model=provider.embedding_model,
        top_k=top_k,
        min_score=min_score,
    )
    return provider, results


def run_search(args: argparse.Namespace, store: KnowledgeStore) -> int:
    _, results = retrieve(
        args.query,
        args.top_k,
        store,
        min_score=args.min_score,
    )
    if not results:
        print(no_results_message(args.min_score))
        return 1

    print_results(results)
    return 0


def run_ask(args: argparse.Namespace, store: KnowledgeStore) -> int:
    provider, results = retrieve(
        args.question,
        args.top_k,
        store,
        min_score=args.min_score,
    )
    if not results:
        print(no_results_message(args.min_score))
        return 1

    print(provider.answer(args.question, results))
    print("\nRetrieved sources:")
    for number, result in enumerate(results, start=1):
        print(f"[S{number}] {source_label(result)} (score={result.score:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="A small RAG knowledge base for learning by building.",
    )
    parser.add_argument(
        "--db",
        default=".kb/knowledge.db",
        help="SQLite database path (default: .kb/knowledge.db)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Embed .md and .txt files")
    ingest.add_argument("path", help="File or directory to ingest")
    ingest.add_argument("--chunk-words", type=int, default=220)
    ingest.add_argument("--overlap-words", type=int, default=40)
    ingest.set_defaults(handler=run_ingest)

    search = subparsers.add_parser(
        "search", help="Retrieve similar chunks without generating an answer"
    )
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Exclude chunks below this cosine similarity score",
    )
    search.set_defaults(handler=run_search)

    ask = subparsers.add_parser("ask", help="Retrieve chunks and answer")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Exclude chunks below this cosine similarity score",
    )
    ask.set_defaults(handler=run_ask)

    stats = subparsers.add_parser("stats", help="Show index counts")
    stats.set_defaults(handler=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = KnowledgeStore(args.db)
    try:
        if args.command == "stats":
            documents, chunks = store.stats()
            print(f"Documents: {documents}\nChunks: {chunks}")
            return 0
        return args.handler(args, store)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
