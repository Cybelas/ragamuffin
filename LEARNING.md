# Learning path

Treat this as a lab, not a tutorial to consume passively. For every session:

1. Predict what will happen.
2. Run an experiment.
3. Inspect the result.
4. Explain the result in your own words.
5. Change one variable and repeat.

## Session 1: see the whole loop

Run:

```powershell
kb ingest knowledge
kb search "Why do chunks overlap?" --top-k 3
kb ask "What can cause a RAG answer to be wrong?"
```

Explain, without looking:

- What gets stored in SQLite?
- Why is the question embedded too?
- What is the difference between retrieval and generation?
- Where could a hallucination enter the system?

If one answer is fuzzy, follow the code path until it is concrete.

## Session 2: break chunking on purpose

Open `src/knowledge_base/chunking.py`. Ingest a long note with different values:

```powershell
kb ingest knowledge --chunk-words 80 --overlap-words 0
kb search "a question whose answer crosses a chunk boundary"
```

Then repeat with overlap. Record which chunks were retrieved and why. Think
about headings, tables, code blocks, conversations, and meeting transcripts:
word windows will not treat all of them well.

## Session 3: test retrieval separately

Write ten questions for which your notes contain known answers. Before calling
`kb ask`, use `kb search` and record whether a useful chunk appears in the top
three results.

This matters because a language model cannot faithfully use evidence that was
never retrieved. A basic metric is `recall@3`: the fraction of questions for
which one of the top three chunks contains the needed evidence.

## Session 4: make one improvement

Choose exactly one:

- preserve Markdown heading context in each chunk;
- add a `kb inspect SOURCE` command;
- store page titles and timestamps as metadata;
- add lexical keyword search and merge it with vector results;
- create a CSV evaluation set and a command that calculates recall@k.

Write a test first. Keep the change small enough to explain at stand-up.

## Stand-up language

A useful update describes the learning and evidence, not just activity:

> I built the ingestion and retrieval path for a small RAG prototype. Documents
> are split into overlapping chunks, embedded, and stored in SQLite. I separated
> search from answer generation so I can measure retrieval failures directly.
> Next I am creating ten known-answer questions and measuring recall at three.

That is meaningful engineering progress even before an enterprise connector
exists.
