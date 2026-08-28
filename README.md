# Ragamuffin

A deliberately small retrieval-augmented generation (RAG) project. It turns
plain-text notes into embeddings, stores them in SQLite, retrieves relevant
chunks with cosine similarity, and asks a language model to answer using only
those chunks.

The goal is not to hide the system behind a framework. The goal is to make each
step small enough that you can read it, change it, break it, and explain it.

## The mental model

```text
INGESTION
files -> text -> overlapping chunks -> embedding vectors -> SQLite

QUESTION TIME
question -> embedding vector -> nearest chunks -> prompt + context -> answer
```

RAG has two distinct jobs:

1. **Retrieval** finds likely evidence. Embeddings make text with similar
   meaning land near each other in a high-dimensional space.
2. **Generation** turns the retrieved evidence into a useful answer. It must be
   told to stay grounded and cite its sources.

## Safety boundary

This learning version sends ingested text and questions to the OpenAI API. Use
only synthetic, public, personal, or explicitly approved data. Do **not** ingest
company Slack, Teams, Confluence, SharePoint, credentials, customer data, or
other internal material until your organisation has approved the architecture.

An enterprise version also needs source permissions, document-level access
control, retention/deletion, audit logging, secret management, and an approved
model/data-processing path. Retrieval without permission filtering can expose
information even if the original system was locked down.

## Setup (PowerShell)

You need Python 3.11 or newer and an OpenAI API key. Clone the repository,
create an isolated virtual environment, and install the project:

```powershell
git clone https://github.com/Cybelas/ragamuffin.git
cd ragamuffin
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
$env:OPENAI_API_KEY = "your-api-key"
```

The SDK reads `OPENAI_API_KEY` from the environment. Do not commit API keys.

## Run the first experiment

```powershell
kb ingest knowledge
kb stats
kb search "Why do chunks overlap?"
kb ask "Explain the two stages of RAG and cite the notes."
```

The default models can be changed without editing code:

```powershell
$env:KB_EMBEDDING_MODEL = "text-embedding-3-small"
$env:KB_CHAT_MODEL = "gpt-5.4-mini"
```

The database is created at `.kb/knowledge.db`. Re-ingesting an unchanged file
skips it. If its contents or embedding model change, that document is replaced.

## Read the implementation in this order

1. `src/knowledge_base/chunking.py` — turn one document into windows of words.
2. `src/knowledge_base/openai_provider.py` — create vectors and grounded answers.
3. `src/knowledge_base/store.py` — persist vectors and calculate similarity.
4. `src/knowledge_base/cli.py` — connect the pipeline into commands.
5. `tests/` — examples of the behaviour we expect.

Then work through `LEARNING.md`.

## Commands

```text
kb ingest PATH       Ingest .md and .txt files from a file or directory
kb search QUERY      Show retrieval results without generating an answer
kb ask QUESTION      Retrieve evidence and generate a cited answer
kb stats             Show the number of indexed documents and chunks
```

Use `kb --help` or `kb <command> --help` for all options.

## Sensible roadmap

1. Measure retrieval quality with a small question-and-answer evaluation set.
2. Improve chunking and add metadata filters.
3. Add PDF/HTML parsing and deletion synchronisation.
4. Add one approved source connector, probably Confluence, with permission data.
5. Add hybrid keyword + vector search, reranking, observability, and access tests.

Starting with every enterprise source at once would make it hard to know what
you actually learned or why an answer failed. One source and ten test questions
will teach you more.
