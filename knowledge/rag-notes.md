# RAG notes

Retrieval-augmented generation, or RAG, combines information retrieval with a
language model. The retrieval stage finds evidence that may answer a question.
The generation stage receives that evidence as context and writes an answer.

## Embeddings

An embedding is a list of numbers representing aspects of a piece of content.
Text with related meaning tends to have vectors that are closer together. The
same embedding model must be used for stored chunks and incoming questions so
their vectors live in the same coordinate system.

Embeddings do not contain a readable answer and do not prove that two passages
mean the same thing. They provide a useful similarity signal. Retrieval can
still fail because a document was missing, parsing lost information, chunks
were badly formed, the query was ambiguous, or semantic similarity was not the
right signal.

## Chunking

Embedding an entire large document gives only one coarse vector and can exceed
model input limits. Instead, an ingestion pipeline divides the document into
smaller chunks. Smaller chunks can be more precise, but chunks that are too
small may lose the context required to understand them.

Adjacent chunks often overlap. Overlap reduces the chance that a fact and its
explanation are separated at a hard boundary. More overlap also duplicates
content, increases embedding and storage cost, and may produce repetitive
retrieval results. Chunk size and overlap should be evaluated on real questions
rather than selected by folklore.

## Retrieval and generation failures

A wrong RAG answer can come from either half of the system. Retrieval may return
irrelevant evidence or miss the needed passage. Generation may ignore good
evidence, overstate it, combine sources incorrectly, or answer from model memory.
Testing retrieval and generation separately makes failures easier to diagnose.

Source citations help a person check an answer, but citations alone do not make
it correct. A useful system preserves source identity, retrieves only material
the user is authorised to read, and says when the evidence is insufficient.
