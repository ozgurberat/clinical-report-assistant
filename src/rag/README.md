# Phase 3 — RAG layer

## Indexing (done)

`build_index.py` embeds every report in `data/processed/reports.jsonl` — the
whole corpus, not a train/val/test split, since this simulates searching a
real historical case archive rather than training anything — and stores the
vectors in a local, embedded (no server process) Qdrant collection.

```bash
python -m src.rag.build_index --processed data/processed
```

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (small, CPU-friendly,
384-dim vectors — no GPU needed to build or query this index). Config in
`configs/rag.yaml`. The index is written to `data/processed/qdrant_index/`,
alongside the rest of the processed corpus, so it persists across Colab
sessions the same way everything else in Drive does.

Deliberately hand-rolled with `qdrant-client` + `sentence-transformers`
directly, rather than going through LangChain's retrieval-chain abstractions
(even though those are listed in the original architecture sketch). Given how
much API drift we already hit in Phase 2 with TRL, adding another fast-moving
framework layer on top wasn't worth the risk or the opacity — this way every
step (embed, store, search, build prompt, generate) is visible, plain code.

## Retrieval (done)

`retrieve.py` embeds a query the same way and searches the index:

```bash
python -m src.rag.retrieve --query "mild cardiomegaly, clear lungs" --top-k 5
```

Or as a library call: `from src.rag.retrieve import retrieve_similar`.
Returns each match's report_id, comparison/indication/findings/impression,
diagnosis list, and a cosine similarity score.

Uses `client.query_points(...)` rather than the deprecated `client.search(...)`,
and `create_collection`/`delete_collection` rather than the deprecated
`recreate_collection`, per current qdrant-client — worth flagging since this
project has already been bitten more than once by library APIs moving past
what any of us knew when the code was written.

## RAG-QA generation (done)

`qa.py` takes a user's question, calls `retrieve_similar()` to pull the
relevant past reports, builds a prompt embedding that retrieved evidence as
context, and generates an answer using the **plain base Qwen3-4B model with
no adapter attached** — not the extraction or summarization fine-tunes,
which are narrowly specialized for their own tasks and would try to force
any input into their trained output shape (see the earlier adapter-vs-base
experiment). The base instruct model already handles general open-ended
synthesis well; RAG's job is only to supply it with real, retrieved facts
instead of letting it guess.

```bash
python -m src.rag.qa --question "..." --show-reasoning
```

Deliberately the opposite choice from `evaluate.py` on one point: thinking
mode is left ON here (`enable_thinking=True`), not suppressed. Extraction and
summarization are single-step transformations our fine-tunes never modeled a
reasoning trace for, so an empty `<think>` block there was pure waste.
Synthesizing across several retrieved documents into one answer is exactly
the multi-step task thinking mode exists for, and this path never touches a
fine-tuned adapter, so there's no risk of an empty, untrained thinking block
— the base model's reasoning here is the real thing.
