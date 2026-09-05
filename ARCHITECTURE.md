# Architecture

Local-first semantic search over ~260k iMessages. No message text ever leaves the
machine — not to an embedding API, not to a reranker, not in a commit.

The design follows the pipeline Tobi Lütke's [qmd](https://github.com/tobi/qmd)
converged on after ~3 months of eval-driven iteration, adapted from markdown
documents to chat messages. Where qmd's choices are load-bearing, they are cited.

## Pipeline

```
query
  ├─ expansion (optional, stage 4)  ──┐
  │                                   │
  ├─ BM25 / FTS5        ──┐           │
  ├─ dense / sqlite-vec  ──┤           │
  │                       ▼           │
  │              RRF fusion (k=60) ◄──┘
  │                       │
  │                 top ~30 candidates
  │                       ▼
  └────────────► cross-encoder rerank
                          │
                    ranked results
```

## Storage: one SQLite file

`index.sqlite` holds everything — documents, FTS5 lexical index, and dense vectors.
qmd makes the same call, and it is the right one here: no separate vector service to
run, atomic rebuilds, and the whole index is one file you can delete.

| table | role |
|---|---|
| `messages` | decoded body + chat/handle/timestamp metadata |
| `messages_fts` | FTS5 external-content index (porter unicode61) |
| `message_vectors` | chunk embeddings *(stage 2)* |
| `vectors_vec` | sqlite-vec ANN index *(stage 2)* |

## Chunking: messages are not documents

This is the main departure from qmd. qmd chunks documents into ~900-token windows
with 15% overlap. iMessages are the opposite problem — the median message is under
20 tokens, far too short to embed meaningfully in isolation ("yeah", "ok sounds
good", "that one").

So the unit of retrieval is a **conversation window**, not a message: consecutive
messages in one chat inside a time gap (start at 30 min, tune against the eval),
capped at ~900 tokens, with one message of overlap. A hit returns the window; the
UI anchors on the best-matching message inside it.

This matters more than any model choice. A query like "when we agreed to split the
rent" is answered by an exchange, not a message.

## Retrieval stages

Each stage ships only when `eval/bench.py` says it beat the previous one.

1. **BM25 baseline** — `search/lexical.py`. FTS5 with porter stemming. Already the
   thing to beat; it will lose on paraphrase and win on names, numbers, and rare
   strings. *(shipped)*
2. **Dense** — EmbeddingGemma-300M or Qwen3-Embedding-0.6B (GGUF, local) over
   conversation windows, ANN via sqlite-vec.
3. **Hybrid via RRF** — fuse the two ranked lists with reciprocal rank fusion,
   `k=60`. RRF is used rather than score blending because BM25 and cosine scores
   are not on a comparable scale; fusing *ranks* sidesteps calibration entirely.
4. **Rerank** — cross-encoder (Qwen3-Reranker-0.6B) over the top ~30. qmd blends
   reranker and retrieval scores by position — trusting retrieval more at the top
   (75/25 for ranks 1–3) and the reranker more further down (40/60 past rank 11) —
   which preserves exact matches that BM25 got right. Adopt that, then verify.
5. **Query expansion** — only if stages 1–4 plateau. Highest cost, least certain
   payoff.

## Why not just embed everything with an API

Three reasons, in order: the corpus is private; 260k embeddings of personal messages
sent to a third party is not recoverable if it goes wrong; and local GGUF inference
on Apple silicon is fast enough that the tradeoff buys nothing.

## Dependencies

Stdlib-only today. Each addition gets recorded here with its justification.

| dep | stage | why |
|---|---|---|
| `sqlite-vec` | 2 | ANN inside SQLite; avoids running a vector DB |
| `llama-cpp-python` | 2 | local GGUF embedding + rerank on Metal |

## Evaluation

`eval/gold.jsonl` is the contract. Queries are labeled by pooling retriever
candidates (`eval/label.py`) and judging the pool — TREC-style, because judging all
260k messages per query is impossible.

Gold labels are message **ids only**; no text is committed.

Target: ~30–50 labeled queries before any stage-2 number is trustworthy. Below ~20,
the metrics are noise. The seed queries in `gold.jsonl` are unlabeled placeholders
and score nothing on purpose.
