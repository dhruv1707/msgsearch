# Architecture

Local-first semantic search over ~260k iMessages. No message text ever leaves the
machine — not to an embedding API, not to a reranker, not in a commit.

The design follows the pipeline Tobi Lütke's [qmd](https://github.com/tobi/qmd)
converged on after ~3 months of eval-driven iteration, adapted from markdown
documents to chat messages. Where qmd's choices are load-bearing, they are cited.
Where measurement has since contradicted an assumption, the measurement wins and
the number is recorded here.

## Pipeline

```
query
  ├─ expansion (optional, stage 5)  ──┐
  │                                   │
  ├─ BM25 / FTS5 over windows    ──┐  │
  ├─ dense over passages         ──┤  │
  │                                ▼  │
  │              RRF fusion (k=60) ◄──┘
  │                       │
  │              top ~50 candidates
  │                       ▼
  └────────────► cross-encoder rerank
                          │
                    ranked results
```

## Storage

Two files in `~/msgsearch/index/`, outside the repo:

| file | role |
|---|---|
| `index.db` | windows, passages, and the FTS5 lexical index |
| `vectors.npy` | one unit-length row per passage, aligned to `passages.vector_row` |

qmd keeps vectors inside SQLite via `sqlite-vec`, and that remains the right call
at document scale. It is not yet worth it here. With ~95k passages a search is a
single `numpy` matrix multiply over a 95k×384 array, which completes in single-digit
milliseconds — faster than an ANN index once its own overhead is counted, and it
avoids a dependency. Revisit if the full-corpus index (all 719 chats) makes the
array unwieldy, which is the point where ANN starts to earn its keep.

| table | role |
|---|---|
| `windows` | conversation windows: text, speakers, tags, per-message offsets |
| `passages` | 3-message slices, each pointing at its parent window |
| `windows_fts` | FTS5 external-content index over window text (porter unicode61) |
| `meta` | model, dimensions, chunking parameters, build time |

## Chunking: messages are not documents, and windows are not passages

This is the main departure from qmd. qmd chunks documents into ~900-token windows
with 15% overlap. iMessages are the opposite problem — the median message is under
20 tokens, far too short to embed meaningfully in isolation ("yeah", "ok sounds
good", "that one").

So messages are grouped into **conversation windows**: consecutive messages in one
chat with no pause longer than 30 minutes. Measured on a 105k-message thread this
yields 5,761 windows, median 5 messages and 68 tokens, but with a long tail — one
window ran to 558 messages. Windows over ~1500 tokens are split at a message
boundary with 2 messages of overlap, which touches about 1% of them. A fixed
message-count cap was rejected: the median window is 5 messages, so a 10-message
cap would have split 99% of the corpus for no benefit.

**The window is not the embedding unit.** An earlier version of this document
asserted it was. That was measured and found wrong. Compressing a 38-message window
onto one vector represents none of its topics well:

| unit embedded | cosine to query | rank of the known answer |
|---|---|---|
| full window (38 messages) | 0.504 | 3539 / 5761 |
| 9-message neighbourhood | 0.515 | 2745 |
| 5-message neighbourhood | 0.527 | 1882 |
| **3-message neighbourhood** | **0.602** | **13** |
| the single answer message | 0.475 | 5117 |

Both failure modes are real and they pull in opposite directions. Too little
context and the answer is unreachable — the message holding a credential shares no
words with "atria login" and ranks 5117th alone. Too much and it is diluted into
noise. So **passages** — 3-message slices sliding across each window, stride 1 —
are what gets embedded, and each one points back at its window.

The window remains the unit that is *retrieved and displayed*, because a result
without context is unreadable. Display anchors on the matched message and runs
forward, since the answer to a query usually arrives just after the message that
matches it.

*Caveat: the table above is one query. Per the evaluation section, that is not yet
sufficient evidence. It is acted on because the effect size is large and the
mechanism is understood, and it is flagged here for re-testing against `gold.jsonl`.*

## Retrieval stages

Each stage ships only when the eval says it beat the previous one.

1. **BM25 baseline** — FTS5 with porter stemming, over window text. *(shipped)*
2. **Dense** — over passages, brute-force cosine. *(shipped)*
3. **Hybrid via RRF** — fuse the two ranked lists, `k=60`. RRF is used rather than
   score blending because BM25 and cosine scores are not on a comparable scale;
   fusing *ranks* sidesteps calibration entirely. Dense results are collapsed to
   one best passage per window before fusion, so the shortlist holds distinct
   conversations. *(shipped)*
4. **Rerank** — cross-encoder over the top ~50, scoring the query against each
   candidate's best *passage* rather than its whole window, for the same dilution
   reason. *(shipped)*
   - qmd blends reranker and retrieval scores by position — trusting retrieval
     more at the top (75/25 for ranks 1–3) and the reranker more further down
     (40/60 past rank 11) — which preserves exact matches that BM25 got right.
     **Not yet adopted**; the reranker currently overrides outright. Worth testing
     once `gold.jsonl` exists: on the one query measured so far, fusion alone
     ranked the target 3rd and the reranker moved it to 2nd, so there is no
     evidence either way yet.
5. **Query expansion** — only if stages 1–4 plateau. Highest cost, least certain
   payoff.

## Models

Both run locally on Metal via MPS.

| role | model | notes |
|---|---|---|
| embedding | `google/embeddinggemma-300m` | 768-dim. **Gated** — needs Gemma licence acceptance and `hf auth login`. Matryoshka truncation to 512/256/128 available if the array grows. |
| rerank | `BAAI/bge-reranker-v2-m3` | cheap at ~50 candidates |

**EmbeddingGemma requires task-specific prompt prefixes.** Queries and documents
are presented to it differently, and that asymmetry is what teaches it to place a
question near its *answer* rather than near other questions — precisely what this
project needs. `embedder.py` therefore always uses `encode_query()` /
`encode_document()`, never bare `encode()`. This is not cosmetic: on the fallback
model, adding the correct query prefix moved a target from rank 4025 to 2948.

`llama-cpp-python` with GGUF weights was the original plan and remains a
reasonable alternative. `sentence-transformers` was chosen instead because it
implements the prompt-prefix handling above correctly for EmbeddingGemma, and
because torch/MPS was already required. Revisit if startup latency (~3s to load
the model per query) becomes annoying for CLI use.

## Why not just embed everything with an API

Three reasons, in order: the corpus is private; 260k embeddings of personal messages
sent to a third party is not recoverable if it goes wrong; and local inference on
Apple silicon is fast enough that the tradeoff buys nothing — 95k passages embed in
165s on the fallback model.

## Dependencies

Each addition is recorded here with its justification.

| dep | stage | why |
|---|---|---|
| `torch` | 2 | Metal/MPS inference backend |
| `sentence-transformers` | 2 | correct prompt-prefix handling for EmbeddingGemma; cross-encoder reranking |
| `numpy` | 2 | vector storage and the cosine matrix multiply |

Deliberately *not* taken:

| dep | why not |
|---|---|
| `sqlite-vec` | brute force is faster at 95k passages; revisit at full-corpus scale |
| `llama-cpp-python` | would duplicate torch, which is already required |

**Requires an arm64 Python.** PyTorch no longer publishes x86_64 macOS wheels, and
a Rosetta process cannot use the GPU regardless. The system `python3` at
`/usr/local` on this machine is an Intel build and cannot be used.

## Evaluation

`eval/gold.jsonl` is the contract. Queries are labeled by pooling retriever
candidates (`eval/label.py`) and judging the pool — TREC-style, because judging all
260k messages per query is impossible.

Gold labels are message **ids only**; no text is committed.

Target: ~30–50 labeled queries before any retrieval number is trustworthy. Below
~20, the metrics are noise.

**Status: not yet built.** Every retrieval claim in this document rests on a single
hand-checked query, which is below that bar and is marked as such where it appears.
The passage-vs-window decision in particular needs re-testing once gold labels
exist.
