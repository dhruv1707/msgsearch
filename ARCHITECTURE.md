# Architecture

Local-first semantic search over ~260k iMessages. No message text ever leaves the
machine — not to an embedding API, not to a reranker, not in a commit.

The pipeline is the conventional one for hybrid retrieval — keyword search and
vector search fused, with an optional reranking stage — adapted from documents to
chat messages, which behave differently enough to change several decisions.
Where measurement has contradicted an assumption, the measurement wins and the
number is recorded here.

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

Keeping vectors inside SQLite via `sqlite-vec` is the conventional choice and a
reasonable one at larger scale. It is not yet worth it here. With ~95k passages a search is a
single `numpy` matrix multiply over a 95k×768 array, which completes in single-digit
milliseconds — faster than an ANN index once its own overhead is counted, and it
avoids a dependency. Revisit if the full-corpus index (398 chats with indexable messages) makes the
array unwieldy, which is the point where ANN starts to earn its keep.

| table | role |
|---|---|
| `windows` | conversation windows: text, speakers, tags, per-message offsets |
| `passages` | 3-message slices, each pointing at its parent window |
| `windows_fts` | FTS5 external-content index over window text (porter unicode61) |
| `meta` | model, dimensions, chunking parameters, build time |

## Chunking: messages are not documents, and windows are not passages

This is where chat departs most sharply from documents. Document retrieval chunks
prose into windows of several hundred tokens with some overlap. Messages are the
opposite problem — the median message is under 20 tokens, far too short to embed
meaningfully in isolation ("yeah", "ok sounds good", "that one").

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
   candidate's best *passage* rather than its whole window. **Implemented and
   disabled by default**, because measuring it showed it makes results worse:
   MRR falls from 0.843 to 0.700 and nDCG@10 from 0.811 to 0.677, with the damage
   concentrated on topical queries (nDCG 0.790 → 0.521). It takes rankings fusion
   already got right and scrambles them. These rerankers are trained on clean QA
   passages, and a window of "Yaaa bro I do" is far outside that distribution.
   - Blending reranker and retrieval scores by position — trusting retrieval more
     at the top few ranks and the reranker more further down — is a known
     technique for preventing exactly the failure measured above. **Untested
     here.** It is the obvious thing to try before concluding the stage is
     worthless; the current implementation lets the reranker override outright,
     which is the version that lost.
5. **Query expansion** — only if stages 1–4 plateau. Highest cost, least certain
   payoff.

## Models

Both run locally on Metal via MPS.

| role | model | notes |
|---|---|---|
| embedding | `google/embeddinggemma-300m` | 768-dim. **Gated** — needs Gemma licence acceptance and `msgsearch login`. Matryoshka truncation to 512/256/128 available if the array grows. |
| rerank | `Qwen/Qwen3-Reranker-0.6B` | implemented, disabled by default |

Rerankers measured over 50 candidates on the one query available:

| model | load | score 50 | on disk | rank produced |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 1.8s | 0.15s | 88 MB | 2 |
| `BAAI/bge-reranker-v2-m3` | 5.0s | 1.12s | 2.1 GB | 2 |
| `Qwen/Qwen3-Reranker-0.6B` | 2.6s | 4.21s | 1.1 GB | 2 |

Qwen3-Reranker is a causal LM scoring relevance from `yes`/`no` logits rather than
a classification head, which is why it is ~28× slower than MiniLM. It loads
through `CrossEncoder` and applies its own `query` prompt automatically; sanity
checks on obvious relevant/irrelevant pairs order correctly with a wide margin.

On a single hand-checked query all three produced identical rankings, which
looked like evidence the stage was merely redundant. The 9-query gold set showed
it is actively harmful — see stage 4 above. This is the clearest case in the
project of one query being worse than no measurement at all, because it produced
false confidence rather than acknowledged ignorance.

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

Declared in `pyproject.toml`, pinned rather than ranged: a different embedding model version silently changes every vector in the index, and the failure mode is bad results rather than an error.

**Requires an arm64 Python.** PyTorch no longer publishes x86_64 macOS wheels, and
a Rosetta process cannot use the GPU regardless. The system `python3` at
`/usr/local` on this machine is an Intel build and cannot be used.

## Roadmap: many sources, one search, exposed to agents

None of this is built. It is written down because it shapes decisions being made
now, and because the architecture already accommodates most of it.

### Why the pipeline is nearly source-agnostic already

Only `extract.py` knows what iMessage is. Windowing, passages, tagging, indexing
and search all operate on the `Message` dataclass and would not change. A
connector's entire job is producing those fields:

    rowid  guid  chat_id  chat_label  chat_is_group  timestamp
    is_from_me  speaker  service  text  reply_to  attachments

Two things need adjusting before a second source can exist:

- **`window_id` must be namespaced.** It is `chat_id:first_rowid` today, which
  would collide across sources. `slack:C01234:5678` or similar.
- **The index needs a `source` column**, so results can be filtered and attributed.

Neither is large. The `service` field already carries `iMessage` / `SMS` and
would extend naturally.

### Connector feasibility, honestly

| source | how | difficulty |
|---|---|---|
| **Telegram** | MTProto user client, or the Desktop JSON export | easy — a real API for your own history |
| **Slack** | Web API (`conversations.history`) via OAuth, or a workspace export | easy — but free workspaces only retain a window, and corporate ones may need admin approval |
| **Discord** | a bot reading channels it has joined | feasible for servers; **not** personal DMs, since automating a user account violates the ToS |
| **GitHub** | REST API for issue and PR comments | easy — token-scoped, and the discussion around a change is often the context an agent most needs |
| **WhatsApp** | per-chat "Export chat", manually | **hard, and not promised.** End-to-end encryption means no API for personal history. `msgstore.db` needs root, iOS backups are encrypted, and libraries automating WhatsApp Web violate the ToS and get numbers banned. The manual export drops metadata and is capped. |

WhatsApp is listed to record that it was investigated, not to imply it is coming.

### An MCP server

The point of unifying sources is not tidiness, it is that **an agent working in
your repository has no access to the conversations where the decisions were
made**. The reasoning behind a design lives in a Slack thread; the client's
actual requirement lives in a DM. Neither is in the code, the commits or the
tickets.

Exposing msgsearch over the [Model Context Protocol](https://modelcontextprotocol.io)
would let any MCP-capable agent query that context directly. Retrieval is already
a clean function, so the server is a thin layer over it. Sketch:

    search_messages(query, limit=10, source=?, from=?, after=?, before=?, type=?)
        -> ranked windows: id, source, participants, timestamps, matched text

    get_conversation(window_id, context=0)
        -> the full window, optionally with neighbouring ones

    list_sources()
        -> which connectors are indexed, and how current each is

The Python SDK is mature (`mcp`, `fastmcp`), and both require Python 3.10+,
matching this project.

### Permissions: index only what the user can already read

Every source after iMessage brings access control with it. A Slack workspace has
private channels, GitHub has private repositories, Discord has servers you were
never in. Indexing content the user cannot see would turn a search tool into a
privilege-escalation device, and doing it accidentally is easy.

The rule is that **msgsearch never sees more than the person running it**. That
falls out almost for free if connectors authenticate *as the user* rather than as
a bot or an admin: the source enforces its own ACLs at the API boundary, and the
connector inherits them without having to model them.

| source | scoped by | what it can reach |
|---|---|---|
| iMessage | the local account | conversations on this Mac |
| Slack | user OAuth token | channels the user is in, plus their DMs — not private channels they are not a member of |
| GitHub | a `repo`-scoped PAT or OAuth app | issue and PR comments in repositories the user can read |
| Telegram | the user's own session | their own chats |
| Discord | a bot token | only channels the bot has been invited to |

Discord is the odd one: a bot's reach is not the user's reach, so a Discord
connector would index what the *bot* can see. That difference has to be shown in
the interface rather than hidden.

**The hard part is not granting access, it is losing it.** An index is a cache of
permission decisions made in the past. Leave a Slack channel, lose access to a
repository, get removed from a workspace — the content stays in the local index
and would be served happily to whatever asks. Nothing in the current design would
notice.

Three things follow, none of them built:

- Every window records the **source scope it came from** (workspace, channel id,
  repository, visibility), so it can be re-checked rather than merely trusted.
- A **revalidation pass** on refresh confirms the user still has access to each
  scope, and prunes windows whose access has gone. Re-indexing already re-reads
  everything, so this is the natural place for it.
- Scopes carry a **last-confirmed timestamp**, so stale entries can be excluded
  rather than silently trusted forever.

**Multi-user is a different project.** Everything above assumes one person on
their own machine, where "what the user can read" is fixed at ingest.
[Lobu](https://github.com/lobu-ai/lobu) solves the harder version — many callers
against one index, filtering every read to the resources that particular caller
belongs to, with a gateway holding OAuth credentials so workers never see tokens.
That is the correct architecture for a team deployment and a substantially larger
undertaking than this. If msgsearch ever grows past one machine, the read-filter
belongs at query time and per-caller, not at ingest.

### The tension this creates, stated plainly

This project's first promise is that nothing leaves your machine. An MCP server
consumed by a **cloud-hosted** agent breaks that promise: message content —
including the credentials this tool is unusually good at surfacing — would be
sent to a model provider.

That is a legitimate trade, but it must be a deliberate one, so the server would:

- be **opt-in**, never running by default;
- **warn loudly** on startup about what it exposes;
- support a **metadata-only mode**, returning ids, participants and timestamps
  but not message text, so an agent can find *where* something was discussed and
  hand the reader a pointer rather than the content;
- honour the existing `--type` filters, so credential-bearing windows can be
  excluded from what an agent can ever see.

An agent running against a local model has no such exposure. The design should
make the difference legible rather than assume either case.

## Evaluation

`eval/gold.jsonl` is the contract. Queries are labeled by pooling retriever
candidates (`eval/label.py`) and judging the pool — TREC-style, because judging all
260k messages per query is impossible.

Gold labels are message **ids only**; no text is committed.

Target: ~30–50 labeled queries before any retrieval number is trustworthy. Below
~20, the metrics are noise.

**Status: 9 labelled queries** — 1 exact, 5 semantic, 3 topical, and none yet of
the `cross-domain` or `alias` kinds. Measured on the full corpus (398 chats,
260,347 messages, 27,551 windows, 220,225 passages):

| retriever | r@50 | MRR | nDCG@10 |
|---|---|---|---|
| **hybrid (default)** | 0.972 | **0.673** | **0.725** |
| keyword only | 1.000 | 0.501 | 0.539 |
| vectors only | 0.903 | 0.537 | 0.519 |

Fusion still beats both of its inputs, which is the result that matters.

### Why these numbers are lower than the earlier ones

An earlier snapshot showed MRR 0.843. That was measured on an index containing a
single conversation and using phone numbers as speaker labels. Moving to the full
corpus with resolved names costs 0.169, decomposed by a controlled rebuild:

| condition | MRR |
|---|---|
| one chat, phone numbers (the old baseline) | 0.843 |
| one chat, **with names** | 0.782 |
| full corpus, with names | 0.673 |

About a third of the loss is naming and two thirds is corpus growth.

**The corpus two-thirds is not a quality regression.** `gold.jsonl` was labelled by
pooling candidates from the single-conversation index, so relevant windows in the
other 397 conversations were never surfaced, never judged, and are scored as
irrelevant. The metric understates quality on the full corpus by an unknown
amount. Re-pool and re-judge against the current index before drawing conclusions
from the absolute numbers; the comparisons between retrievers remain valid,
because they all run against the same labels.

**The naming third is a real cost, accepted deliberately.** Resolved names make
results readable and make `--from` work, and the speaker label is part of the
embedded text, so changing it changes every vector. Worth testing later: keep
names for display and keyword search but embed a neutral role label instead, which
might recover the 0.061 while keeping the benefits. Untested — it needs a full
rebuild to find out.

Still resting on one query, and flagged where they appear: the passage size of 3,
the stride of 1, the 30-minute window gap, `RRF_K = 60`, and the claim that
EmbeddingGemma beats bge-small. The passage-vs-window decision in particular needs
re-testing once gold labels
exist.
