# Changelog

Notable changes to msgsearch. Retrieval changes carry the measurement that
justified them; see `ARCHITECTURE.md` for the full evaluation.

## [0.1.0] — 2026-09-06

First release.

### Added
- `msgsearch` command with `doctor`, `explore`, `contacts`, `index` and `search`.
- Decoding of Apple's `attributedBody` typedstream, which holds the text of ~86%
  of messages. Verified to reproduce Apple's own `text` column exactly on all
  37,745 rows where both are present.
- Hybrid retrieval: BM25 over conversation windows, vector search over
  three-message passages, merged with Reciprocal Rank Fusion.
- Contact-name resolution from macOS Contacts, with an alias file that overrides
  it entry by entry and needs no permission.
- Incremental indexing. Embeddings are reused when the passage text is unchanged,
  so topping up an index takes seconds rather than an hour.
- `msgsearch doctor`, which reports what is wrong with a setup and the command
  that fixes it.
- An evaluation harness (`eval/`) with labelled queries and per-stage metrics.

### Measured
- Fusion beats either retriever alone: MRR 0.815 against 0.657 for keyword only
  and 0.537 for vectors only, over 9 labelled queries on a 260k-message archive.
- Reranking is implemented but **off by default**: it measurably hurt, dropping
  MRR from 0.843 to 0.700, worst on topical queries.
- Passages are 3 messages because embedding whole windows buried a target answer
  at rank 3539 and embedding single messages at 5117; three-message passages put
  it at 13.

### Known limitations
- macOS with Apple silicon only. PyTorch publishes no x86_64 macOS wheels.
- The default embedding model is gated and needs a Gemma licence acceptance.
- The gold set is 9 queries and covers none of the `cross-domain` category, so
  absolute metrics should be read as indicative rather than settled.
