# query_messages

Semantic search over ~271k personal iMessages (2018-08 → 2026-09). The goal is
recall by *meaning*, not keyword — macOS Messages search only does exact matching,
so "that password someone sent me" is unfindable today.

This file is the single source of truth for all coding agents. `CLAUDE.md` imports
it; do not duplicate rules across the two.

## Hard constraints

- **Never** read or write `~/Library/Messages`. All work goes against the read-only
  copy at `~/msgsearch/chat.db`, opened as `file:...?mode=ro`.
- ~86% of `message` rows have `text IS NULL`; the real content is in the
  `attributedBody` typedstream blob. Always go through
  `attributed_body.message_text(text, blob)` — never read `message.text` directly.
- `message.date` is Apple-epoch (2001-01-01). Rows are *mixed*: pre-10.13 rows are
  seconds, later ones nanoseconds. Use `explore.apple_ts()`; never assume one scale.
- Message content is private. Do not print message bodies to stdout in committed
  scripts, do not send them to any network service, and do not paste them into
  commits, PRs, or issues. Eval fixtures use redacted excerpts only.

## Commands

```
python3 explore.py                  # structural recon, no bodies printed
python3 eval/bench.py               # THE verification loop — run before every commit
python3 eval/bench.py --baseline    # compare against the naive LIKE retriever
```

## The verification loop (most important rule)

Every retrieval change must be justified by `eval/bench.py`. A change that does not
move precision@10 / MRR / recall@50 on `eval/gold.jsonl` is not an improvement — it
is an unverified guess. Do not report a retrieval change as done without pasting the
before/after metrics table.

If you cannot measure it, add a gold query to `eval/gold.jsonl` first, then change
the code.

## Architecture

See `ARCHITECTURE.md`. Short version: SQLite is the whole system — FTS5 for lexical,
sqlite-vec for dense, fused with RRF (k=60), then a local cross-encoder rerank.
Local-first: no message text leaves the machine, ever.

## Conventions

- Python 3.14, stdlib-first. Add a dependency only when it earns its place; record
  why in `ARCHITECTURE.md`.
- Every module opens with a docstring saying what it does and any non-obvious
  Apple/SQLite quirk it works around (see `attributed_body.py` for the bar).
- Prefer plain functions over classes until state actually accumulates.
- Comment the *why* (the Apple quirk, the ranking tradeoff), never the *what*.

## Learned corrections

Append a line here every time an agent gets something wrong. This file is the
compounding asset of the project — mistakes get written down, not re-explained.

- Do not use `datetime.utcfromtimestamp` (deprecated in 3.12+); use
  timezone-aware `datetime(..., tzinfo=timezone.utc)` as `explore.py` does.
- Tapbacks/reactions (`associated_message_type != 0`) are not messages; exclude
  them from the index or they flood results with "Liked "…"".
