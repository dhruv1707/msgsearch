# query_messages

Semantic search over ~260k personal iMessages (2018-08 → 2026-09), by meaning
rather than keyword.

Source of truth for all coding agents; `CLAUDE.md` symlinks here.

## Rules

Kept deliberately short. A rule earns a line here only if breaking it fails
*silently* or *irreversibly* — anything that fails loudly teaches itself, and
anything already enforced in code or `settings.json` is not repeated here.

- **Never** touch `~/Library/Messages`. Work against `~/msgsearch/chat.db`,
  opened `file:...?mode=ro`. (Irreversible.)
- Message content is private: no bodies printed to stdout in committed scripts,
  none sent over the network, none in commits or eval fixtures. (Irreversible.)
- `message.date` is Apple-epoch and **mixes scales** — pre-10.13 rows are
  seconds, later ones nanoseconds. Use `explore.apple_ts()`. (Silent: the wrong
  scale yields plausible, wrong dates.)
- ~86% of rows have `text IS NULL` with content in the `attributedBody`
  typedstream blob. Go through `attributed_body.message_text(text, blob)`.
  (Silent: reading `message.text` looks fine and loses most of the corpus.)

## Commands

```
python3 search/index.py             # rebuild index.sqlite (~3s)
python3 eval/bench.py --against baseline
python3 eval/label.py "a query"     # add a labeled gold query
python3 explore.py                  # structural recon
```

## The verification loop

A retrieval change is justified by `eval/bench.py` or it is a guess. Report the
before/after table; don't claim an improvement without it.

Architecture: `ARCHITECTURE.md`.

## Learned corrections

Append here **only** when an agent actually gets something wrong in practice.
Nothing speculative. If a rule can instead be enforced in code, a hook, or
`settings.json`, do that and leave this file alone.

_(empty — nothing has gone wrong yet)_
