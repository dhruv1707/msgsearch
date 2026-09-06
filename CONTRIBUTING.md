# Contributing

Thanks for looking at this. The project is small and the rules are few, but two
of them are unusual and matter more than the rest, so they come first.

## The two rules that are not negotiable

**Never point anything at `~/Library/Messages`.** That database is live, Messages
holds locks on it, and it is irreplaceable. Work against a copy, opened read-only
via a `file:...?mode=ro` URI. Every code path in this repo already does; keep it
that way.

**No real message data in the repository. Ever.** Not in tests, not in fixtures,
not in issues, not in a commit you plan to amend later. Every fixture in `tests/`
is invented, and `eval/gold.jsonl` stores window ids only — never text. If you are
reporting a bug that depends on message content, describe the *shape* of the
message rather than pasting it.

`.gitignore` blocks databases, indexes and `config_local.py`, but it is a
backstop, not a guarantee.

## Getting set up

You need an **arm64** Python 3.10–3.13 on macOS. PyTorch stopped publishing
x86_64 macOS wheels, and a Rosetta process cannot use the GPU regardless. Check:

```bash
python3 -c "import platform; print(platform.machine())"   # must print arm64
```

Then:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

Copy a database to work against (needs Full Disk Access for your terminal):

```bash
mkdir -p ~/msgsearch && cp ~/Library/Messages/chat.db* ~/msgsearch/
```

## Before you open a pull request

```bash
./.venv/bin/ruff check .
./.venv/bin/ruff format .
./.venv/bin/python -m unittest discover -s tests
```

CI runs exactly these, plus an import check, on Python 3.10 through 3.13. It
deliberately does *not* install PyTorch, so please keep model imports lazy —
inside functions rather than at module scope — so the pure-Python modules stay
importable without it.

## If you change retrieval, measure it

This is the one process rule with teeth, and it exists because it has already
caught two wrong decisions in this codebase.

```bash
./.venv/bin/python eval/bench.py -r eval.retriever --against baseline
```

Report the before/after table in your pull request. A retrieval change with no
movement in `eval/bench.py` is an unverified guess, however plausible the
reasoning. Adding a reranker *sounded* obviously correct and measurably made
results worse; tuning passage size against a single query produced a number that
looked authoritative and meant nothing.

If your change is an improvement that the current gold set cannot detect, say so
explicitly rather than claiming a win, and consider adding labelled queries that
would show it:

```bash
./.venv/bin/python eval/label.py "a query you actually wanted answered"
```

Note that gold queries should come from real recall failures rather than being
invented after browsing the index. A query chosen because you already know it is
findable will flatter whatever produced it.

## Style

The linter settles formatting; don't argue with it in review. Beyond that:

- **Explain why, not what.** The code says what it does. Comments should carry
  the reason a non-obvious choice was made, ideally with the number that
  justified it. `chunk.py` and `config.py` are the model here.
- **Name the failure mode.** When something guards against a subtle bug, say
  which bug. "Vectors from different models are not comparable" is worth more
  than "check the model".
- **Prefer measurements to adjectives** in both comments and commit messages.

## Reporting bugs

Include your macOS version, `python3 -c "import platform; print(platform.machine())"`,
and the output of `msgsearch explore`, which prints structural statistics and no
message content. Redact anything you are unsure about.
