# msgsearch

Search your iMessage history by meaning, entirely on your own machine.

Apple's Messages search matches literal words. That fails whenever you remember
*what happened* but not *what was said* — someone sent you a login months ago and
the message containing it never uses the word "login", so no amount of typing
finds it. This tool retrieves by meaning as well as by keyword, and it never sends
your messages anywhere.

## ⚠️ Read this before you build an index

The index is a **plaintext, searchable copy of every private thing anyone has ever
texted you**, including passwords, addresses and medical details. It is written to
`~/msgsearch/index/` by default, outside this repository, and `.gitignore` is set
up to make committing it difficult. Treat that directory the way you would treat a
password manager's database. Both models run locally, so nothing is uploaded, but
what lands on your disk is unencrypted.

## Requirements

- macOS with an **arm64** Python 3.12 or 3.13. PyTorch stopped publishing x86_64
  macOS wheels, so an Intel-built interpreter (common at `/usr/local` even on
  Apple Silicon) cannot install torch and cannot use the GPU. Check with
  `python3 -c "import platform; print(platform.machine())"` — it must say `arm64`.
- About 3 GB of disk for dependencies and model weights.

## Setup

```bash
python3 -m venv .venv                  # must be an arm64 interpreter
./.venv/bin/python -m pip install -r requirements.txt
```

Copy the Messages database. **Never point this tool at `~/Library/Messages`**:
that file is live, Messages.app holds locks on it, and it is irreplaceable.
Copying requires Full Disk Access for your terminal (System Settings → Privacy &
Security → Full Disk Access).

```bash
mkdir -p ~/msgsearch
cp ~/Library/Messages/chat.db* ~/msgsearch/
```

The default embedding model, `google/embeddinggemma-300m`, is gated. You must
accept the Gemma licence at
<https://huggingface.co/google/embeddinggemma-300m> and authenticate:

```bash
./.venv/bin/hf auth login
```

Any sentence-transformers model works instead if you would rather not, for example
`MSGSEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5`.

## Use

```bash
./.venv/bin/python explore.py                       # what is in your database
./.venv/bin/python index.py                         # build the index
./.venv/bin/python search.py "atria login"          # search it
```

Restrict the index to one conversation while trying things out:

```bash
MSGSEARCH_TESTBED_CHAT='+15551234567' ./.venv/bin/python index.py
```

Useful search flags:

```
--limit N        how many results
--chat TEXT      restrict to conversations matching TEXT
--from NAME      restrict to a speaker
--after / --before YYYY-MM-DD
--type credential    only windows that appear to CONTAIN a credential
                     (credential_talk is the separate tag for windows that only
                     discuss one; also: email, phone, url, address)
--full           show the whole conversation window, not just the match
--no-rerank      skip the cross-encoder, to see what it contributes
--no-dense       keyword search only
--no-bm25        vector search only
```

## How it works, and why

**86% of your messages have no text.** Apple sets `message.text` to NULL on most
rows and stores the content in an `attributedBody` blob instead, as an archived
`NSAttributedString` in the old typedstream format. Indexing the `text` column
alone would cover about one message in seven, and the gap is worst on old messages
and on messages you sent. `attributed_body.py` decodes the blob; it is verified to
reproduce Apple's own `text` column exactly on all rows where both are present.

**Timestamps are nanoseconds since 2001-01-01**, not seconds since 1970.

**`handle_id` is not the sender.** It identifies the other party in the
conversation, in both directions. Direction comes from `is_from_me` alone.

**Retrieval separates the unit that is matched from the unit that is shown.**
Messages are grouped into *windows*: runs of conversation with no pause longer
than thirty minutes. Windows are what you get shown, because a result without
context is unreadable. But windows are the wrong thing to embed — compressing
thirty messages on eight topics into one vector represents none of them. So
embedding runs over *passages*: three-message slices that slide across each
window. Measured on a real thread, the same target ranked 3539th when whole
windows were embedded, 13th when three-message passages were, and 5117th when
single messages were. Both too much context and too little are fatal.

**Search combines four stages.** BM25 finds literal words; vector search finds
meaning; Reciprocal Rank Fusion merges the two ranked lists, whose scores are on
incomparable scales, into a shortlist; and a cross-encoder reranks that shortlist
by reading the query and each passage together, which is much more accurate and
much too slow to run over everything.

## Layout

```
explore.py           inspect a chat.db and report what is in it
attributed_body.py   decode the attributedBody blob
extract.py           database rows -> clean message records
tagging.py           shape tags (credential, email, phone, url, address)
chunk.py             messages -> conversation windows -> passages
embedder.py          local embedding and reranking models
index.py             build the index
search.py            query it
tests/               synthetic fixtures only, never real message data
```

## Contributing

Tests use invented data exclusively. Never commit a fixture derived from a real
conversation, and never commit an index, a database, or a `config_local.py`.

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Licence

MIT.
