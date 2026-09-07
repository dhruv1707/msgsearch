# msgsearch

Search your iMessage history by meaning, entirely on your own machine.

Apple's Messages search matches literal words. That fails whenever you remember
*what happened* but not *what was said* — someone sent you a login months ago and
the message containing it never uses the word "login", so no amount of typing
finds it.

```console
$ msgsearch search "the wifi password at the airbnb"

1. 2025-06-14 19:02-19:20  Sam Rivera  (Me, Sam Rivera)  [credential]
   bm25 #2 · vector #1
  Sam Rivera: just got here, place is nice
  Me: what's the wifi
  Sam Rivera: network is Coastal_5G, password Harbour2019Blue
```

Nothing leaves your machine: both the embedding and reranking models run locally.

> [!WARNING]
> The index it builds is a **plaintext, searchable copy of every private thing
> anyone has ever texted you** — passwords, addresses, medical details. It lives
> in `~/msgsearch/index/`, unencrypted. Treat that directory the way you would
> treat a password manager's database.

## Requirements

- **macOS on Apple silicon.** PyTorch no longer publishes x86_64 macOS wheels.
- **Python 3.10 or newer, running as arm64.** This is the one thing that
  reliably goes wrong — see [Install](#install) if it does.
- **About 3 GB of disk** for dependencies and model weights.

## Install

```bash
pipx install msgsearch
msgsearch doctor          # verifies your machine, names any problem and its fix
```

The install pulls PyTorch, around 1 GB, so it is slow once and fast thereafter.

<details>
<summary><b>If it fails with "No matching distribution found for torch"</b></summary>

Your Python is an Intel build, which is common on Apple silicon after migrating
from an Intel Mac. Check:

```bash
python3 -c "import platform; print(platform.machine())"   # must say arm64
```

The confusing part is that `pipx install --python /path/to/arm64/python` often
**does not fix it**. Python from python.org is a *universal* binary that runs as
whichever architecture its parent process is, so a pipx installed by an Intel
Homebrew launches it as x86_64 whichever interpreter you name. pip then hunts for
x86_64 wheels PyTorch does not publish.

Create the environment explicitly under `arch -arm64` instead:

```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    -m venv ~/.msgsearch-venv
~/.msgsearch-venv/bin/pip install msgsearch
ln -sf ~/.msgsearch-venv/bin/msgsearch /usr/local/bin/msgsearch
```

Substitute any arm64 Python 3.10+. To undo:
`rm /usr/local/bin/msgsearch && rm -rf ~/.msgsearch-venv`.

Architecture cannot be expressed in package metadata, which is why this appears
as a wall of dependency errors rather than a useful message.
</details>

## Setup

Three one-time steps. `msgsearch doctor` will tell you which of them you still
need at any point.

### 1. Get access to the embedding model

`google/embeddinggemma-300m` is gated — Google requires you to accept its licence.

1. Create a free account at <https://huggingface.co/join>
2. Open <https://huggingface.co/google/embeddinggemma-300m>, sign in, and accept
   the Gemma terms at the top of the page. Approval is normally immediate.
3. Create a token at <https://huggingface.co/settings/tokens> with the **Read**
   role, and copy it
4. Run `msgsearch login` and paste it when prompted

The model itself (1.2 GB) downloads automatically the first time you index.
There is no separate download step.

Being logged in and having accepted the licence are different things, and
HuggingFace reports both failures as the same "please log in" error — so
`msgsearch login` checks afterwards that the model is genuinely reachable.

Prefer not to make an account? Use an ungated model. Retrieval is somewhat
weaker; nothing else changes:

```bash
export MSGSEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5
```

### 2. Grant Full Disk Access

Reading your messages needs it. macOS grants this to the **application**, not to
your shell, so it goes to whatever program you type commands into.

1. Open **System Settings → Privacy & Security → Full Disk Access**
2. Click **+**
3. Press **⌘⇧G**, paste `/Applications/Utilities/Terminal.app`, then Open
   (choose iTerm, VS Code or whatever you actually use, if not Terminal)
4. Make sure its toggle is **on**
5. **Quit and reopen that application** — the permission is only read at launch

### 3. Snapshot your messages and build the index

```bash
msgsearch sync            # copy the live database  (safe: never writes to it)
msgsearch index           # ~30 minutes per 100k messages, once
```

`sync` uses SQLite's backup API rather than `cp`. That matters: Messages runs in
WAL mode, so your most recent messages sit in a `chat.db-wal` sidecar until they
are checkpointed, and `cp chat.db` drops exactly the messages you are most likely
to search for, silently.

## Use

```bash
msgsearch search "that restaurant we talked about"
```

Useful flags:

```
--limit N            how many results (default 10)
--chat TEXT          restrict to conversations matching TEXT
--from WHO           restrict to a speaker
--after / --before   YYYY-MM-DD
--type credential    only windows that appear to CONTAIN a credential
                     (also: credential_talk, email, phone, url, address)
--full               show the whole conversation, not just the matching part
--no-dense           keyword search only
--no-bm25            vector search only
--rerank             run the cross-encoder (off by default; it measurably hurts)
```

`--type credential` is the one worth remembering. Searching for a forgotten
password without it returns mostly people *discussing* passwords; with it, the
message containing one tends to come first.

### Keeping it current

Your snapshot is frozen at the moment you took it, so new messages need both a
refresh and a re-index. That is one command, and it takes seconds rather than the
original half hour, because only genuinely new text is embedded:

```bash
msgsearch sync --index
```

### Names (optional)

Without this, speakers appear as phone numbers. Resolving them makes results
readable *and* improves retrieval, because the speaker label is part of the text
that gets embedded — `Sam: ...` carries meaning where `+15551234567: ...` does not.

Grant **Contacts** access the same way you granted Full Disk Access (it is a
separate permission; one does not imply the other) and names are picked up
automatically. Otherwise, or to correct a name:

```bash
msgsearch contacts                 # what is resolved, and from where
msgsearch contacts --template 20   # a stub for the 20 busiest handles
$EDITOR ~/msgsearch/contacts.json  # fill in names; blanks are ignored
```

A handful goes a long way: on a typical archive the ten busiest handles account
for over 90% of received messages. Do this *before* indexing — changing a name
changes the embedded text, so it costs a full rebuild afterwards.

## All commands

```
msgsearch doctor      check this machine is set up correctly, and name any fix
msgsearch login       authenticate with HuggingFace for the embedding model
msgsearch sync        refresh the working copy of the messages database
msgsearch index       build the search index
msgsearch search      search it
msgsearch contacts    map phone numbers and emails to names
msgsearch explore     structural report on a database (prints no message content)
```

## Where this is going

Today msgsearch reads iMessage. The pipeline is nearly source-agnostic already —
only the extraction step knows what iMessage is — so the plan is **one search
across every conversation you have**: Telegram and Slack next, Discord for
servers a bot can join. WhatsApp was investigated and is genuinely hard; it is
not promised.

The reason for unifying them is the second half: exposing msgsearch over the
[Model Context Protocol](https://modelcontextprotocol.io), so any MCP-capable
coding agent can search your conversations as a tool. An agent working in your
repository has no access to the thread where a design was argued out, or the DM
where a client stated what they actually wanted. None of that is in the code.

That cuts against this project's first promise, and deliberately so: an agent
running on someone else's servers would receive message content. So the server
will be opt-in, will warn about what it exposes, and will support a
metadata-only mode that returns *where* something was discussed without the text.
See [ARCHITECTURE.md](ARCHITECTURE.md#roadmap-many-sources-one-search-exposed-to-agents).

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

**Search combines two retrievers.** BM25 finds literal words; vector search finds
meaning; Reciprocal Rank Fusion merges the two ranked lists, whose scores sit on
incomparable scales, by using positions rather than scores. A cross-encoder
reranking stage exists but is disabled, because measuring it showed it made
results worse.

## Does it work?

`eval/` holds a small set of labelled queries and the metrics for them, so claims
here can be checked rather than believed. Current numbers, on a 9-query gold set
over a single 105k-message conversation:

| retriever | MRR | nDCG@10 | recall@50 |
|---|---|---|---|
| **hybrid (default)** | **0.843** | **0.811** | 1.000 |
| keyword only | 0.667 | 0.706 | 1.000 |
| vectors only | 0.722 | 0.618 | 0.891 |
| hybrid + reranking | 0.700 | 0.677 | 1.000 |

Two things that table settles. Fusing the two retrievers genuinely beats either
alone, which is the central design bet. And `recall@50` of 1.000 means the right
answer is always in the shortlist, so what remains is a ranking problem rather
than a finding problem.

Nine queries is a small set, and it is stated here so you can weigh the numbers
accordingly. See `CONTRIBUTING.md` if you want to change retrieval — measuring
first is the one process rule this project insists on.

## Layout

```
src/msgsearch/
  cli.py             the msgsearch command; all argument parsing lives here
  config.py          every tunable, with the measurement that justified it
  contacts.py        handle -> name resolution (alias file, vCard, AddressBook)
  doctor.py          setup preflight checks, each with the command that fixes it
  explore.py         inspect a chat.db and report what is in it
  attributed_body.py decode the attributedBody blob
  extract.py         database rows -> clean message records
  tagging.py         shape tags (credential, email, phone, url, address)
  chunk.py           messages -> conversation windows -> passages
  embedder.py        local embedding and reranking models
  index.py           build the index
  search.py          query it
tests/               synthetic fixtures only, never real message data
eval/                the verification loop: gold queries, metrics, labelling tool
```

The modules are importable as a library if you want the pieces without the CLI:

```python
from msgsearch.extract import connect, iter_messages
from msgsearch.chunk import windows

with connect() as db:
    for window in windows(iter_messages(db)):
        ...
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Two rules matter more than the rest: never
point anything at `~/Library/Messages`, and never commit real message data — not
in tests, not in fixtures, not in issues.

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/ruff check . && ./.venv/bin/ruff format --check .
./.venv/bin/python -m unittest discover -s tests
```

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Licence

MIT.
