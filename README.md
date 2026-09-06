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

## Quickstart

```bash
# 1. install  (needs an arm64 Python — see Install if this errors)
pipx install msgsearch

# 2. get the embedding model  (accept the licence in a browser first, then:)
msgsearch login

# 3. snapshot your messages            [needs Full Disk Access]
msgsearch sync

# 4. check the setup, and fix whatever it names
msgsearch doctor

# 5. build the index                   [~30 min per 100k messages, once]
msgsearch index

# 6. search
msgsearch search "that restaurant we talked about"
```

Afterwards, one command keeps it current — only genuinely new text is embedded,
so it takes seconds:

```bash
msgsearch sync --index
```

### Granting Full Disk Access

Step 3 fails without it. macOS grants this to the **application**, not to your
shell, so the grant goes to whatever program you type commands into.

1. Open **System Settings → Privacy & Security → Full Disk Access**
2. Click **+**
3. Press **⌘⇧G** and paste `/Applications/Utilities/Terminal.app`, then Open
   (if you use iTerm, VS Code or another terminal, choose that instead)
4. Make sure its toggle is **on**
5. **Quit and reopen that application** — the permission is only read at launch

Resolving contact names needs a *separate* permission, **Contacts**, granted the
same way in the same place. It is optional; without it speakers appear as phone
numbers. Full Disk Access does not include it.

## Requirements

- **macOS on Apple silicon.** PyTorch no longer publishes x86_64 macOS wheels.
- **Python 3.10 or newer, running as arm64.** See the note below — this is the
  one thing that reliably goes wrong.
- About 3 GB of disk for dependencies and model weights.

## Install

```bash
pipx install msgsearch
msgsearch doctor
```

This pulls PyTorch, around 1 GB, so the first install is slow. Everything after
that is local and fast.

<details>
<summary><b>If that fails with "No matching distribution found for torch"</b></summary>

You have an Intel-built Python, which is common on Apple silicon after migrating
from an Intel Mac. Check:

```bash
python3 -c "import platform; print(platform.machine())"   # must say arm64
```

The confusing part is that `pipx install --python /path/to/arm64/python` often
**does not fix it**. Python from python.org is a *universal* binary that runs as
whichever architecture its parent process is, and if pipx itself was installed by
an Intel Homebrew (`/usr/local/...`), it launches that Python as x86_64. pip then
looks for x86_64 wheels that PyTorch does not publish.

Install into a venv created explicitly under arm64 instead:

```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    -m venv ~/.msgsearch-venv
~/.msgsearch-venv/bin/pip install msgsearch
ln -sf ~/.msgsearch-venv/bin/msgsearch /usr/local/bin/msgsearch
```

Adjust the interpreter path to any arm64 Python 3.10+. To undo:
`rm /usr/local/bin/msgsearch && rm -rf ~/.msgsearch-venv`.

Architecture cannot be expressed in package metadata, which is why this surfaces
as a dependency-resolution wall rather than a useful error. `msgsearch doctor`
reports it in one line.
</details>

To work on msgsearch rather than use it:

```bash
git clone https://github.com/dhruv1707/msgsearch && cd msgsearch
arch -arm64 python3 -m venv .venv          # must be an arm64 interpreter
./.venv/bin/python -m pip install -e ".[dev]"
```

## Setup

### 1. Model access

The default embedding model, `google/embeddinggemma-300m`, is gated: Google
requires you to accept its licence before downloading. This is a one-time thing.

1. **Create a HuggingFace account** if you do not have one — <https://huggingface.co/join>. Free.
2. **Accept the licence.** Open <https://huggingface.co/google/embeddinggemma-300m>,
   sign in, and click the button acknowledging the Gemma terms at the top of the
   page. Approval is normally immediate.
3. **Create a token.** Go to <https://huggingface.co/settings/tokens>, click
   *Create new token*, give it the **Read** role, and copy it. It looks like
   `hf_...`.
4. **Log in:**

   ```bash
   msgsearch login
   ```

   Paste the token when prompted. It is stored in `~/.cache/huggingface`, not by
   msgsearch. You can also pass it directly with `msgsearch login --token hf_...`.

`msgsearch login` checks afterwards that the model is genuinely reachable, rather
than only that the token is valid. Those are different conditions — you can be
perfectly logged in and still be refused because step 2 was skipped — and
HuggingFace reports both as the same 401 saying "please log in", which sends
people back to re-do the step they already did.

The model itself downloads automatically the first time you index, about 1.2 GB,
cached in `~/.cache/huggingface`. There is no separate download step.

**Don't want a HuggingFace account?** Use an ungated model instead. Retrieval is
somewhat weaker, but nothing else changes:

```bash
export MSGSEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5
```

### 2. Snapshot your messages

**Never point this tool at `~/Library/Messages`** — that file is live, Messages
holds locks on it, and it is irreplaceable.

```bash
msgsearch sync
```

This needs Full Disk Access (see above for how to grant it). It uses SQLite's
backup API rather than `cp`, which matters more than it sounds: Messages runs in
WAL mode, so your most recent messages live in a `chat.db-wal` sidecar until they
are checkpointed. `cp chat.db` alone drops exactly the messages you are most
likely to search for, silently. The snapshot folds the write-ahead log in and
leaves a single self-contained file.

### 3. Names (optional)

Without names, speakers appear as phone numbers. Resolving them makes results
readable *and* improves retrieval, because the speaker label is part of the text
that gets embedded — `Sam: ...` carries meaning where `+15551234567: ...` does not.

The simplest route is to grant Contacts access to whatever runs msgsearch
(System Settings → Privacy & Security → Contacts — that is your terminal, or your
editor if you run it from one). Names are then read automatically and stay
current; nothing else is needed.

Contacts is a separate permission from Full Disk Access, and macOS grants it to
the *app*, not the shell — so if you run msgsearch from an editor's integrated
terminal, the grant has to go to the editor. To sidestep that entirely, copy the
database from an app that does hold the permission and point at the copy, exactly
as you did for `chat.db`:

```bash
cp ~/Library/Application\ Support/AddressBook/AddressBook-v22.abcddb ~/msgsearch/
export MSGSEARCH_ADDRESSBOOK=~/msgsearch/AddressBook-v22.abcddb
```

If you would rather not do either, or want to correct a name or label someone who
is not in your address book, use the alias file instead:

```bash
./.venv/bin/msgsearch contacts                 # what is resolved, and from where
./.venv/bin/msgsearch contacts --template 20   # stub for the 20 busiest handles
$EDITOR ~/msgsearch/contacts.json              # fill in names; blanks are ignored
./.venv/bin/msgsearch contacts --import out.vcf  # or import a vCard export
```

The alias file overrides Contacts entry by entry, so a nickname you prefer wins.
A handful of names goes a long way: on a typical archive the ten busiest handles
account for over 90% of received messages.

Do this *before* building the index. Changing a speaker's name changes the text
that was embedded, so it costs a full re-index afterwards.

## Use

```bash
./.venv/bin/msgsearch doctor                  # is this machine set up correctly?
./.venv/bin/msgsearch explore                 # what is in your database
./.venv/bin/msgsearch index                   # build the index
./.venv/bin/msgsearch search "atria login"    # search it
```

If anything goes wrong, run `msgsearch doctor` first: it checks the interpreter
architecture, PyTorch, the database, model access and the index, and prints the
command that fixes whatever is broken.

Indexing everything takes a while — roughly half an hour per 100k messages on an
M-series Mac — so restrict it to one conversation while trying things out:

```bash
./.venv/bin/msgsearch index --chat '+15551234567'
```

When new messages arrive, refresh and re-index:

```bash
msgsearch sync --index
```

Re-running `msgsearch index` later is cheap. Embeddings are cached by passage
content, so an ordinary top-up only embeds text that is genuinely new — a rebuild
with nothing new to do takes seconds rather than half an hour. Use `--rebuild` to
force everything to be recomputed.

Useful search flags:

```
--limit N        how many results
--chat TEXT      restrict to conversations matching TEXT
--from WHO       restrict to a speaker
--after / --before YYYY-MM-DD
--type credential    only windows that appear to CONTAIN a credential
                     (credential_talk is the separate tag for windows that only
                     discuss one; also: email, phone, url, address)
--full           show the whole conversation window, not just the match
--rerank         run the cross-encoder (off by default; see below)
--no-dense       keyword search only
--no-bm25        vector search only
```

`--type credential` is the flag worth knowing about. Searching for a forgotten
password without it returns mostly people *discussing* the password; with it, the
message containing one tends to come first.

Reranking is **off by default** because it was measured and it hurts: over the
gold set it drops MRR from 0.84 to 0.70, and on broad topical queries it takes
rankings that fusion got right and scrambles them. `--rerank` turns it back on if
you want to see for yourself.

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
