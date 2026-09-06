# Changelog

Notable changes to msgsearch. Retrieval changes carry the measurement that
justified them; see `ARCHITECTURE.md` for the full evaluation.

## [0.2.3] — 2026-09-06

### Fixed
- Model-loading failures printed a Python traceback around their instructions,
  which reads as a crash rather than "you need to log in" and buries the steps
  under a stack. They now print the guidance alone and exit non-zero.

## [0.2.2] — 2026-09-06

### Added
- `msgsearch login`, which authenticates with HuggingFace. The instructions
  previously said to run `hf auth login`, but installing msgsearch does not put
  `hf` on your PATH -- pipx exposes only the entry points a package declares, so
  that step failed with "command not found" for everyone who installed normally.
  It also verifies afterwards that the gated model is actually reachable, since
  being logged in and having accepted the model licence are different things
  whose failures look identical.

### Changed
- Full Disk Access instructions are now step by step, including that macOS grants
  it to the application rather than the shell and that the application must be
  restarted before it takes effect.

## [0.2.1] — 2026-09-06

Documentation only. PyPI freezes a project's description at publish time, so
correcting the README required a release.

### Fixed
- The install instructions recommended `pipx install --python <arm64 python>`,
  which does not work on Apple silicon with an Intel Homebrew. Python from
  python.org is a universal binary that runs as whichever architecture its
  parent process is, so an Intel-built pipx launches it as x86_64 whichever
  interpreter you name, and PyTorch has no x86_64 macOS wheels. The working
  recipe creates the venv under `arch -arm64`.

### Added
- A quickstart covering install through first search in one block, and an
  explicit note that Full Disk Access and Contacts are separate permissions
  granted to the application rather than the shell.

## [0.2.0] — 2026-09-06

### Added
- `msgsearch sync`, which refreshes the working copy of the messages database.
  `--index` chains straight into a rebuild, so keeping an index current is one
  command.

### Fixed
- `requires-python` excluded Python 3.14, so `pipx install msgsearch` failed with
  "Could not find a version that satisfies the requirement" on any machine whose
  default interpreter was 3.14. The bound was set on a misdiagnosis: PyTorch does
  publish cp314 wheels, and the real constraint is Apple silicon, which package
  metadata cannot express. CI now covers 3.14 so this cannot recur.
- The documented `cp ~/Library/Messages/chat.db*` could silently lose recent
  messages. Messages runs SQLite in WAL mode, so the newest messages sit in a
  `chat.db-wal` sidecar until checkpointed; copying only the main file drops
  them, and copying a live database at all risks a torn read. `sync` uses
  SQLite's backup API, which takes a consistent snapshot with the WAL folded in
  and leaves a single self-contained file.

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
