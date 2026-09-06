"""Configuration for msgsearch.

Nothing personal belongs in this file, because it is tracked in git. Paths and any
chat filter come from environment variables, or from a `config_local.py` that is
listed in .gitignore and never committed.
"""

import contextlib
import os
from pathlib import Path

# --- paths -------------------------------------------------------------------

# A *copy* of the Messages database. Never point this at ~/Library/Messages: that
# file is live, and Messages.app holds locks on it.
DB_PATH = Path(os.environ.get("MSGSEARCH_DB", "~/msgsearch/chat.db")).expanduser()

# Where the built index lives. Kept outside the repo so it can never be committed.
INDEX_DIR = Path(os.environ.get("MSGSEARCH_INDEX", "~/msgsearch/index")).expanduser()

# --- scope -------------------------------------------------------------------

# macOS Contacts, for turning handles into names. Reading the system copy needs
# the Contacts permission (which is separate from Full Disk Access), so this can
# instead point at a copy made by an app that has it — exactly as DB_PATH points
# at a copy of chat.db:
#   cp ~/Library/Application\ Support/AddressBook/AddressBook-v22.abcddb ~/msgsearch/
ADDRESSBOOK_PATH = Path(
    os.environ.get(
        "MSGSEARCH_ADDRESSBOOK",
        "~/Library/Application Support/AddressBook/AddressBook-v22.abcddb",
    )
).expanduser()

# Handle -> name mapping, so speakers appear by name rather than phone number.
# Optional: without it, handles are used as-is. Kept outside the repo because it
# is a list of everyone you have ever texted.
CONTACTS_FILE = Path(
    os.environ.get("MSGSEARCH_CONTACTS", "~/msgsearch/contacts.json")
).expanduser()

# Restrict indexing to a single conversation while developing. This is a phone
# number or email address, so it must come from the environment and never be
# written into a tracked file.
TESTBED_CHAT = os.environ.get("MSGSEARCH_TESTBED_CHAT") or None

# --- chunking ----------------------------------------------------------------

# A pause longer than this ends a conversation window.
WINDOW_GAP_SECONDS = 30 * 60

# Windows longer than this are split. The embedding model accepts 2048 tokens;
# the budget sits below that so a split window still has room for its date header.
# Measured on real data, this splits about 1% of windows and leaves the rest alone.
WINDOW_TOKEN_BUDGET = 1500

# When a window is split, the last few messages are repeated at the start of the
# next piece so an exchange that straddles the boundary survives in one of them.
WINDOW_OVERLAP_MESSAGES = 2

# Rough bytes-per-token ratio, used only to decide when to split. Deliberately
# approximate: the real tokenizer is not loaded during chunking.
CHARS_PER_TOKEN = 4

# --- passages ----------------------------------------------------------------

# What actually gets embedded. A whole conversation window is the right unit to
# *show* someone, but the wrong unit to embed: compressing thirty-odd messages on
# mixed topics into one vector buries any specific fact in them. Measured on real
# data, embedding a 3-message neighbourhood moved a target from rank 3539 to rank
# 13, while embedding the single message on its own was worst of all at 5117.
# So passages are small, they slide across the window, and each one points back
# at the window it came from for display.
PASSAGE_MESSAGES = 3
PASSAGE_STRIDE = 1

# --- models ------------------------------------------------------------------

# Gated on HuggingFace: accept the model terms and `hf auth login` once, or this
# 401s. Must match the model the index was built with -- search.py checks.
EMBED_MODEL = os.environ.get("MSGSEARCH_EMBED_MODEL", "google/embeddinggemma-300m")

# Reranking is OFF by default because it measurably hurts. Over the 8-query gold
# set it drops MRR from 0.823 to 0.700, and on topical queries from 1.000 to
# 0.700: the cross-encoder takes rankings fusion got right and scrambles them.
# These rerankers are trained on clean QA passages, and a window of "Yaaa bro I
# do" is far outside that distribution. It is also slow -- measured over 50
# candidates, Qwen3-Reranker takes 4.2s against 1.1s for bge-reranker-v2-m3 and
# 0.15s for ms-marco-MiniLM.
#
# Re-test it whenever chunking, the embedding model or the gold set changes:
#   MSGSEARCH_RERANK=1 .venv/bin/python eval/bench.py -r eval.retriever
RERANK_ENABLED = os.environ.get("MSGSEARCH_RERANK", "0") == "1"
RERANK_MODEL = os.environ.get("MSGSEARCH_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

# --- retrieval ---------------------------------------------------------------

BM25_CANDIDATES = 100
DENSE_CANDIDATES = 100
RRF_CANDIDATES = 50
RRF_K = 60
DEFAULT_LIMIT = 10


# Optional untracked overrides. Anything defined in config_local.py wins.
with contextlib.suppress(ImportError):
    from config_local import *  # noqa: F403
