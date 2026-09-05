"""Configuration for msgsearch.

Nothing personal belongs in this file, because it is tracked in git. Paths and any
chat filter come from environment variables, or from a `config_local.py` that is
listed in .gitignore and never committed.
"""

import os
from pathlib import Path

# --- paths -------------------------------------------------------------------

# A *copy* of the Messages database. Never point this at ~/Library/Messages: that
# file is live, and Messages.app holds locks on it.
DB_PATH = Path(os.environ.get("MSGSEARCH_DB", "~/msgsearch/chat.db")).expanduser()

# Where the built index lives. Kept outside the repo so it can never be committed.
INDEX_DIR = Path(os.environ.get("MSGSEARCH_INDEX", "~/msgsearch/index")).expanduser()

# --- scope -------------------------------------------------------------------

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

# --- models ------------------------------------------------------------------

EMBED_MODEL = os.environ.get("MSGSEARCH_EMBED_MODEL", "google/embeddinggemma-300m")
RERANK_MODEL = os.environ.get("MSGSEARCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# --- retrieval ---------------------------------------------------------------

BM25_CANDIDATES = 100
DENSE_CANDIDATES = 100
RRF_CANDIDATES = 50
RRF_K = 60
DEFAULT_LIMIT = 10


# Optional untracked overrides. Anything defined in config_local.py wins.
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
