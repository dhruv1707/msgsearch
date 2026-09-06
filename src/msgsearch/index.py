"""Build the search index.

The index is two files sitting side by side in `config.INDEX_DIR`:

  index.db     conversation windows (what you get shown, and what keyword search
               runs over), plus the passages carved out of them
  vectors.npy  one unit-length row per *passage*, in the same order as the
               `vector_row` column of the passages table

The split between windows and passages is the important part. A window is a run
of messages with no long pause in it, which is the right amount of context to put
in front of a person. A passage is a three-message slice of a window, which is
the right amount of text to embed: measured on real data, embedding whole windows
buried a target answer at rank 3539, while embedding three-message slices of the
same conversation put it at rank 13.

Keeping the vectors in a plain array is deliberate. A search is one matrix
multiply, which at this scale beats any index structure once its overhead is
counted, and it removes a dependency.

The output belongs outside the repository. It is a plaintext, searchable copy of
every private thing anyone has ever sent you.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from . import config
from .chunk import passages, windows
from .embedder import Embedder
from .extract import connect, iter_messages

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_row  INTEGER PRIMARY KEY,
    window_id   TEXT UNIQUE NOT NULL,
    chat_id     INTEGER NOT NULL,
    chat_label  TEXT NOT NULL,
    start_ts    TEXT NOT NULL,
    end_ts      TEXT NOT NULL,
    n_messages  INTEGER NOT NULL,
    speakers    TEXT NOT NULL,
    tags        TEXT NOT NULL,
    one_sided   INTEGER NOT NULL,
    search_text TEXT NOT NULL,
    offsets     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS windows_start ON windows(start_ts);
CREATE INDEX IF NOT EXISTS windows_chat ON windows(chat_id);

CREATE TABLE IF NOT EXISTS passages (
    vector_row  INTEGER PRIMARY KEY,
    passage_id  TEXT UNIQUE NOT NULL,
    window_id   TEXT NOT NULL,
    window_row  INTEGER NOT NULL,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    rowids      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS passages_window ON passages(window_row);
CREATE INDEX IF NOT EXISTS passages_hash ON passages(text_hash);

-- An external-content table: FTS5 indexes the window text without copying it.
CREATE VIRTUAL TABLE IF NOT EXISTS windows_fts USING fts5(
    search_text,
    content='windows',
    content_rowid='window_row',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def passage_hash(text: str) -> str:
    """Content address for a passage, used to reuse its embedding across builds."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def load_embedding_cache(
    db_path: Path, vec_path: Path, model_name: str, dimension: int
) -> dict[str, np.ndarray]:
    """Map passage content hash to its embedding, from a previous build.

    Reuse is keyed on the passage text rather than on message ids or dates, which
    is what makes this correct in the awkward cases. New messages can *extend* the
    last window of a conversation, changing which passages it yields; a passage
    whose text is unchanged keeps its vector, and one whose text changed is
    re-embedded, without either case needing to be detected explicitly. Changing
    `PASSAGE_MESSAGES` likewise rewrites every passage text, so the cache misses
    everywhere and the whole index is rebuilt, which is the correct behaviour.

    An empty cache is returned whenever reuse would be unsound: no previous index,
    a different embedding model, a different dimension, or an index predating this
    scheme.
    """
    if not (db_path.exists() and vec_path.exists()):
        return {}

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        meta = dict(db.execute("SELECT key, value FROM meta"))
        if meta.get("embed_model") != model_name:
            return {}
        if int(meta.get("dimension", 0)) != dimension:
            return {}
        try:
            rows = db.execute("SELECT text_hash, vector_row FROM passages").fetchall()
        except sqlite3.OperationalError:
            return {}  # index built before passages carried a hash
    finally:
        db.close()

    vectors = np.load(vec_path)
    return {
        text_hash: vectors[row]
        for text_hash, row in rows
        if text_hash and 0 <= row < len(vectors)
    }


def build(
    chat_identifier: str | None = None,
    index_dir: Path | None = None,
    batch_size: int = 32,
    rebuild: bool = False,
) -> Path:
    """Build the index, reusing embeddings from a previous build where possible.

    Windows and passages are always recomputed, which is cheap. Embedding is the
    expensive step, so passages whose text is unchanged keep the vector they had.
    Pass `rebuild=True` to embed everything from scratch.
    """
    index_dir = Path(index_dir or config.INDEX_DIR).expanduser()
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.npy"

    started = time.time()

    print("reading messages...", flush=True)
    source = connect()
    try:
        messages = list(iter_messages(source, chat_identifier=chat_identifier))
    finally:
        source.close()
    print(f"  {len(messages):,} messages", flush=True)

    print("building conversation windows...", flush=True)
    all_windows = list(windows(messages))
    print(f"  {len(all_windows):,} windows", flush=True)

    print("carving passages...", flush=True)
    all_passages = []
    window_rows = {w.window_id: row for row, w in enumerate(all_windows)}
    for window in all_windows:
        all_passages.extend(passages(window))
    print(
        f"  {len(all_passages):,} passages "
        f"({config.PASSAGE_MESSAGES} messages, stride {config.PASSAGE_STRIDE})",
        flush=True,
    )

    print("loading embedding model...", flush=True)
    embedder = Embedder()
    print(
        f"  {embedder.model_name} on {embedder.device}, {embedder.dimension} dims",
        flush=True,
    )

    hashes = [passage_hash(p.text) for p in all_passages]
    cache = (
        {}
        if rebuild
        else load_embedding_cache(
            db_path, vec_path, embedder.model_name, embedder.dimension
        )
    )

    vectors = np.zeros((len(all_passages), embedder.dimension), dtype=np.float32)
    missing = []
    for row, text_hash in enumerate(hashes):
        cached = cache.get(text_hash)
        if cached is None:
            missing.append(row)
        else:
            vectors[row] = cached

    reused = len(all_passages) - len(missing)
    if cache:
        print(
            f"  reusing {reused:,} embeddings from the previous index "
            f"({100 * reused / max(len(all_passages), 1):.1f}%)",
            flush=True,
        )

    if missing:
        print(f"embedding {len(missing):,} new or changed passages...", flush=True)
        fresh = embedder.embed_documents(
            [all_passages[row].text for row in missing],
            batch_size=batch_size,
            show_progress=True,
        )
        for row, vector in zip(missing, fresh, strict=True):
            vectors[row] = vector
    else:
        print("  nothing new to embed", flush=True)

    print("writing index...", flush=True)
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)

    db.executemany(
        """INSERT INTO windows (window_row, window_id, chat_id, chat_label, start_ts,
                                end_ts, n_messages, speakers, tags, one_sided,
                                search_text, offsets)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                row,
                w.window_id,
                w.chat_id,
                w.chat_label,
                w.start.isoformat(),
                w.end.isoformat(),
                len(w.messages),
                json.dumps(sorted(set(w.speakers))),
                " ".join(sorted(w.tags)),
                int(w.is_one_sided),
                w.search_text,
                json.dumps([list(o) for o in w.offsets]),
            )
            for row, w in enumerate(all_windows)
        ],
    )

    db.executemany(
        """INSERT INTO passages (vector_row, passage_id, window_id, window_row,
                                 text, text_hash, rowids)
           VALUES (?,?,?,?,?,?,?)""",
        [
            (
                row,
                p.passage_id,
                p.window_id,
                window_rows[p.window_id],
                p.text,
                hashes[row],
                json.dumps(list(p.rowids)),
            )
            for row, p in enumerate(all_passages)
        ],
    )

    db.execute("INSERT INTO windows_fts(windows_fts) VALUES('rebuild')")

    db.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
        [
            ("embed_model", embedder.model_name),
            ("dimension", str(embedder.dimension)),
            ("window_count", str(len(all_windows))),
            ("passage_count", str(len(all_passages))),
            ("message_count", str(len(messages))),
            ("gap_seconds", str(config.WINDOW_GAP_SECONDS)),
            ("token_budget", str(config.WINDOW_TOKEN_BUDGET)),
            ("passage_messages", str(config.PASSAGE_MESSAGES)),
            ("passage_stride", str(config.PASSAGE_STRIDE)),
            ("embeddings_reused", str(reused)),
            ("embeddings_computed", str(len(missing))),
            ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            ("chat_filter", chat_identifier or ""),
        ],
    )
    db.commit()
    db.close()

    np.save(vec_path, vectors)

    elapsed = time.time() - started
    print(
        f"done in {elapsed:.0f}s: {len(all_windows):,} windows, "
        f"{len(all_passages):,} passages "
        f"({reused:,} reused, {len(missing):,} embedded), vectors {vectors.shape}",
        flush=True,
    )
    return index_dir
