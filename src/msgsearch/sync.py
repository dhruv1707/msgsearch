"""Take a snapshot of the live Messages database.

The working copy goes stale as soon as new messages arrive, so refreshing it is
a routine chore rather than a one-off. Two things make doing it by hand
error-prone.

First, `cp chat.db` alone is wrong. Messages runs SQLite in WAL mode, so recent
messages live in a `chat.db-wal` sidecar until they are checkpointed into the
main file — a couple of megabytes of exactly the messages you are most likely to
search for. Copying only the main file loses them silently.

Second, copying a database that is being written to can produce a torn file,
because `cp` has no idea it is reading a moving target.

SQLite's backup API solves both: it takes a consistent snapshot with the WAL
already folded in, and writes a single self-contained file with no sidecars.
It opens the source read-only and never writes to it.

This still needs Full Disk Access, which macOS grants to the *application*
running msgsearch. A terminal usually has it; an editor's integrated terminal
often does not.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config

LIVE_DB = Path("~/Library/Messages/chat.db").expanduser()


def snapshot(source: Path | None = None, destination: Path | None = None) -> Path:
    """Copy the live message database to the working location.

    Returns the path written. Raises PermissionError when macOS denies access,
    which is the normal outcome without Full Disk Access.
    """
    source = Path(source or LIVE_DB).expanduser()
    destination = Path(destination or config.DB_PATH).expanduser()

    if not source.exists():
        raise FileNotFoundError(f"No Messages database at {source}")

    # Backing up onto the source would be writing to the live database, which is
    # the one thing this project must never do.
    if source.resolve() == destination.resolve():
        raise ValueError(
            f"Source and destination are the same file ({source}). "
            f"MSGSEARCH_DB must point at a copy, never at the live database."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Sidecars left by an *earlier* copy belong to a different database. SQLite
    # would replay them over the new snapshot, so they have to go before the
    # backup rather than after it.
    for suffix in ("-wal", "-shm"):
        stale = destination.with_name(destination.name + suffix)
        if stale.exists():
            stale.unlink()

    try:
        live = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise PermissionError(
            f"Cannot read {source}. Grant Full Disk Access to the application "
            f"running msgsearch, under System Settings > Privacy & Security > "
            f"Full Disk Access, then restart it."
        ) from error

    try:
        live.execute("SELECT 1 FROM message LIMIT 1")
    except sqlite3.Error as error:
        live.close()
        raise PermissionError(
            f"Opened {source} but cannot read from it, which usually means Full "
            f"Disk Access has not been granted to the application running "
            f"msgsearch."
        ) from error

    copy = sqlite3.connect(destination)
    try:
        live.backup(copy)
        # The backup inherits WAL journalling from the source, which means every
        # later read recreates -wal and -shm beside the file. Switching to
        # DELETE mode makes the snapshot a genuinely self-contained single file,
        # which is what makes it safe to move, and costs nothing for a workload
        # that only ever reads.
        copy.execute("PRAGMA journal_mode=DELETE")
    finally:
        copy.close()
        live.close()

    return destination


def describe(path: Path | None = None) -> str:
    """A one-line summary of what a snapshot contains."""
    path = Path(path or config.DB_PATH).expanduser()
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        total, newest = db.execute("SELECT count(*), max(date) FROM message").fetchone()
    finally:
        db.close()

    size = path.stat().st_size / 1e6
    if newest:
        from .extract import apple_timestamp

        when = apple_timestamp(newest).astimezone().strftime("%Y-%m-%d %H:%M")
    else:
        when = "unknown"
    return f"{total:,} messages, {size:.0f} MB, newest {when}"
