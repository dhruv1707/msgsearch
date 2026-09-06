"""Tests for snapshotting the messages database.

These run against synthetic databases. Nothing here touches the real
~/Library/Messages, which is the one file this project must never write to.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from msgsearch.sync import snapshot


def make_db(path: Path, rows=3, wal=True):
    db = sqlite3.connect(path)
    if wal:
        db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT)")
    db.executemany(
        "INSERT INTO message (text) VALUES (?)", [(f"m{i}",) for i in range(rows)]
    )
    db.commit()
    db.close()
    return path


def count(path: Path) -> int:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return db.execute("SELECT count(*) FROM message").fetchone()[0]
    finally:
        db.close()


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = make_db(self.dir / "live.db", rows=5)

    def test_copies_every_row(self):
        out = snapshot(self.source, self.dir / "copy.db")
        self.assertEqual(count(out), 5)

    def test_leaves_the_source_untouched(self):
        before = self.source.stat().st_mtime_ns
        snapshot(self.source, self.dir / "copy.db")
        self.assertEqual(self.source.stat().st_mtime_ns, before)
        self.assertEqual(count(self.source), 5)

    def test_snapshot_is_a_single_self_contained_file(self):
        # A WAL-mode copy would sprout -wal and -shm on every later read, which
        # makes it unsafe to move and easy to half-copy.
        out = snapshot(self.source, self.dir / "copy.db")
        count(out)  # a read, which is what would recreate the sidecars
        self.assertFalse(out.with_name("copy.db-wal").exists())
        self.assertFalse(out.with_name("copy.db-shm").exists())

    def test_stale_sidecars_from_an_earlier_copy_are_removed(self):
        # These belong to a different database; SQLite would replay them over
        # the new snapshot.
        destination = self.dir / "copy.db"
        destination.with_name("copy.db-wal").write_bytes(b"stale")
        snapshot(self.source, destination)
        self.assertFalse(destination.with_name("copy.db-wal").exists())
        self.assertEqual(count(destination), 5)

    def test_overwrites_an_existing_snapshot(self):
        destination = make_db(self.dir / "copy.db", rows=1)
        snapshot(self.source, destination)
        self.assertEqual(count(destination), 5)

    def test_refuses_to_write_onto_the_source(self):
        with self.assertRaises(ValueError):
            snapshot(self.source, self.source)

    def test_missing_source_is_reported_clearly(self):
        with self.assertRaises(FileNotFoundError):
            snapshot(self.dir / "nope.db", self.dir / "copy.db")

    def test_creates_the_destination_directory(self):
        out = snapshot(self.source, self.dir / "nested" / "deeper" / "copy.db")
        self.assertTrue(out.exists())
        self.assertEqual(count(out), 5)


if __name__ == "__main__":
    unittest.main()
