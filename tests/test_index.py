"""Tests for incremental indexing.

Embedding is the expensive part of a build, so passages whose text has not
changed keep the vector they had. These tests cover the conditions under which
reuse is *unsound* and must not happen, because a silently stale or mismatched
vector produces confident nonsense rather than an error.

No model is loaded here; the cache is exercised against a synthetic index.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from msgsearch.index import SCHEMA, load_embedding_cache, passage_hash


def write_index(directory: Path, model="test-model", dimension=4, passages=(("a", 0),)):
    """Write a minimal index containing the given (text_hash, vector_row) pairs."""
    db_path = directory / "index.db"
    vec_path = directory / "vectors.npy"

    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    db.executemany(
        "INSERT INTO meta (key, value) VALUES (?,?)",
        [("embed_model", model), ("dimension", str(dimension))],
    )
    db.executemany(
        """INSERT INTO passages
           (vector_row, passage_id, window_id, window_row, text, text_hash, rowids)
           VALUES (?,?,?,?,?,?,?)""",
        [
            (row, f"p{row}", "w1", 0, "text", text_hash, "[]")
            for text_hash, row in passages
        ],
    )
    db.commit()
    db.close()

    np.save(vec_path, np.ones((len(passages), dimension), dtype=np.float32))
    return db_path, vec_path


class TestPassageHash(unittest.TestCase):
    def test_stable_across_calls(self):
        self.assertEqual(passage_hash("Me: hello"), passage_hash("Me: hello"))

    def test_differs_on_any_change(self):
        self.assertNotEqual(passage_hash("Me: hello"), passage_hash("Me: hello "))

    def test_handles_non_ascii(self):
        self.assertTrue(passage_hash("Me: 😂 déjà vu"))


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reuses_vectors_from_a_matching_index(self):
        db, vec = write_index(self.dir, passages=(("hash-a", 0), ("hash-b", 1)))
        cache = load_embedding_cache(db, vec, "test-model", 4)
        self.assertEqual(set(cache), {"hash-a", "hash-b"})
        self.assertEqual(cache["hash-a"].shape, (4,))

    def test_no_previous_index_yields_nothing(self):
        cache = load_embedding_cache(
            self.dir / "missing.db", self.dir / "missing.npy", "test-model", 4
        )
        self.assertEqual(cache, {})

    def test_a_different_model_invalidates_everything(self):
        # Vectors from two models are not comparable, so reusing across a model
        # change would corrupt the index without any error being raised.
        db, vec = write_index(self.dir, model="model-one")
        self.assertEqual(load_embedding_cache(db, vec, "model-two", 4), {})

    def test_a_different_dimension_invalidates_everything(self):
        db, vec = write_index(self.dir, dimension=4)
        self.assertEqual(load_embedding_cache(db, vec, "test-model", 768), {})

    def test_an_index_without_hashes_invalidates_everything(self):
        # Indexes built before content hashing have no text_hash column. Reading
        # one must degrade to a full rebuild rather than raise.
        db, vec = write_index(self.dir)
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE passages")
        conn.execute("CREATE TABLE passages (vector_row INTEGER, text TEXT)")
        conn.commit()
        conn.close()
        self.assertEqual(load_embedding_cache(db, vec, "test-model", 4), {})

    def test_rows_beyond_the_vector_array_are_skipped(self):
        # A truncated or mismatched vectors.npy must not produce an IndexError.
        db, vec = write_index(self.dir, passages=(("hash-a", 0), ("hash-b", 99)))
        cache = load_embedding_cache(db, vec, "test-model", 4)
        self.assertEqual(set(cache), {"hash-a"})


if __name__ == "__main__":
    unittest.main()
