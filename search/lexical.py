"""BM25 lexical retriever over the FTS5 index. The floor every other approach must beat.

Retriever contract (all retrievers in this package implement it):

    search(query: str, k: int) -> list[tuple[int, float]]   # (message id, score desc)
"""

import os
import re
import sqlite3

INDEX_PATH = os.environ.get("MSGSEARCH_INDEX", "index.sqlite")

_conn = None


def _db():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(f"file:{INDEX_PATH}?mode=ro", uri=True)
    return _conn


def _fts_query(q):
    """FTS5 has its own query syntax; user text must be quoted term-by-term or a
    stray '-' / '*' / ':' becomes an operator and the query errors out."""
    terms = re.findall(r"\w+", q)
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


def search(query, k=50):
    if not _fts_query(query).strip('"'):
        return []
    rows = _db().execute(
        "SELECT rowid, -bm25(messages_fts) AS score FROM messages_fts"
        " WHERE messages_fts MATCH ? ORDER BY score DESC LIMIT ?",
        (_fts_query(query), k),
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]
