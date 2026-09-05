"""Build the local search index: decode every message once, store it in SQLite FTS5.

This is the substrate for every retriever. Dense vectors get added as a second
table later (see ARCHITECTURE.md); FTS5 alone is the lexical baseline to beat.

Reads ~/msgsearch/chat.db read-only. Writes ./index.sqlite, which is gitignored
because it contains message bodies.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attributed_body import message_text
from explore import apple_ts, connect

INDEX_PATH = os.environ.get("MSGSEARCH_INDEX", "index.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,   -- message.ROWID
    chat_id     INTEGER,
    chat_label  TEXT,
    handle      TEXT,
    is_from_me  INTEGER,
    ts          TEXT,                  -- ISO8601 UTC
    body        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_ts ON messages(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(body, content='messages', content_rowid='id', tokenize='porter unicode61');
"""

# Tapbacks ("Liked “...”") are not messages; they flood any ranked list.
SOURCE_SQL = """
SELECT m.ROWID, m.text, m.attributedBody, m.is_from_me, m.date,
       h.id AS handle,
       c.ROWID AS chat_id,
       COALESCE(NULLIF(c.display_name, ''), c.chat_identifier) AS chat_label
FROM message m
LEFT JOIN handle h ON h.ROWID = m.handle_id
LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
LEFT JOIN chat c ON c.ROWID = cmj.chat_id
WHERE m.associated_message_type = 0
"""


def build(index_path=INDEX_PATH, limit=None):
    src = connect(os.path.expanduser(os.environ.get("MSGSEARCH_DB", "~/msgsearch/chat.db")))
    out = sqlite3.connect(index_path)
    out.executescript(SCHEMA)

    sql = SOURCE_SQL + (f" LIMIT {int(limit)}" if limit else "")
    rows, kept, skipped = [], 0, 0
    for rid, text, blob, from_me, date, handle, chat_id, chat_label in src.execute(sql):
        body = message_text(text, blob)
        if not body or not body.strip():
            skipped += 1
            continue
        ts = apple_ts(date)
        rows.append((rid, chat_id, chat_label, handle, from_me,
                     ts.isoformat() if ts else None, body.strip()))
        kept += 1
        if len(rows) >= 5000:
            _flush(out, rows)
            rows = []
    if rows:
        _flush(out, rows)

    out.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    out.commit()
    src.close()
    out.close()
    return kept, skipped


def _flush(out, rows):
    out.executemany(
        "INSERT OR REPLACE INTO messages(id, chat_id, chat_label, handle, is_from_me, ts, body)"
        " VALUES (?,?,?,?,?,?,?)", rows)
    out.commit()


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    kept, skipped = build(limit=limit)
    print(f"indexed {kept:,} messages ({skipped:,} empty/undecodable) -> {INDEX_PATH}")
