#!/usr/bin/env python3
"""Read-only reconnaissance of a copy of the iMessage chat.db.

Never point this at ~/Library/Messages -- work on a copy (default ~/msgsearch/chat.db).
Prints structural stats only; no message bodies are read or displayed.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

DB_PATH = os.path.expanduser(os.environ.get("MSGSEARCH_DB", "~/msgsearch/chat.db"))

# Apple's Core Data epoch: 2001-01-01 00:00:00 UTC
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def apple_ts(value):
    """Convert a message.date value to a UTC datetime.

    Older rows store seconds since the Apple epoch; macOS 10.13+ stores
    nanoseconds. Anything above ~1e12 is far beyond any plausible second
    count (year 33000+), so it is nanoseconds.
    """
    if value is None:
        return None
    seconds = value / 1e9 if abs(value) > 1e12 else float(value)
    return APPLE_EPOCH + timedelta(seconds=seconds)


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "n/a"


def connect(path):
    if not os.path.exists(path):
        sys.exit(f"error: no database at {path}")
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def main():
    conn = connect(DB_PATH)
    cur = conn.cursor()

    print(f"database: {DB_PATH}")
    print(f"size:     {os.path.getsize(DB_PATH) / 1e6:.1f} MB")
    print()

    total, = cur.execute("SELECT count(*) FROM message").fetchone()
    null_text, = cur.execute(
        "SELECT count(*) FROM message WHERE text IS NULL"
    ).fetchone()
    null_text_with_body, = cur.execute(
        "SELECT count(*) FROM message "
        "WHERE text IS NULL AND attributedBody IS NOT NULL"
    ).fetchone()
    unrecoverable = null_text - null_text_with_body

    def pct(n, d):
        return f"{(100.0 * n / d):5.1f}%" if d else "  n/a"

    print("== message volume ==")
    print(f"total messages ............................. {total:>9,}")
    print(f"  text IS NULL ............................. {null_text:>9,}  ({pct(null_text, total)} of total)")
    print(f"    ...but attributedBody IS NOT NULL ...... {null_text_with_body:>9,}  ({pct(null_text_with_body, null_text)} of NULL-text)")
    print(f"    ...and no attributedBody either ........ {unrecoverable:>9,}  ({pct(unrecoverable, null_text)} of NULL-text)")
    recoverable = total - unrecoverable
    print(f"messages with recoverable text ............. {recoverable:>9,}  ({pct(recoverable, total)} of total)")
    print()

    print("== date range (message.date, Apple epoch 2001-01-01) ==")
    lo, hi = cur.execute(
        "SELECT min(date), max(date) FROM message WHERE date IS NOT NULL AND date != 0"
    ).fetchone()
    print(f"raw min .... {lo}")
    print(f"raw max .... {hi}")
    print(f"earliest ... {fmt(apple_ts(lo))}")
    print(f"latest ..... {fmt(apple_ts(hi))}")
    if lo and hi:
        span = apple_ts(hi) - apple_ts(lo)
        print(f"span ....... {span.days:,} days (~{span.days / 365.25:.1f} years)")
    ns_rows, = cur.execute(
        "SELECT count(*) FROM message WHERE date IS NOT NULL AND abs(date) > 1000000000000"
    ).fetchone()
    print(f"nanosecond-scale rows ... {ns_rows:,} / second-scale rows ... {total - ns_rows:,}")
    print()

    print("== top 10 chats by message count ==")
    rows = cur.execute(
        """
        SELECT c.ROWID,
               c.style,
               COALESCE(NULLIF(c.display_name, ''), c.chat_identifier) AS label,
               c.service_name,
               count(*) AS n,
               sum(m.text IS NULL) AS n_null_text,
               max(m.date) AS last_date
        FROM chat_message_join cmj
        JOIN chat c ON c.ROWID = cmj.chat_id
        JOIN message m ON m.ROWID = cmj.message_id
        GROUP BY c.ROWID
        ORDER BY n DESC
        LIMIT 10
        """
    ).fetchall()

    hdr = f"{'#':>2}  {'msgs':>8}  {'null':>7}  {'kind':<6}  {'svc':<8}  {'last activity':<16}  chat"
    print(hdr)
    print("-" * len(hdr))
    for i, (rowid, style, label, service, n, n_null, last_date) in enumerate(rows, 1):
        kind = "group" if style == 43 else "1:1"
        last = apple_ts(last_date)
        print(
            f"{i:>2}  {n:>8,}  {n_null:>7,}  {kind:<6}  {(service or '?'):<8}  "
            f"{(last.strftime('%Y-%m-%d') if last else 'n/a'):<16}  "
            f"[chat {rowid}] {label or '(unnamed)'}"
        )
    print()

    print("== other counts ==")
    for label, sql in [
        ("distinct chats", "SELECT count(*) FROM chat"),
        ("distinct handles (senders)", "SELECT count(*) FROM handle"),
        ("messages from me", "SELECT count(*) FROM message WHERE is_from_me = 1"),
        ("messages with attachments", "SELECT count(*) FROM message WHERE cache_has_attachments = 1"),
        ("reactions/tapbacks", "SELECT count(*) FROM message WHERE associated_message_type != 0"),
        ("orphan messages (no chat)",
         "SELECT count(*) FROM message m LEFT JOIN chat_message_join cmj "
         "ON cmj.message_id = m.ROWID WHERE cmj.chat_id IS NULL"),
    ]:
        n, = cur.execute(sql).fetchone()
        print(f"{label:.<45} {n:>9,}")

    conn.close()


if __name__ == "__main__":
    main()
