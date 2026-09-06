"""Turn rows of Apple's chat.db into clean, readable message records.

This module is responsible for three things that the rest of the pipeline should
never have to think about again:

1. Recovering message text. About 86% of rows have `text` set to NULL and keep
   their content in the `attributedBody` blob instead, so every row goes through
   `attributed_body.message_text`.
2. Working out who spoke. Apple's `handle_id` column is *not* the sender; it is
   the other party in the conversation, in both directions. Direction comes from
   `is_from_me` alone.
3. Discarding rows that are not messages, such as tapback reactions and group
   membership events.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .attributed_body import message_text
from .contacts import Contacts

# Apple stores timestamps as nanoseconds since this moment, not since 1970.
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

# chat.style: 43 is a group conversation, 45 is one-to-one.
GROUP_STYLE = 43

SELF = "Me"


def apple_timestamp(value: int) -> datetime:
    """Convert a chat.db timestamp to a UTC datetime.

    Modern macOS writes nanoseconds; much older databases wrote seconds. Any
    value past 1e12 is implausible as seconds (it would be the year 33000), so
    magnitude is a safe way to tell the two apart.
    """
    seconds = value / 1e9 if abs(value) > 1e12 else float(value)
    return APPLE_EPOCH + timedelta(seconds=seconds)


@dataclass(frozen=True)
class Chat:
    chat_id: int
    label: str
    is_group: bool


@dataclass(frozen=True)
class Message:
    rowid: int
    guid: str
    chat_id: int
    chat_label: str
    chat_is_group: bool
    timestamp: datetime
    is_from_me: bool
    speaker: str
    service: str
    text: str
    reply_to: str | None = None
    attachments: tuple[str, ...] = field(default_factory=tuple)


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database read-only.

    The URI form is what actually enforces this. Opening the file normally would
    let a stray write corrupt an irreplaceable personal archive.
    """
    path = Path(db_path or config.DB_PATH).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"No database at {path}. Copy one from ~/Library/Messages/chat.db "
            f"(requires Full Disk Access) or set MSGSEARCH_DB."
        )
    safe = str(path).replace("?", "%3f").replace("#", "%23")
    return sqlite3.connect(f"file:{safe}?mode=ro", uri=True)


def load_handles(conn: sqlite3.Connection) -> dict[int, str]:
    """Map handle ROWID to the address behind it (a phone number or email).

    Note that a handle is an address, not a person: someone reachable on both SMS
    and iMessage occupies two rows. Merging those identities is deliberately left
    for later.
    """
    return {rowid: addr for rowid, addr in conn.execute("SELECT ROWID, id FROM handle")}


def load_chats(
    conn: sqlite3.Connection,
    handles: dict[int, str],
    contacts: Contacts | None = None,
) -> dict[int, Chat]:
    """Map chat ROWID to a display label and whether it is a group."""
    contacts = contacts if contacts is not None else Contacts()
    participants: dict[int, list[int]] = defaultdict(list)
    for chat_id, handle_id in conn.execute(
        "SELECT chat_id, handle_id FROM chat_handle_join"
    ):
        participants[chat_id].append(handle_id)

    chats: dict[int, Chat] = {}
    for rowid, style, display_name, identifier in conn.execute(
        "SELECT ROWID, style, display_name, chat_identifier FROM chat"
    ):
        is_group = style == GROUP_STYLE
        label = (display_name or "").strip()

        # Most groups here are unnamed, so fall back to listing who is in them.
        if not label and is_group:
            members = [
                contacts.label(handles.get(h, "?")) for h in participants.get(rowid, [])
            ]
            label = ", ".join(members[:3])
            if len(members) > 3:
                label += f" +{len(members) - 3}"

        if not label and identifier:
            label = contacts.label(identifier)
        chats[rowid] = Chat(rowid, label or f"chat {rowid}", is_group)
    return chats


def load_attachments(conn: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    """Map message ROWID to the mime types attached to it.

    Only the metadata is kept. The files themselves are not read, and there is no
    OCR: knowing that a message carried a JPEG is enough for a first version.
    """
    by_message: dict[int, list[str]] = defaultdict(list)
    for message_id, mime in conn.execute(
        """SELECT j.message_id, a.mime_type
           FROM message_attachment_join j
           JOIN attachment a ON a.ROWID = j.attachment_id"""
    ):
        by_message[message_id].append(mime or "unknown")
    return {k: tuple(v) for k, v in by_message.items()}


# Rows excluded here are not messages at all:
#   associated_message_type != 0  -> tapback reactions ("Liked ...")
#   item_type != 0                -> group renames, joins and leaves
#   is_empty = 1                  -> placeholder rows
# The final clause drops the handful of rows with no recoverable content in
# either column.
_MESSAGE_SQL = """
    SELECT m.ROWID, m.guid, j.chat_id, m.date, m.is_from_me, m.handle_id,
           m.service, m.text, m.attributedBody, m.thread_originator_guid
    FROM message m
    JOIN chat_message_join j ON j.message_id = m.ROWID
    WHERE m.associated_message_type = 0
      AND m.item_type = 0
      AND m.is_empty = 0
      AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
    {chat_filter}
    ORDER BY m.date
"""

_CHAT_FILTER = """
      AND j.chat_id IN (SELECT ROWID FROM chat WHERE chat_identifier = ?)
"""


def iter_messages(
    conn: sqlite3.Connection | None = None,
    chat_identifier: str | None = None,
    contacts: Contacts | None = None,
) -> Iterator[Message]:
    """Yield every indexable message, oldest first.

    `chat_identifier` restricts the results to a single conversation, which is how
    the testbed thread is indexed without processing the whole archive.

    `contacts` maps handles to names. It defaults to the alias file, so speakers
    appear as "Sam" rather than "+15551234567" — which reads better and, because
    the speaker label is part of what gets embedded, retrieves better too.
    """
    own_connection = conn is None
    conn = conn or connect()
    contacts = contacts if contacts is not None else Contacts.load()
    try:
        handles = load_handles(conn)
        chats = load_chats(conn, handles, contacts)
        attachments = load_attachments(conn)

        sql = _MESSAGE_SQL.format(chat_filter=_CHAT_FILTER if chat_identifier else "")
        params = (chat_identifier,) if chat_identifier else ()

        # A message can be joined to more than one chat, so the same ROWID can
        # arrive twice. Keep the first and skip repeats.
        seen: set[int] = set()

        for row in conn.execute(sql, params):
            (
                rowid,
                guid,
                chat_id,
                date,
                is_from_me,
                handle_id,
                service,
                text,
                blob,
                reply_to,
            ) = row

            if rowid in seen:
                continue
            seen.add(rowid)

            body = message_text(text, blob)
            if not body or not body.strip():
                continue

            chat = chats.get(chat_id) or Chat(chat_id, f"chat {chat_id}", False)

            yield Message(
                rowid=rowid,
                guid=guid,
                chat_id=chat_id,
                chat_label=chat.label,
                chat_is_group=chat.is_group,
                timestamp=apple_timestamp(date),
                is_from_me=bool(is_from_me),
                speaker=(
                    SELF
                    if is_from_me
                    else contacts.label(handles.get(handle_id, "unknown"))
                ),
                service=service or "unknown",
                text=body,
                reply_to=reply_to,
                attachments=attachments.get(rowid, ()),
            )
    finally:
        if own_connection:
            conn.close()
