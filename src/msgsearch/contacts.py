"""Resolve handles — phone numbers and email addresses — to human names.

This matters twice over. Obviously it lets you search for what someone said by
their name rather than their phone number. Less obviously, it improves retrieval:
the speaker label is part of the text that gets embedded, so a passage currently
begins `+15551234567: ...`, spending tokens on a string that carries no meaning.
`Sam: ...` is both searchable and a better thing to embed.

Names come from two places, merged:

1. **macOS Contacts.** Reading the system database needs the Contacts permission,
   which is separate from Full Disk Access and is granted to the *app* that runs
   msgsearch. If that is awkward — an editor's integrated terminal, say — copy the
   database from an app that does have the permission and point
   MSGSEARCH_ADDRESSBOOK at the copy, exactly as MSGSEARCH_DB points at a copy of
   chat.db.
2. **An alias file** (`~/msgsearch/contacts.json`), which overrides Contacts
   entry by entry. Use it to correct a name, to label someone who is not in your
   address book, or to work without granting the permission at all. A vCard
   export can populate it.

The alias file wins where both have a name, so a nickname you prefer survives.
Nothing here is required: with neither source, handles are used as-is and
everything still works.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from pathlib import Path

from . import config


def addressbook_path() -> Path:
    """Where to read Contacts from: the system copy, or a copy you made."""
    return Path(config.ADDRESSBOOK_PATH).expanduser()


# Enough digits to identify a person without demanding that two records agree on
# country code or punctuation: "+1 (555) 123-4567" and "5551234567" both reduce
# to the same key. Shorter strings (short codes, some international numbers) are
# kept whole rather than truncated.
PHONE_KEY_DIGITS = 10


def aliases_path() -> Path:
    return Path(config.CONTACTS_FILE).expanduser()


def normalise(handle: str) -> str:
    """Reduce a handle to a key that survives formatting differences."""
    handle = (handle or "").strip()
    if "@" in handle:
        return handle.lower()
    digits = re.sub(r"\D", "", handle)
    if len(digits) > PHONE_KEY_DIGITS:
        return digits[-PHONE_KEY_DIGITS:]
    return digits or handle.lower()


class Contacts:
    """A lookup from handle to display name."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self._by_key = {normalise(k): v for k, v in (mapping or {}).items() if v}

    def __len__(self) -> int:
        return len(self._by_key)

    def __bool__(self) -> bool:
        return bool(self._by_key)

    def name(self, handle: str) -> str | None:
        """The name for this handle, or None if unknown."""
        return self._by_key.get(normalise(handle))

    def label(self, handle: str) -> str:
        """The name for this handle, falling back to the handle itself."""
        return self._by_key.get(normalise(handle)) or handle

    @classmethod
    def load(cls, path: Path | None = None, use_addressbook: bool = True) -> Contacts:
        """Names from macOS Contacts, overridden by the alias file.

        Contacts is read first and the alias file applied on top, so an entry you
        have written by hand always wins. Blank entries in the alias file are
        skipped rather than blanking a name Contacts supplied — the template
        writes blanks for you to fill in, and a half-filled template must not
        erase anything.

        Contacts being unreadable is the ordinary case rather than an error: the
        permission may simply not be granted, and the alias file alone is a
        perfectly good source.
        """
        mapping: dict[str, str] = {}

        if use_addressbook:
            with contextlib.suppress(PermissionError, FileNotFoundError, sqlite3.Error):
                mapping.update(read_addressbook())

        mapping.update({k: v for k, v in load_mapping(path).items() if v})
        return cls(mapping)


def save(mapping: dict[str, str], path: Path | None = None) -> Path:
    """Write an alias file, sorted so that diffs stay readable."""
    path = Path(path or aliases_path()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1].lower()))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _unfold(text: str) -> list[str]:
    """Join vCard continuation lines, which begin with a space or tab."""
    lines: list[str] = []
    for raw in text.splitlines():
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def parse_vcard(text: str) -> dict[str, str]:
    """Extract {handle: name} from vCard text.

    Deliberately minimal rather than a full RFC 6350 implementation: the only
    fields that matter here are the display name and the phone/email handles it
    belongs to.
    """
    contacts: dict[str, str] = {}
    name = ""
    structured = ""
    handles: list[str] = []

    def flush():
        display = name or structured
        if display:
            for handle in handles:
                contacts[handle] = display

    for line in _unfold(text):
        upper = line.upper()
        if upper.startswith("BEGIN:VCARD"):
            name, structured, handles = "", "", []
        elif upper.startswith("END:VCARD"):
            flush()
            name, structured, handles = "", "", []
        elif upper.startswith("FN"):
            name = line.split(":", 1)[-1].strip()
        elif upper.startswith("N:") or upper.startswith("N;"):
            parts = line.split(":", 1)[-1].split(";")
            given = parts[1].strip() if len(parts) > 1 else ""
            family = parts[0].strip() if parts else ""
            structured = " ".join(p for p in (given, family) if p)
        elif upper.startswith("TEL") or upper.startswith("EMAIL"):
            value = line.split(":", 1)[-1].strip()
            if value:
                handles.append(value)

    flush()  # tolerate a file whose final END:VCARD is missing
    return contacts


def read_vcard_file(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return parse_vcard(handle.read())


def addressbook_databases(path: Path) -> list[Path]:
    """Every AddressBook database to read, given the top-level one.

    When contacts sync from iCloud, the top-level `AddressBook-v22.abcddb` holds
    almost nothing and the real records live in per-account databases under
    `Sources/<uuid>/`. Reading only the top-level file finds zero phone numbers on
    most machines, which looks like a broken query rather than the wrong file.
    """
    databases = [path] if path.exists() else []
    sources = path.parent / "Sources"
    # An unreadable Sources directory is not fatal: the top-level file may still
    # hold usable records, and the caller reports what was actually found.
    with contextlib.suppress(OSError):
        databases.extend(sorted(sources.glob("*/AddressBook-v22.abcddb")))
    return databases


def _read_one_addressbook(db: sqlite3.Connection) -> dict[str, str]:
    contacts: dict[str, str] = {}
    query = """
        SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION, {column}
        FROM {table} v JOIN ZABCDRECORD r ON r.Z_PK = v.ZOWNER
        WHERE {column} IS NOT NULL
    """
    for table, column in (
        ("ZABCDPHONENUMBER", "v.ZFULLNUMBER"),
        ("ZABCDEMAILADDRESS", "v.ZADDRESS"),
    ):
        try:
            rows = db.execute(query.format(table=table, column=column)).fetchall()
        except sqlite3.Error:
            continue
        for first, last, org, value in rows:
            display = " ".join(p for p in (first, last) if p) or (org or "")
            if display and value:
                contacts[value] = display
    return contacts


def read_addressbook(path: Path | None = None) -> dict[str, str]:
    """Read names from macOS Contacts, across every account database.

    Raises PermissionError when macOS denies access, which is the normal outcome
    unless the app running this has been granted Contacts access. Point
    MSGSEARCH_ADDRESSBOOK at a copy to sidestep the permission entirely — but copy
    the whole AddressBook directory, not just the top-level file, or the per-account
    databases under Sources/ are left behind and almost nothing is found.
    """
    path = Path(path or addressbook_path()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No AddressBook database at {path}")

    databases = addressbook_databases(path)
    contacts: dict[str, str] = {}
    denied: sqlite3.Error | None = None

    for database in databases:
        try:
            db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            db.execute("SELECT 1 FROM ZABCDRECORD LIMIT 1")
        except sqlite3.Error as error:
            denied = error
            continue
        try:
            contacts.update(_read_one_addressbook(db))
        finally:
            db.close()

    if not contacts and denied is not None:
        raise PermissionError(
            "macOS denied access to Contacts. Grant Contacts access to the app "
            "running msgsearch under System Settings -> Privacy & Security -> "
            "Contacts, or copy the AddressBook directory from an app that has it "
            "and set MSGSEARCH_ADDRESSBOOK."
        ) from denied

    return contacts


def load_mapping(path: Path | None = None) -> dict[str, str]:
    """The alias file as a plain dict, for editing and merging."""
    path = Path(path or aliases_path()).expanduser()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def handle_volumes(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Every handle in the message database, busiest first.

    Ordering by volume is what makes naming tractable: a few dozen handles cover
    almost all of a typical archive, and the long tail rarely matters.
    """
    rows = conn.execute(
        """SELECT h.id, count(*) AS n
           FROM message m JOIN handle h ON h.ROWID = m.handle_id
           WHERE m.associated_message_type = 0 AND m.item_type = 0
           GROUP BY h.id ORDER BY n DESC"""
    ).fetchall()
    return [(handle, count) for handle, count in rows if handle]
