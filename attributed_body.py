"""Decode the `message.attributedBody` blob (Apple NSArchiver 'typedstream').

~86% of rows in chat.db have text IS NULL and carry their content here instead.

Layout, per hexdump of a real row:

    04 0b "streamtyped" 81 e8 03 ... "NSString" 01 94 84 01 2b <len> <utf8 bytes>
                                                          '+'  ^ varint length

The 0x2b ('+') marker introduces a length-prefixed byte string. Length is one
byte when < 0x80, otherwise 0x81 => uint16 LE, 0x82 => uint32 LE.
"""

import struct

_MARKER = b"NSString"


def _read_len(buf, i):
    """Return (length, next_index) for the varint length at buf[i]."""
    n = buf[i]
    if n < 0x80:
        return n, i + 1
    if n == 0x81:
        return struct.unpack_from("<H", buf, i + 1)[0], i + 3
    if n == 0x82:
        return struct.unpack_from("<I", buf, i + 1)[0], i + 5
    return None, i + 1


def decode(blob):
    """Extract the message text from an attributedBody blob, or None."""
    if not blob:
        return None
    buf = bytes(blob)

    start = buf.find(_MARKER)
    if start == -1:
        return None
    i = buf.find(b"\x2b", start + len(_MARKER))
    if i == -1:
        return None

    length, i = _read_len(buf, i + 1)
    if not length or i + length > len(buf):
        return None

    raw = buf[i : i + length]
    # Runs of the string are UTF-8; a leading BOM-ish 0xff 0xfe marks UTF-16.
    if raw[:2] == b"\xff\xfe":
        return raw[2:].decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def message_text(text, blob):
    """Prefer the plain `text` column, fall back to decoding the blob."""
    if text is not None:
        return text
    return decode(blob)
