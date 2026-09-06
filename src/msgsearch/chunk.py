"""Group messages into conversation windows.

Embedding messages one at a time does not work for this problem. The thing you
want to find is often a reply whose text has nothing in common with your query:
someone asks for a login, and two minutes later a message arrives containing an
email address and a password but neither the word "login" nor the name of the
service. Searched alone, that message is unreachable. Searched as part of the
exchange it belongs to, it is easy to find.

So the unit of retrieval is a window: a run of messages in one conversation with
no long pause in it. Windows are cut where a real pause happened, and only split
further when they are too long for the embedding model to read.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Sequence

import config
import tagging
from extract import Message


@dataclass(frozen=True)
class Window:
    window_id: str
    chat_id: int
    chat_label: str
    start: datetime
    end: datetime
    messages: tuple[Message, ...]

    # What the embedding model reads. Link-only messages are left out because
    # a bare URL has no meaning for the model to work with.
    embed_text: str

    # What keyword search reads, and what is shown to the user. Nothing is left
    # out here, so a URL someone sent is still findable by typing part of it.
    search_text: str

    # (message rowid, start offset, end offset) into search_text, so a result can
    # highlight the message that matched while displaying its whole context.
    offsets: tuple[tuple[int, int, int], ...]

    speakers: tuple[str, ...]
    tags: frozenset

    @property
    def is_one_sided(self) -> bool:
        """True if only one person spoke, which is usually a link dump."""
        return len(set(self.speakers)) <= 1


def _line_cost(message: Message) -> int:
    """Characters a message contributes once rendered as `Speaker: text`."""
    return len(message.speaker) + 2 + len(message.text) + 1


def _rendered_cost(messages: Sequence[Message]) -> int:
    """Length of these messages once rendered, including the date headers.

    The headers have to be counted here. Leaving them out lets a window creep
    past the budget by a few characters per day it spans.
    """
    total = 0
    day = None
    for message in messages:
        this_day = message.timestamp.strftime("%Y-%m-%d")
        if this_day != day:
            total += len(f"[{this_day}]\n")
            day = this_day
        total += _line_cost(message)
    return total


def _render(messages: Sequence[Message]) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    """Render messages as dated, speaker-labelled lines.

    Returns the text and, for each message, where its body starts and ends inside
    that text.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    position = 0
    current_day = None

    for message in messages:
        day = message.timestamp.strftime("%Y-%m-%d")
        if day != current_day:
            header = f"[{day}]\n"
            parts.append(header)
            position += len(header)
            current_day = day

        prefix = f"{message.speaker}: "
        line = f"{prefix}{message.text}\n"
        body_start = position + len(prefix)
        parts.append(line)
        position += len(line)
        offsets.append((message.rowid, body_start, position - 1))

    return "".join(parts), tuple(offsets)


def _split_on_pauses(messages: Sequence[Message], gap_seconds: int) -> list[list[Message]]:
    """Cut a conversation wherever nobody spoke for longer than `gap_seconds`."""
    groups: list[list[Message]] = []
    current: list[Message] = [messages[0]]

    for previous, message in zip(messages, messages[1:]):
        if (message.timestamp - previous.timestamp).total_seconds() > gap_seconds:
            groups.append(current)
            current = []
        current.append(message)

    groups.append(current)
    return groups


def _split_oversized(
    messages: Sequence[Message], budget_chars: int, overlap: int
) -> list[list[Message]]:
    """Break a window that is too long for the embedding model to read.

    Only windows over the budget are touched, which on real data is around 1% of
    them. Each piece after the first repeats the last couple of messages of the
    previous piece, so an exchange sitting on a boundary stays intact in one of
    the two pieces.
    """
    if _rendered_cost(messages) <= budget_chars:
        return [list(messages)]

    pieces: list[list[Message]] = []
    current: list[Message] = []

    for message in messages:
        # Cost is recomputed against the real rendering rather than tracked
        # incrementally, because a date header appears only on the first message
        # of each day and that depends on where the piece boundaries fall. These
        # windows are small, so the repeated work is not worth optimising away.
        if current and _rendered_cost(current + [message]) > budget_chars:
            pieces.append(current)

            # Carry a little context forward, but never so much that the overlap
            # alone fills the next piece.
            tail = list(current[-overlap:]) if overlap else []
            if _rendered_cost(tail) > budget_chars // 2:
                tail = []
            current = tail

        current.append(message)

    if current:
        pieces.append(current)
    return pieces


def build_window(messages: Sequence[Message]) -> Window:
    search_text, offsets = _render(messages)

    # Link-only messages are dropped from the embedded text only. They stay in
    # search_text, in the offsets and in the displayed result.
    speaking = [m for m in messages if not tagging.is_bare_url(m.text)]
    embed_text, _ = _render(speaking) if speaking else ("", ())

    first = messages[0]
    return Window(
        window_id=f"{first.chat_id}:{first.rowid}",
        chat_id=first.chat_id,
        chat_label=first.chat_label,
        start=first.timestamp,
        end=messages[-1].timestamp,
        messages=tuple(messages),
        embed_text=embed_text,
        search_text=search_text,
        offsets=offsets,
        speakers=tuple(m.speaker for m in messages),
        tags=frozenset().union(*(tagging.tags(m.text) for m in messages)),
    )


@dataclass(frozen=True)
class Passage:
    """A small slice of a window, and the thing that actually gets embedded.

    A window is the right unit to show a person, because it carries the context
    that makes a result make sense. It is the wrong unit to embed, because a
    single vector for thirty messages on eight topics represents none of them
    well. Passages slide across the window in small overlapping steps so that
    every message sits near the middle of at least one of them, and each passage
    remembers which window it belongs to.
    """

    passage_id: str
    window_id: str
    chat_id: int
    text: str
    rowids: tuple[int, ...]


def passages(
    window: Window, size: int | None = None, stride: int | None = None
) -> list[Passage]:
    """Slice a window into overlapping passages for embedding."""
    size = size if size is not None else config.PASSAGE_MESSAGES
    stride = stride if stride is not None else config.PASSAGE_STRIDE

    # Link-only messages are dropped here for the same reason they are dropped
    # from embed_text: a bare URL gives the model nothing to work with.
    speaking = [m for m in window.messages if not tagging.is_bare_url(m.text)]
    if not speaking:
        return []

    if len(speaking) <= size:
        starts = [0]
    else:
        starts = list(range(0, len(speaking) - size + 1, stride))
        # Make sure the final messages are covered even when the stride does not
        # divide the window evenly.
        last = len(speaking) - size
        if starts[-1] != last:
            starts.append(last)

    out = []
    for start in starts:
        selected = speaking[start : start + size]
        text, _ = _render(selected)
        out.append(
            Passage(
                passage_id=f"{window.window_id}#{start}",
                window_id=window.window_id,
                chat_id=window.chat_id,
                text=text,
                rowids=tuple(m.rowid for m in selected),
            )
        )
    return out


def windows(
    messages: Iterable[Message],
    gap_seconds: int | None = None,
    token_budget: int | None = None,
    overlap: int | None = None,
) -> Iterator[Window]:
    """Group messages into conversation windows, oldest first."""
    gap_seconds = gap_seconds if gap_seconds is not None else config.WINDOW_GAP_SECONDS
    token_budget = token_budget if token_budget is not None else config.WINDOW_TOKEN_BUDGET
    overlap = overlap if overlap is not None else config.WINDOW_OVERLAP_MESSAGES
    budget_chars = token_budget * config.CHARS_PER_TOKEN

    by_chat: dict[int, list[Message]] = defaultdict(list)
    for message in messages:
        by_chat[message.chat_id].append(message)

    for chat_id in sorted(by_chat):
        conversation = sorted(by_chat[chat_id], key=lambda m: m.timestamp)
        for group in _split_on_pauses(conversation, gap_seconds):
            for piece in _split_oversized(group, budget_chars, overlap):
                if piece:
                    yield build_window(piece)
