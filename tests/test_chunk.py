"""Tests for conversation windowing.

Every fixture here is invented. Nothing in this file may come from a real
message, because this file is published.
"""

import unittest
from datetime import datetime, timedelta, timezone

from msgsearch import chunk
from msgsearch.extract import Message

BASE = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def only_window(messages, **kwargs):
    """The single window these messages produce, asserting there is exactly one."""
    result = list(chunk.windows(messages, **kwargs))
    assert len(result) == 1, f"expected exactly one window, got {len(result)}"
    return result[0]


def msg(rowid, minutes, speaker, text, chat_id=1):
    return Message(
        rowid=rowid,
        guid=f"guid-{rowid}",
        chat_id=chat_id,
        chat_label="Test Chat",
        chat_is_group=False,
        timestamp=BASE + timedelta(minutes=minutes),
        is_from_me=(speaker == "Me"),
        speaker=speaker,
        service="iMessage",
        text=text,
    )


class TestPauseSplitting(unittest.TestCase):
    def test_close_messages_form_one_window(self):
        msgs = [msg(1, 0, "Me", "hey"), msg(2, 10, "Sam", "hi"), msg(3, 20, "Me", "ok")]
        self.assertEqual(len(list(chunk.windows(msgs))), 1)

    def test_a_long_pause_starts_a_new_window(self):
        msgs = [
            msg(1, 0, "Me", "hey"),
            msg(2, 5, "Sam", "hi"),
            msg(3, 120, "Me", "still there?"),
        ]
        result = list(chunk.windows(msgs))
        self.assertEqual(len(result), 2)
        self.assertEqual([len(w.messages) for w in result], [2, 1])

    def test_a_pause_of_exactly_the_gap_does_not_split(self):
        msgs = [msg(1, 0, "Me", "hey"), msg(2, 30, "Sam", "hi")]
        self.assertEqual(len(list(chunk.windows(msgs))), 1)

    def test_separate_chats_never_share_a_window(self):
        msgs = [msg(1, 0, "Me", "hey", chat_id=1), msg(2, 1, "Me", "hey", chat_id=2)]
        result = list(chunk.windows(msgs))
        self.assertEqual(len(result), 2)


class TestOversizedSplitting(unittest.TestCase):
    def _long_conversation(self, count=40, size=200):
        # All within the gap, so only the size cap can split these.
        return [msg(i, i, "Me" if i % 2 else "Sam", "x" * size) for i in range(count)]

    def test_windows_stay_within_budget(self):
        budget = 100  # tokens
        result = list(chunk.windows(self._long_conversation(), token_budget=budget))
        limit = budget * chunk.config.CHARS_PER_TOKEN
        self.assertGreater(len(result), 1)
        for window in result:
            self.assertLessEqual(len(window.search_text), limit)

    def test_small_windows_are_left_alone(self):
        msgs = [msg(1, 0, "Me", "hey"), msg(2, 1, "Sam", "hi")]
        result = list(chunk.windows(msgs, token_budget=1500))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].messages), 2)

    def test_pieces_overlap(self):
        # Messages small relative to the budget, which is the ordinary case: a
        # 1500-token budget against messages of a few dozen characters.
        msgs = self._long_conversation(count=40, size=40)
        result = list(chunk.windows(msgs, token_budget=100, overlap=2))
        first = {m.rowid for m in result[0].messages}
        second = {m.rowid for m in result[1].messages}
        self.assertTrue(first & second, "consecutive pieces should share messages")

    def test_overlap_is_dropped_when_it_would_fill_the_next_piece(self):
        # If single messages are large compared to the budget, carrying two of
        # them forward would leave no room for new content, so the overlap is
        # abandoned rather than allowed to crowd out the rest of the window.
        msgs = self._long_conversation(count=6, size=200)
        result = list(chunk.windows(msgs, token_budget=100, overlap=2))
        first = {m.rowid for m in result[0].messages}
        second = {m.rowid for m in result[1].messages}
        self.assertFalse(first & second)

    def test_every_message_survives_a_split(self):
        msgs = self._long_conversation()
        result = list(chunk.windows(msgs, token_budget=100))
        kept = {m.rowid for w in result for m in w.messages}
        self.assertEqual(kept, {m.rowid for m in msgs})


class TestRendering(unittest.TestCase):
    def test_offsets_point_at_the_message_body(self):
        msgs = [msg(1, 0, "Me", "first thing"), msg(2, 1, "Sam", "second thing")]
        window = only_window(msgs)
        by_id = {m.rowid: m.text for m in msgs}
        for rowid, start, end in window.offsets:
            self.assertEqual(window.search_text[start:end], by_id[rowid])

    def test_both_speakers_appear_in_the_text(self):
        msgs = [msg(1, 0, "Me", "hey"), msg(2, 1, "Sam", "hi")]
        window = only_window(msgs)
        self.assertIn("Me: hey", window.search_text)
        self.assertIn("Sam: hi", window.search_text)

    def test_link_only_messages_are_dropped_from_embed_text_but_kept_elsewhere(self):
        msgs = [
            msg(1, 0, "Me", "have a look"),
            msg(2, 1, "Me", "https://example.com/report"),
        ]
        window = only_window(msgs)
        self.assertIn("example.com", window.search_text)
        self.assertNotIn("example.com", window.embed_text)
        self.assertIn("have a look", window.embed_text)
        self.assertEqual(len(window.messages), 2)

    def test_tags_are_the_union_over_messages(self):
        msgs = [
            msg(1, 0, "Sam", "here you go"),
            msg(2, 1, "Sam", "sam@example.com  Frog7Basket"),
        ]
        window = only_window(msgs)
        self.assertIn("credential", window.tags)
        self.assertIn("email", window.tags)

    def test_one_sided_detection(self):
        solo = only_window([msg(1, 0, "Me", "a"), msg(2, 1, "Me", "b")])
        both = only_window([msg(1, 0, "Me", "a"), msg(2, 1, "Sam", "b")])
        self.assertTrue(solo.is_one_sided)
        self.assertFalse(both.is_one_sided)


if __name__ == "__main__":
    unittest.main()
