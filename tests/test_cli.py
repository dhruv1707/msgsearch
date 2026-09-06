"""Tests for the command line interface.

These check argument wiring only. Nothing here touches a database, an index or a
model, so they run in milliseconds and need no downloads.
"""

import contextlib
import io
import unittest

from msgsearch import cli


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()

    def test_search_collects_a_multi_word_query(self):
        args = self.parser.parse_args(["search", "atria", "login"])
        self.assertEqual(args.query, ["atria", "login"])

    def test_search_defaults(self):
        args = self.parser.parse_args(["search", "anything"])
        self.assertIsNone(args.chat)
        self.assertIsNone(args.type)
        self.assertFalse(args.no_dense)
        self.assertFalse(args.no_bm25)

    def test_reranking_is_off_unless_asked_for(self):
        # Reranking measurably hurts on the gold set, so the default is off and
        # --rerank is the opt-in. See config.RERANK_ENABLED.
        default = self.parser.parse_args(["search", "q"])
        opted_in = self.parser.parse_args(["search", "q", "--rerank"])
        self.assertTrue(default.no_rerank)
        self.assertFalse(opted_in.no_rerank)

    def test_from_is_stored_without_shadowing_the_keyword(self):
        args = self.parser.parse_args(["search", "q", "--from", "Sam"])
        self.assertEqual(args.from_, "Sam")

    def test_date_filters(self):
        args = self.parser.parse_args(
            ["search", "q", "--after", "2024-01-01", "--before", "2024-12-31"]
        )
        self.assertEqual(args.after, "2024-01-01")
        self.assertEqual(args.before, "2024-12-31")

    def test_index_accepts_a_chat_filter(self):
        args = self.parser.parse_args(["index", "--chat", "+15551234567"])
        self.assertEqual(args.chat, "+15551234567")

    def test_each_command_has_a_handler(self):
        for argv in (["explore"], ["index"], ["search", "q"]):
            with self.subTest(argv=argv):
                args = self.parser.parse_args(argv)
                self.assertTrue(callable(args.handler))


class TestEntryPoint(unittest.TestCase):
    def test_bare_invocation_prints_help_and_fails(self):
        # Exiting non-zero matters: a bare `msgsearch` in a script should not
        # look like success. stdout is captured so the help text does not end up
        # in the test report.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([])
        self.assertEqual(code, 1)
        self.assertIn("usage: msgsearch", out.getvalue())

    def test_unknown_command_is_rejected(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            cli.main(["nonsense"])


if __name__ == "__main__":
    unittest.main()
