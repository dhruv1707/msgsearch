"""Tests for query construction.

No index or model is touched here; these check the FTS5 expression only.
"""

import unittest

from msgsearch.search import _fts_expression


class TestFtsExpression(unittest.TestCase):
    def test_empty_query(self):
        self.assertEqual(_fts_expression(""), "")
        self.assertEqual(_fts_expression("!!!"), "")

    def test_single_term_needs_no_layers(self):
        self.assertEqual(_fts_expression("atria"), '"atria"')

    def test_multi_term_prefers_the_phrase_but_still_allows_any(self):
        expression = _fts_expression("atria login")
        # The phrase ranks best, all-terms next, any-term last. All three are
        # present so nothing OR would have matched is lost.
        self.assertIn('"atria login"', expression)
        self.assertIn('"atria" AND "login"', expression)
        self.assertIn('"atria" OR "login"', expression)

    def test_punctuation_is_stripped_not_passed_through(self):
        # A stray quote or hyphen is an FTS5 operator and would error out.
        expression = _fts_expression('order #A-1234 "x"')
        self.assertNotIn("#", expression)
        self.assertNotIn("-", expression)
        self.assertIn('"order"', expression)

    def test_case_is_normalised(self):
        self.assertEqual(_fts_expression("Atria"), _fts_expression("atria"))


if __name__ == "__main__":
    unittest.main()
