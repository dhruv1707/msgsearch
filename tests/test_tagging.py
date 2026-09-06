"""Tests for shape tagging.

Every fixture here is invented. Nothing in this file may come from a real
message, because this file is published.
"""

import unittest

from msgsearch import tagging


class TestBareUrl(unittest.TestCase):
    def test_single_link_is_bare(self):
        self.assertTrue(tagging.is_bare_url("https://example.com/thing/12345"))

    def test_several_links_are_bare(self):
        self.assertTrue(
            tagging.is_bare_url("https://example.com/a https://example.com/b")
        )

    def test_link_with_a_comment_is_not_bare(self):
        self.assertFalse(tagging.is_bare_url("look at this https://example.com/a"))

    def test_plain_text_is_not_bare(self):
        self.assertFalse(tagging.is_bare_url("see you at six"))

    def test_empty_is_not_bare(self):
        self.assertFalse(tagging.is_bare_url("   "))


class TestTags(unittest.TestCase):
    def test_plain_message_has_no_tags(self):
        self.assertEqual(tagging.tags("running ten minutes late"), frozenset())

    def test_email_detected(self):
        self.assertIn("email", tagging.tags("write to sam@example.com about it"))

    def test_url_detected(self):
        self.assertIn("url", tagging.tags("https://example.com/report"))

    def test_labelled_secret_is_a_credential(self):
        self.assertIn("credential", tagging.tags("Pword: Frog7Basket"))

    def test_bare_credential_detected_without_the_word(self):
        # The case the whole feature exists for: an address next to a
        # password-shaped token, and no giveaway vocabulary anywhere.
        self.assertIn("credential", tagging.tags("sam@example.com  Frog7Basket"))

    def test_email_alone_is_not_a_credential(self):
        self.assertNotIn("credential", tagging.tags("forward it to sam@example.com"))

    def test_merely_discussing_a_login_is_not_a_credential(self):
        # These are the false positives that made --type credential useless:
        # the subject comes up constantly, the secret almost never does.
        for text in (
            "did you get the atria login",
            "let me find you the login",
            "the password is on the fridge",
            "Shane changed the login?",
        ):
            with self.subTest(text=text):
                found = tagging.tags(text)
                self.assertNotIn("credential", found)
                self.assertIn("credential_talk", found)

    def test_address_detected(self):
        self.assertIn("address", tagging.tags("meet me at 240 Bramble Road"))


class TestPasswordShape(unittest.TestCase):
    def test_mixed_letters_and_digits(self):
        self.assertTrue(tagging.has_password_shape("Frog7Basket"))

    def test_plain_words_are_not_passwords(self):
        self.assertFalse(tagging.has_password_shape("see you tomorrow morning"))

    def test_bare_numbers_are_not_passwords(self):
        self.assertFalse(tagging.has_password_shape("12345678"))

    def test_urls_are_not_passwords(self):
        self.assertFalse(tagging.has_password_shape("https://example.com/a1b2c3d4"))


if __name__ == "__main__":
    unittest.main()
