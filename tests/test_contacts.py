"""Tests for handle-to-name resolution.

Every fixture here is invented. Nothing in this file may come from a real
contact, because this file is published.
"""

import json
import tempfile
import unittest
from pathlib import Path

from msgsearch import contacts


class TestNormalise(unittest.TestCase):
    def test_formatting_does_not_matter(self):
        # The same person is written half a dozen ways across chat.db and
        # Contacts; all of them have to land on one key.
        forms = ["+1 (555) 123-4567", "+15551234567", "555-123-4567", "5551234567"]
        keys = {contacts.normalise(form) for form in forms}
        self.assertEqual(len(keys), 1, keys)

    def test_emails_are_lowercased(self):
        self.assertEqual(contacts.normalise("Sam@Example.COM"), "sam@example.com")

    def test_short_codes_are_kept_whole(self):
        self.assertEqual(contacts.normalise("262966"), "262966")

    def test_empty_input_is_survivable(self):
        self.assertEqual(contacts.normalise(""), "")


class TestContacts(unittest.TestCase):
    def test_lookup_ignores_formatting(self):
        lookup = contacts.Contacts({"+1 (555) 123-4567": "Sam"})
        self.assertEqual(lookup.name("5551234567"), "Sam")
        self.assertEqual(lookup.label("+15551234567"), "Sam")

    def test_unknown_handles_fall_back_to_themselves(self):
        lookup = contacts.Contacts({"+15551234567": "Sam"})
        self.assertIsNone(lookup.name("+15559999999"))
        self.assertEqual(lookup.label("+15559999999"), "+15559999999")

    def test_blank_names_are_ignored(self):
        # The template writes empty strings for the user to fill in; until they
        # do, the handle must be used rather than an empty label.
        lookup = contacts.Contacts({"+15551234567": ""})
        self.assertEqual(lookup.label("+15551234567"), "+15551234567")
        self.assertFalse(lookup)

    def test_empty_lookup_is_falsey(self):
        self.assertFalse(contacts.Contacts())


class TestVcard(unittest.TestCase):
    VCARD = """BEGIN:VCARD
VERSION:3.0
N:Rivera;Sam;;;
FN:Sam Rivera
TEL;type=CELL:+1 (555) 123-4567
EMAIL;type=INTERNET:sam@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
N:Okafor;Ada;;;
FN:Ada Okafor
TEL;type=HOME:+15559876543
END:VCARD
"""

    def test_reads_names_and_handles(self):
        parsed = contacts.parse_vcard(self.VCARD)
        lookup = contacts.Contacts(parsed)
        self.assertEqual(lookup.name("5551234567"), "Sam Rivera")
        self.assertEqual(lookup.name("sam@example.com"), "Sam Rivera")
        self.assertEqual(lookup.name("+1 555 987 6543"), "Ada Okafor")

    def test_falls_back_to_structured_name_without_fn(self):
        card = "BEGIN:VCARD\nN:Okafor;Ada;;;\nTEL:+15559876543\nEND:VCARD\n"
        lookup = contacts.Contacts(contacts.parse_vcard(card))
        self.assertEqual(lookup.name("5559876543"), "Ada Okafor")

    def test_handles_folded_lines(self):
        # vCard wraps long lines and continues them with a leading space.
        card = "BEGIN:VCARD\nFN:Sam\n Rivera\nTEL:+15551234567\nEND:VCARD\n"
        lookup = contacts.Contacts(contacts.parse_vcard(card))
        self.assertEqual(lookup.name("5551234567"), "SamRivera")

    def test_a_card_with_no_handles_is_skipped(self):
        card = "BEGIN:VCARD\nFN:Nobody\nEND:VCARD\n"
        self.assertEqual(contacts.parse_vcard(card), {})

    def test_empty_input(self):
        self.assertEqual(contacts.parse_vcard(""), {})


class TestMerging(unittest.TestCase):
    """Contacts supplies names; the alias file corrects them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "contacts.json"
        self.addCleanup(self._tmp.cleanup)

    def _with_fake_addressbook(self, entries):
        """Stand in for macOS Contacts so the merge itself is what gets tested."""
        original = contacts.read_addressbook
        contacts.read_addressbook = lambda *a, **k: dict(entries)
        self.addCleanup(setattr, contacts, "read_addressbook", original)

    def test_alias_file_overrides_contacts(self):
        self._with_fake_addressbook({"+15551234567": "Samuel Rivera"})
        contacts.save({"+15551234567": "Sam (work)"}, self.path)
        merged = contacts.Contacts.load(self.path)
        self.assertEqual(merged.name("5551234567"), "Sam (work)")

    def test_contacts_supplies_names_the_alias_file_lacks(self):
        self._with_fake_addressbook({"+15559876543": "Ada Okafor"})
        contacts.save({"+15551234567": "Sam"}, self.path)
        merged = contacts.Contacts.load(self.path)
        self.assertEqual(merged.name("5559876543"), "Ada Okafor")
        self.assertEqual(merged.name("5551234567"), "Sam")

    def test_blank_alias_entries_do_not_erase_a_name(self):
        # The template writes blanks for the user to fill in. A half-filled
        # template must not wipe out names that Contacts already supplied.
        self._with_fake_addressbook({"+15551234567": "Samuel Rivera"})
        contacts.save({"+15551234567": ""}, self.path)
        merged = contacts.Contacts.load(self.path)
        self.assertEqual(merged.name("5551234567"), "Samuel Rivera")

    def test_a_denied_permission_is_not_an_error(self):
        def denied(*args, **kwargs):
            raise PermissionError("nope")

        original = contacts.read_addressbook
        contacts.read_addressbook = denied
        self.addCleanup(setattr, contacts, "read_addressbook", original)

        contacts.save({"+15551234567": "Sam"}, self.path)
        self.assertEqual(contacts.Contacts.load(self.path).name("5551234567"), "Sam")

    def test_addressbook_can_be_skipped(self):
        # Hermetic tests, and the escape hatch for anyone who has not granted
        # the Contacts permission.
        contacts.save({"+15551234567": "Sam"}, self.path)
        lookup = contacts.Contacts.load(self.path, use_addressbook=False)
        self.assertEqual(lookup.name("5551234567"), "Sam")


class TestAliasFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "contacts.json"
        self.addCleanup(self._tmp.cleanup)

    def test_round_trip(self):
        contacts.save({"+15551234567": "Sam"}, self.path)
        self.assertEqual(
            contacts.Contacts.load(self.path, use_addressbook=False).name("5551234567"),
            "Sam",
        )

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(len(contacts.Contacts.load(self.path, use_addressbook=False)), 0)

    def test_corrupt_file_degrades_rather_than_crashing(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(len(contacts.Contacts.load(self.path, use_addressbook=False)), 0)

    def test_saved_file_is_sorted_by_name(self):
        contacts.save({"+15550000002": "Bea", "+15550000001": "Ada"}, self.path)
        with open(self.path) as handle:
            self.assertEqual(list(json.load(handle).values()), ["Ada", "Bea"])


if __name__ == "__main__":
    unittest.main()
