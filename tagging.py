"""Pattern tags for messages.

Some messages that people badly want to find have almost no semantic content. A
bare credential such as "someone@example.com  Hunter2Pass" is not *about*
anything, so an embedding model has nothing to grip. Tagging these by shape at
index time lets a query like "what is the login for X" combine meaning (near talk
of X) with form (looks like a credential).

Everything here works on plain strings and needs no database, so it is cheap to
test with synthetic input.
"""

import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
PHONE = re.compile(r"(?<!\w)\+?\d[\d\-\s().]{7,}\d(?!\w)")

# Words people use when handing over access.
CREDENTIAL_WORD = re.compile(
    r"\b(?:pass\s?word|passwd|pwd|pword|passcode|log\s?in|credentials?|"
    r"user\s?name|otp|2fa|pin\s?code)\b",
    re.I,
)

# Street addresses, deliberately conservative: a number followed by words and a
# street-type suffix. Loose patterns here produce far more noise than value.
ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z][\w'.-]*\s+){0,4}"
    r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|"
    r"ct|court|pl|place|ter|terrace|hwy|highway)\b\.?",
    re.I,
)


def _looks_like_password(token: str) -> bool:
    """True if a whitespace-delimited token has the shape of a password.

    Checked by shape rather than by regex soup: 6-32 characters, containing both
    a letter and a digit, and not something we can already identify as a URL,
    an email address or a plain number.
    """
    if not 6 <= len(token) <= 32:
        return False
    if not any(c.isalpha() for c in token) or not any(c.isdigit() for c in token):
        return False
    if EMAIL.fullmatch(token) or URL.match(token):
        return False
    if token.isdigit():
        return False
    # Long lowercase-and-digits runs are usually ids, hashes or tracking codes.
    if token.islower() and len(token) > 24:
        return False
    return True


def has_password_shape(text: str) -> bool:
    return any(_looks_like_password(tok) for tok in text.split())


def is_bare_url(text: str) -> bool:
    """True if the message is nothing but one or more links.

    These are common in high-volume threads and carry no meaning for an embedding
    model, so they are dropped from embedded text while remaining searchable by
    keyword and fully visible when displayed.
    """
    stripped = text.strip()
    if not stripped:
        return False
    remainder = URL.sub("", stripped).strip()
    return not remainder and bool(URL.search(stripped))


def tags(text: str) -> frozenset:
    """Return the set of shape tags describing this text."""
    found = set()
    if EMAIL.search(text):
        found.add("email")
    if URL.search(text):
        found.add("url")
    if PHONE.search(text):
        found.add("phone")
    if ADDRESS.search(text):
        found.add("address")

    # A credential is either named outright, or implied by an address-like
    # identifier sitting next to a password-shaped token.
    if CREDENTIAL_WORD.search(text):
        found.add("credential")
    elif ("email" in found or "phone" in found) and has_password_shape(text):
        found.add("credential")

    return frozenset(found)
