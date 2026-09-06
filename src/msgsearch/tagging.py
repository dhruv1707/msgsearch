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

# Words people use when handing over access. On their own these only tell you the
# subject came up, not that a secret is present: "did you get the atria login" and
# "Pword: hunter2" both contain the vocabulary but only one is worth retrieving
# when you are trying to recover an actual password.
CREDENTIAL_WORD = re.compile(
    r"\b(?:pass\s?word|passwd|pwd|pword|passcode|log\s?in|credentials?|"
    r"user\s?name|otp|2fa|pin\s?code)\b",
    re.I,
)

# A label immediately followed by its value, which is how people actually send
# these: "Pword: hunter2", "password - abc123", "login: admin".
LABELLED_SECRET = re.compile(
    r"\b(?:pass\s?word|passwd|pwd|pword|passcode|pin|otp|user\s?name|login)\b"
    r"\s*[:=\-]\s*(\S{4,64})",
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


def has_secret_value(text: str) -> bool:
    """True if the text appears to contain a credential, not merely mention one.

    Two ways to qualify. Either a label is followed by its value ("Pword: ..."),
    or an identifier such as an email address sits next to a password-shaped
    token, which is how a bare credential arrives with no vocabulary at all.
    """
    match = LABELLED_SECRET.search(text)
    if match and not URL.match(match.group(1)):
        return True
    if (EMAIL.search(text) or PHONE.search(text)) and has_password_shape(text):
        return True
    return False


def tags(text: str) -> frozenset:
    """Return the set of shape tags describing this text.

    `credential` means a secret appears to be present. `credential_talk` means the
    subject came up without one. Keeping these apart matters: searching for a
    forgotten password otherwise returns mostly people discussing it.
    """
    found = set()
    if EMAIL.search(text):
        found.add("email")
    if URL.search(text):
        found.add("url")
    if PHONE.search(text):
        found.add("phone")
    if ADDRESS.search(text):
        found.add("address")

    if has_secret_value(text):
        found.add("credential")
    elif CREDENTIAL_WORD.search(text):
        found.add("credential_talk")

    return frozenset(found)
