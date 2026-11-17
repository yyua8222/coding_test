"""Validation utilities for output checking only.

These helpers are intentionally NOT importable as a "library to constrain
generation". They exist to validate whatever string the candidate's
solution returns.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

# A "word" is a maximal letter-run optionally containing internal '-' or '\''.
# Examples: hello, long-term, she's
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")

# Punctuation/whitespace explicitly allowed by the problem statement.
_ALLOWED_PUNCT = set(string.punctuation)  # ASCII punctuation
_ALLOWED_WS = set(" \t\r\n")


def load_vocab(path: str | Path) -> set[str]:
    """Load wordlist as a set of single lowercase words.

    Multi-word entries (e.g., "according to", "all right", "no one") are
    split on whitespace so that each component word is independently allowed.
    Hyphenated entries (e.g., "long-term", "old-fashioned") are kept whole.
    Apostrophe forms are excluded so contractions cannot pass validation.
    """
    p = Path(path)
    raw_lines = [ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    vocab: set[str] = set()
    for entry in raw_lines:
        vocab.update(part for part in entry.split() if "'" not in part)
    return vocab


def is_allowed_char(ch: str) -> bool:
    """ASCII letters, digits, common punctuation, whitespace are allowed."""
    if ch in _ALLOWED_WS:
        return True
    if not ch.isascii():
        return False
    if ch.isalnum():
        return True
    if ch in _ALLOWED_PUNCT:
        return True
    return False


def find_forbidden_chars(text: str) -> list[tuple[int, str]]:
    """Return (index, char) pairs for any disallowed character."""
    return [(i, c) for i, c in enumerate(text) if not is_allowed_char(c)]


def find_out_of_vocab_words(text: str, vocab: set[str]) -> list[str]:
    """Return list of words present in `text` but absent from `vocab` (lowercased)."""
    bad: list[str] = []
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w not in vocab:
            bad.append(w)
    return bad


def count_words(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))


def validate_output(text: str, vocab: set[str]) -> bool:
    """Return whether text obeys both the character and vocabulary policies."""
    return not find_forbidden_chars(text) and not find_out_of_vocab_words(text, vocab)
