"""Deterministic check that a verifier's quote really appears in the cited chunk.

This is the guard against the verifier itself hallucinating support.
"""

import re
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
    }
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_EDGE_CHARS = " \"'.,;:!?()[]"
MIN_FUZZY_LENGTH = 20  # shorter quotes must match exactly
_NEGATIONS = {"not", "no", "never", "nor", "none", "without", "cannot"}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES).lower()
    s = re.sub(r"(\d)\s+%", r"\1%", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(_EDGE_CHARS)


def _words(s: str) -> list[str]:
    """Punctuation-insensitive word list ("isn't" becomes "is not")."""
    return re.sub(r"[^a-z0-9%]+", " ", s.replace("n't", " not")).split()


def _negations(words: list[str]) -> int:
    return sum(w in _NEGATIONS for w in words)


def quote_in_chunk(quote: str, chunk: str, threshold: int = 90) -> bool:
    q, c = normalize(quote), normalize(chunk)
    if not q:
        return False
    # Every number in the quote must exist in the source: a changed figure is the most
    # common way a "nearly identical" quote misrepresents the source.
    chunk_numbers = set(NUMBER.findall(c))
    if any(n not in chunk_numbers for n in NUMBER.findall(q)):
        return False
    qw, cw = _words(q), _words(c)
    if not qw:
        return False
    q_loose, c_loose = " ".join(qw), " ".join(cw)
    if f" {q_loose} " in f" {c_loose} ":
        return True
    if len(q) < MIN_FUZZY_LENGTH:
        return False
    # Fuzzy fallback for small transcription slips, but it may not change the meaning:
    # every quoted word must occur in the matched source window, with the same negations.
    align = fuzz.partial_ratio_alignment(q_loose, c_loose)
    if align is None or align.score < threshold:
        return False
    start = c_loose.rfind(" ", 0, align.dest_start) + 1
    end = c_loose.find(" ", align.dest_end)
    window = c_loose[start : end if end != -1 else len(c_loose)].split()
    if Counter(qw) - Counter(window):
        return False
    return _negations(qw) == _negations(window)
