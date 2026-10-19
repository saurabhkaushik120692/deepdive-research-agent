"""Deterministic check that a verifier's quote really appears in the cited chunk.

This is the guard against the verifier itself hallucinating support.
"""

import re
import unicodedata

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
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_EDGE_CHARS = " \"'.,;:!?()[]"
MIN_FUZZY_LENGTH = 20  # shorter quotes must match exactly


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES).lower()
    s = re.sub(r"(\d)\s+%", r"\1%", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(_EDGE_CHARS)


def quote_in_chunk(quote: str, chunk: str, threshold: int = 90) -> bool:
    q, c = normalize(quote), normalize(chunk)
    if not q:
        return False
    # Every number in the quote must exist in the source: a changed figure is the most
    # common way a "nearly identical" quote misrepresents the source.
    chunk_numbers = set(_NUMBER.findall(c))
    if any(n not in chunk_numbers for n in _NUMBER.findall(q)):
        return False
    if q in c:
        return True
    if len(q) < MIN_FUZZY_LENGTH:
        return False
    return fuzz.partial_ratio(q, c) >= threshold
