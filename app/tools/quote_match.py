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


# --- names --------------------------------------------------------------------------------------
# The quote check proves the verifier's quote is in the source; it can't tell that the CLAIM
# names someone else ("BloombergNEF says..." approved with a quote saying "the IEA says...").
# So every name in a claim must also appear in the source text it cites.

_MONTHS = (
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec"
)
_DAYS = "monday tuesday wednesday thursday friday saturday sunday"
_COMMON = (
    "the a an in on at by for of and or but as to from with per according however this that "
    "these those it its their there here while after before during since when"
)
# Codes and words that are capitalized without being anyone's name: currencies, units, company
# suffixes, roman numerals (an earlier version flagged these in 2% of approved claims).
_CODES = (
    "usd eur inr gbp jpy cny rmb rs bn mn tn twh gwh mwh kwh gw mw kw "
    "limited ltd inc corp co llc plc gmbh ag sa "
    "ii iii iv v vi vii viii ix x xi xii"
)
_NOT_NAMES = set(f"{_MONTHS} {_DAYS} {_COMMON} {_CODES}".split())
_GLUE = {"of", "and", "the", "for", "de", "&"}  # may sit inside a multi-word name
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9.&'’\-]*")
_DOTTED = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]?\.?")
# Chinese/China, European/Europe, Indian/India, Bodies/Body, Trends/Trend
_STEM_SUFFIXES = (("ies", "y"), ("ese", ""), ("ean", ""), ("ian", ""), ("an", ""), ("s", ""))
_MAX_SPELLED = 5  # longest name an acronym is matched against ("Reserve Bank of India" = 3)


def _tokens(text: str) -> list[tuple[str, bool]]:
    """(word, possessive) pairs: possessives stripped, dotted acronyms joined, hyphens split.
    Words with digits (Q1, Li2O, G20) are left out: numbers have their own check."""
    out = []
    for raw in _TOKEN.findall(unicodedata.normalize("NFKC", text)):
        raw = raw.replace("’", "'")
        possessive = raw.endswith(("'s", "'"))
        raw = re.sub(r"'s?$", "", raw)
        if _DOTTED.fullmatch(raw):
            out.append((raw.replace(".", ""), possessive))
            continue
        parts = [p.strip(".'&") for p in raw.split("-")]
        out += [(p, possessive) for p in parts if p and not any(ch.isdigit() for ch in p)]
    return out


def _acronym(word: str) -> str | None:
    """The acronym a word stands for, lowercased: IEA, ETFs (plural), BoE (mixed case)."""
    if word.endswith("s") and word[:-1].isupper() and len(word) >= 3:
        word = word[:-1]
    letters = sum(ch.isupper() for ch in word)
    if word.isalpha() and len(word) >= 2 and (word.isupper() or (letters >= 2 and len(word) <= 4)):
        return word.lower()
    return None


def _capitalized(word: str) -> bool:
    return word[:1].isupper() and word.lower() not in _NOT_NAMES


def _spelled(words: list[str]) -> set[str]:
    """Acronyms that runs of words can spell, with and without small words: "Reserve Bank of
    India" gives rbi and rboi; "electric vehicles" gives ev."""
    out = set()
    for start in range(len(words)):
        if words[start] in _GLUE:
            continue
        for end in range(start + 2, min(start + _MAX_SPELLED + 2, len(words) + 1)):
            run = words[start:end]
            out.add("".join(w[0] for w in run))
            out.add("".join(w[0] for w in run if w not in _GLUE))
    return out


def _stem(word: str) -> str:
    for suffix, replacement in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + replacement
    return word


def _same_word(name: str, source_word: str) -> bool:
    """Forms of one word (China/Chinese), or the name abbreviating it (Fed/Federal)."""
    a, b = _stem(name), _stem(source_word)
    short, long = sorted((a, b), key=len)
    if len(short) >= 4 and long.startswith(short):
        return True
    return len(name) >= 3 and source_word.startswith(name)


def missing_names(claim: str, source: str) -> list[str]:
    """Names in the claim (capitalized words after the first, acronyms, and a first word that
    is clearly a name) that the source text doesn't contain, each once, in claim order."""
    src_words = [w.lower() for w, _ in _tokens(source)]
    src_set = set(src_words)
    joined = {"".join(src_words[i : i + n]) for n in (2, 3) for i in range(len(src_words))}
    src_spelled = _spelled(src_words)

    tokens = _tokens(claim)
    words = [w.lower() for w, _ in tokens]
    skip = set()  # claim words whose multi-word name the source writes as an acronym
    for start in range(len(words)):
        for end in range(start + 2, min(start + _MAX_SPELLED + 1, len(words) + 1)):
            run = [i for i in range(start, end) if words[i] not in _GLUE]
            caps = all(_capitalized(tokens[i][0]) for i in run)
            # An acronym inside the name keeps all its letters: "LG Energy Solution" -> lges.
            spelled = "".join(_acronym(tokens[i][0]) or words[i][0] for i in run)
            if caps and len(run) >= 2 and spelled in src_set:
                skip.update(run)

    missing: list[str] = []
    for i, (word, possessive) in enumerate(tokens):
        if i in skip or word.lower() in _NOT_NAMES:
            continue
        acronym = _acronym(word)
        is_name = acronym is not None or (
            _capitalized(word) and (i > 0 or possessive or not word[1:].islower())
        )
        if not is_name:
            continue
        w = word.lower()
        found = (
            w in src_set
            or w in joined
            or (acronym is not None and (acronym in src_set or acronym in src_spelled))
            or any(_same_word(w, s) for s in src_set)
        )
        if not found and word not in missing:
            missing.append(word)
    return missing
