import re

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
# Some pages store curly quotes and dashes as control characters (U+2019 saved as 0x19), which
# showed up in reports as "RBIs". Those become punctuation; other control characters become
# spaces. Tabs and newlines stay.
_PUNCT = {0x13: "–", 0x14: "—", 0x18: "‘", 0x19: "’", 0x1C: "“", 0x1D: "”"}
_CONTROLS = {
    c: " " for c in [*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), *range(0x7F, 0xA0)]
} | _PUNCT
_WORD = re.compile(r"[A-Za-z]{3,}")
_VOWEL = re.compile(r"[aeiouAEIOU]")


def strip_controls(text: str) -> str:
    return text.translate(_CONTROLS)


def clean_text(text: str) -> str:
    """Drop markdown images, keep link text without the URL, remove bare URLs.

    Search providers return page text as markdown; URLs carry no evidence but cost tokens and
    push chunks past the embedding model's input limit.
    """
    text = strip_controls(text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    return " ".join(text.split())


def chunk_text(
    text: str, max_words: int = 600, overlap_words: int = 75, max_chars: int = 3000
) -> list[str]:
    """Split text into overlapping word windows (~600 words, about 800 tokens), never longer than
    max_chars, so every chunk fits the embedding model's input limit."""
    words = []
    for w in text.split():  # a single "word" longer than max_chars is hard-split
        words += [w[i : i + max_chars] for i in range(0, len(w), max_chars)]
    if not words:
        return []
    chunks = []
    start = 0
    while True:
        end, size = start, 0
        while end < len(words) and end - start < max_words:
            extra = len(words[end]) + (1 if end > start else 0)
            if end > start and size + extra > max_chars:
                break
            size += extra
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            return chunks
        overlap = min(overlap_words, (end - start) // 4)  # char-capped windows can be short
        start = max(start + 1, end - overlap)


def is_noise(
    text: str,
    min_letter_share: float = 0.5,
    max_rare_letter_share: float = 0.07,
    max_vowelless_word_share: float = 0.35,
) -> bool:
    """Chunks that carry no quotable evidence but cost tokens: mostly digits/symbols (number
    tables, leader dots, SVG/markup), or the bytes of a PDF read as text. Prose has about 0.5%
    j/q/x/z and few words without a vowel; PDF bytes have far more of both."""
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return True
    if sum(c.isalpha() for c in visible) / len(visible) < min_letter_share:
        return True
    letters = [c.lower() for c in visible if c.isascii() and c.isalpha()]
    if (
        len(letters) >= 200
        and sum(c in "jqxz" for c in letters) / len(letters) > max_rare_letter_share
    ):
        return True
    words = _WORD.findall(text)
    vowelless = sum(not _VOWEL.search(w) for w in words)
    return len(words) >= 20 and vowelless / len(words) > max_vowelless_word_share
