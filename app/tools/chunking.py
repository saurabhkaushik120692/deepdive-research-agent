import re

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Drop markdown images, keep link text without the URL, remove bare URLs.

    Search providers return page text as markdown; URLs carry no evidence but cost tokens and
    push chunks past the embedding model's input limit.
    """
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    return " ".join(text.split())


def chunk_text(
    text: str, max_words: int = 600, overlap_words: int = 75, max_chars: int = 3000
) -> list[str]:
    """Split text into overlapping word windows (~600 words ≈ 800 tokens), never longer than
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
        start = max(start + 1, end - overlap_words)
