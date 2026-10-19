def chunk_text(text: str, max_words: int = 600, overlap_words: int = 75) -> list[str]:
    """Split text into overlapping word windows (~600 words ≈ 800 tokens)."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap_words)
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            return chunks
        start += step
