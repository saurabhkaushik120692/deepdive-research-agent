from app.tools.chunking import chunk_text


def test_chunking_overlap_and_coverage():
    words = [f"w{i}" for i in range(1500)]
    chunks = chunk_text(" ".join(words), max_words=600, overlap_words=75)
    assert all(len(c.split()) <= 600 for c in chunks)
    assert chunks[1].split()[:75] == chunks[0].split()[-75:]
    assert chunks[-1].split()[-1] == "w1499"


def test_chunking_empty_and_short():
    assert chunk_text("") == []
    assert chunk_text("a b c") == ["a b c"]
