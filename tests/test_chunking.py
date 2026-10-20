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


def test_chunks_respect_char_cap_even_with_long_words():
    text = " ".join(["https://example.com/" + "x" * 80] * 300)  # 300 very long "words"
    chunks = chunk_text(text, max_words=600, overlap_words=75, max_chars=3000)
    assert chunks and all(len(c) <= 3000 for c in chunks)
    assert chunks[-1].split()[-1] == text.split()[-1]


def test_single_giant_word_is_hard_split():
    chunks = chunk_text("a" * 7000, max_words=600, overlap_words=75, max_chars=3000)
    assert [len(c) for c in chunks] == [3000, 3000, 1000]


def test_clean_text_strips_markdown_links_and_urls():
    from app.tools.chunking import clean_text

    raw = (
        "Demand rose [28%](https://iea.org/x?y=1) per ![chart](https://i.png) "
        "the IEA https://iea.org/report ."
    )
    assert clean_text(raw) == "Demand rose 28% per the IEA ."
