from app.tools.quote_match import normalize, quote_in_chunk

CHUNK = "In 2025, global EV battery demand rose 28% to 1.1 TWh, according to the IEA."


def test_exact_quote():
    assert quote_in_chunk("global EV battery demand rose 28% to 1.1 TWh", CHUNK)


def test_case_and_whitespace():
    assert quote_in_chunk("Global  EV battery\ndemand ROSE 28%", CHUNK)


def test_curly_quotes_and_nbsp():
    chunk = "He said “demand rose 28%” in 2025."
    assert quote_in_chunk("demand rose 28%", chunk)
    assert quote_in_chunk('"demand rose 28%"', chunk)


def test_minor_formatting_passes_fuzzy():
    assert quote_in_chunk("global EV battery demand rose 28 % to 1.1 TWh", CHUNK)


def test_fabricated_quote_rejected():
    assert not quote_in_chunk("global EV battery demand fell 12% to 0.9 TWh", CHUNK)


def test_changed_number_in_long_quote_rejected():
    # one digit changed: fuzzy similarity alone would be very high, numbers must still match
    assert not quote_in_chunk("global EV battery demand rose 29% to 1.1 TWh", CHUNK)


def test_short_quote_requires_exact():
    assert not quote_in_chunk("rose 29%", CHUNK)


def test_empty_quote_rejected():
    assert not quote_in_chunk("", CHUNK)
    assert not quote_in_chunk('  ""  ', CHUNK)


def test_normalize():
    assert normalize("  “Hello   World!” ") == "hello world"


def test_dropped_punctuation_passes_via_fuzzy():
    quote = "demand rose 28% to 1.1 TWh according to the IEA"  # source has ", according"
    assert normalize(quote) not in normalize(CHUNK)
    assert quote_in_chunk(quote, CHUNK)
