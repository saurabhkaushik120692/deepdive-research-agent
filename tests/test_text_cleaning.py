"""Control characters from badly encoded pages, and scrambled PDF text."""

import random
import string

from app.tools.chunking import clean_text, is_noise, strip_controls

# What a fetched PDF looks like when its bytes are read as text (120 such chunks in the eval runs).
_RNG = random.Random(0)
BINARY = "%PDF-1.7 endobj 33 0 obj <>stream " + "".join(
    _RNG.choice(string.ascii_letters + string.digits + string.punctuation + "  ")
    for _ in range(900)
)


def test_quote_and_dash_codes_become_punctuation():
    assert strip_controls("the RBI\x19s view") == "the RBI’s view"
    assert strip_controls("\x18good\x19 and \x1cquoted\x1d") == "‘good’ and “quoted”"
    assert strip_controls("2024\x132025 \x14 a pause") == "2024–2025 — a pause"


def test_other_control_characters_become_spaces():
    assert strip_controls("twice\x1b25 basis points\x07") == "twice 25 basis points "
    assert strip_controls("range of \x00a2\x7f") == "range of  a2 "
    assert strip_controls("line one\nline two\ttab") == "line one\nline two\ttab"


def test_clean_text_removes_control_characters():
    assert clean_text("the RBI\x19s cut\x1bof 25 bps\x7f") == "the RBI’s cut of 25 bps"


def test_raw_pdf_bytes_are_noise():
    assert is_noise(clean_text(BINARY))


def test_normal_prose_is_not_noise():
    prose = (
        "The Reserve Bank of India cut the repo rate by 25 basis points to 6.25% in February "
        "2025, its first cut in nearly five years, citing easing inflation and slower growth."
    )
    assert not is_noise(prose)


def test_rendered_pages_strip_control_characters_from_old_data():
    from app.api.pages import templates

    page = templates.env.from_string("<p>{{ text }}</p>").render(text="the RBI\x19s view\x07")
    assert page == "<p>the RBI’s view </p>"


def test_trusted_html_like_tojson_is_not_escaped_twice():
    from app.api.pages import templates

    source = "{% autoescape true %}{{ data | tojson }} {{ text }}{% endautoescape %}"
    page = templates.env.from_string(source).render(data={"a": "b"}, text="<b>")
    assert page == '{"a": "b"} &lt;b&gt;’'
