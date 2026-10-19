import httpx
import pytest

from app.tools.fetch import HttpFetcher
from app.tools.search import TavilySearch


class FakeTavily:
    def __init__(self, results):
        self.results = results
        self.calls = 0
        self.last_kwargs = {}

    async def search(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return {"results": self.results.get(kwargs["query"], [])}


def mock_http(status, body, ctype):
    def handler(request):
        return httpx.Response(status, text=body, headers={"content-type": ctype})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def mock_http_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


ARTICLE = (
    "<html><head><title>EV batteries</title></head><body><article>"
    + "".join(
        f"<p>Paragraph {i}: battery demand rose sharply this year in market segment {i}, "
        f"according to analysts tracking cell production and pricing.</p>"
        for i in range(12)
    )
    + "</article></body></html>"
)


async def test_search_is_cached(settings, session_factory):
    fake = FakeTavily({"q": [{"url": "https://a.com", "title": "A", "raw_content": "text"}]})
    s = TavilySearch(settings, session_factory, client=fake)
    r1 = await s.search("q", 5)
    r2 = await s.search("q", 5)
    assert r1 == r2 and r1[0].url == "https://a.com" and r1[0].content == "text"
    assert fake.calls == 1
    assert fake.last_kwargs["include_answer"] is False
    assert fake.last_kwargs["include_raw_content"] is True


async def test_search_missing_raw_content_is_none(settings, session_factory):
    fake = FakeTavily({"q": [{"url": "https://a.com", "title": "A", "content": "snippet"}]})
    r = await TavilySearch(settings, session_factory, client=fake).search("q", 5)
    assert r[0].content is None


async def test_fetch_extracts_html(session_factory):
    f = HttpFetcher(session_factory, http=mock_http(200, ARTICLE, "text/html; charset=utf-8"))
    text = await f.fetch_text("https://a.com/x")
    assert text and "battery demand rose sharply" in text


@pytest.mark.parametrize("status,ctype", [(403, "text/html"), (200, "application/pdf")])
async def test_fetch_failures_return_none(session_factory, status, ctype):
    f = HttpFetcher(session_factory, http=mock_http(status, ARTICLE, ctype))
    assert await f.fetch_text("https://b.com") is None


async def test_fetch_timeout_returns_none(session_factory):
    f = HttpFetcher(session_factory, http=mock_http_timeout())
    assert await f.fetch_text("https://c.com") is None


async def test_fetch_short_page_returns_none(session_factory):
    html = "<html><body><p>Too short.</p></body></html>"
    assert (
        await HttpFetcher(session_factory, http=mock_http(200, html, "text/html")).fetch_text(
            "https://d.com"
        )
        is None
    )


async def test_fetch_result_is_cached(session_factory):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})

    f = HttpFetcher(session_factory, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await f.fetch_text("https://e.com")
    await f.fetch_text("https://e.com")
    assert len(calls) == 1
