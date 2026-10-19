import pytest

from app.tools.tiers import source_tier


@pytest.mark.parametrize(
    "url,tier",
    [
        ("https://www.iea.org/reports/global-ev-outlook-2025", 1),
        ("https://www.sec.gov/Archives/edgar/x.htm", 1),
        ("https://pib.gov.in/release", 1),
        ("https://mit.edu/news/a", 1),
        ("https://pubmed.ncbi.nlm.nih.gov/123", 1),
        ("https://www.reuters.com/business/autos/x", 2),
        ("https://economictimes.indiatimes.com/x", 2),
        ("https://medium.com/@someone/ev-take", 3),
        ("https://www.reddit.com/r/ev/x", 3),
        ("not a url", 3),
    ],
)
def test_source_tier(url, tier):
    assert source_tier(url) == tier
