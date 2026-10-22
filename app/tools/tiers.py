"""Source-quality tiers: 1 = official / academic, 2 = major news, 3 = everything else."""

from typing import Literal
from urllib.parse import urlparse

TIER_LABELS = {1: "official", 2: "major news", 3: "other"}

TIER1_SUFFIXES = (".gov", ".gov.in", ".edu", ".ac.uk", ".int", ".mil")
TIER1_DOMAINS = {
    "iea.org",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "who.int",
    "nature.com",
    "science.org",
    "arxiv.org",
    "sciencedirect.com",
    "ncbi.nlm.nih.gov",
    "rbi.org.in",
    "sebi.gov.in",
}
TIER2_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "economist.com",
    "wsj.com",
    "nytimes.com",
    "theguardian.com",
    "cnbc.com",
    "thehindu.com",
    "livemint.com",
    "economictimes.indiatimes.com",
}


def _matches(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def source_tier(url: str) -> Literal[1, 2, 3]:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return 3
    if host.endswith(TIER1_SUFFIXES) or _matches(host, TIER1_DOMAINS):
        return 1
    if _matches(host, TIER2_DOMAINS):
        return 2
    return 3
