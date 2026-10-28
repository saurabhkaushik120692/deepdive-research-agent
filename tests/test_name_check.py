"""The name check: every name in a claim must appear in the source it cites."""

from app.graph.nodes.verifier import verify_claim
from app.graph.schemas import VerdictOutput
from app.tools.quote_match import missing_names

IEA = "In 2025, global EV battery demand rose 28% to 1.1 TWh, according to the IEA."


def test_swapped_organisation_is_missing():
    claim = "EV battery demand rose 28% in 2025, according to BloombergNEF."
    assert missing_names(claim, IEA) == ["BloombergNEF"]


def test_swapped_country_with_possessive_is_missing():
    source = "China's exports of EV batteries rose 20% in 2025."
    assert missing_names("India's exports of EV batteries rose 20% in 2025.", source) == ["India"]


def test_claim_using_the_sources_names_passes():
    assert missing_names("Global EV battery demand rose 28% in 2025, per the IEA.", IEA) == []


def test_first_word_and_months_are_not_names():
    source = "Demand for batteries rose sharply in Feb 2025."
    assert missing_names("Demand for batteries rose sharply in February 2025.", source) == []


def test_demonyms_match_their_country():
    source = "EV sales in China and India rose; Japan lagged behind Europe."
    claim = "Chinese and Indian EV sales rose while Japanese sales lagged European ones."
    assert missing_names(claim, source) == []


def test_acronym_matches_full_name_both_ways():
    full = "The Reserve Bank of India cut the repo rate by 25 basis points."
    short = "The RBI cut the repo rate by 25 basis points."
    assert missing_names("The RBI cut the repo rate by 25 basis points.", full) == []
    assert missing_names("The Reserve Bank of India cut the repo rate.", short) == []


def test_dotted_acronym_matches():
    source = "The U.S. economy grew 2.8% in 2024."
    assert missing_names("The US economy grew 2.8% in 2024.", source) == []


def test_country_prefix_lookalikes_are_still_caught():
    source = "Austria's grid added 2 GW of storage."
    assert missing_names("Australia's grid added 2 GW of storage.", source) == ["Australia"]


def test_each_missing_name_reported_once():
    claim = "Tesla and Tesla's rival BYD sold more cars than Ford."
    source = "Tesla sold more cars than Ford."
    assert missing_names(claim, source) == ["BYD"]


async def test_verifier_approval_overruled_when_a_name_is_missing(deps):
    claim = "EV battery demand rose 28% in 2025, according to BloombergNEF."
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="supported", quote=IEA, reason="same figure"
    )
    out, found = await verify_claim(deps, None, claim, IEA)
    assert out.label == "unsupported" and found is True
    assert "BloombergNEF" in out.reason


async def test_verifier_approval_kept_when_names_match(deps):
    claim = "Global EV battery demand rose 28% in 2025, according to the IEA."
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="supported", quote=IEA, reason="stated"
    )
    out, _ = await verify_claim(deps, None, claim, IEA)
    assert out.label == "supported"


# --- false alarms found by running the rule over 2,918 claims the verifier had approved --------


def test_currency_and_unit_codes_are_not_names():
    source = "The market was valued at US$ 85.9 billion; the deal was worth Rs 1.5 lakh crore."
    claim = "The market was valued at USD 85.9 Bn; the deal was worth Rs 1.5 lakh crore."
    assert missing_names(claim, source) == []


def test_acronym_spelled_out_in_lowercase():
    source = "Battery packs for electric vehicles; the monetary policy committee met in April."
    assert missing_names("Battery packs for EV use; the MPC met in April.", source) == []


def test_plural_and_mixed_case_acronyms():
    source = (
        "Inflows into exchange traded funds rose as the Bank of England and the Federal Reserve"
        " held."
    )
    claim = "Inflows into ETFs rose as the BoE and the Fed held."
    assert missing_names(claim, source) == []


def test_codes_with_digits_and_roman_numerals_are_not_names():
    source = "In the first quarter the grade was 1.24% lithium oxide, under the annex rules."
    claim = "In Q1 the grade was 1.24% Li2O under Annex VI and Annex VII."
    assert missing_names(claim, source) == []


def test_company_suffixes_and_joined_names():
    source = "Synthesia Ltd and Persistence Market Research were cited."
    claim = "Synthesia Limited and Persistencemarketresearch were cited."
    assert missing_names(claim, source) == []


def test_full_name_starting_with_an_acronym_matches_its_short_form():
    source = "LGES from South Korea has announced solid-state batteries after 2030."
    claim = "LG Energy Solution (LGES) has announced solid-state batteries after 2030."
    assert missing_names(claim, source) == []
