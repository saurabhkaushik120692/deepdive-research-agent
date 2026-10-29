"""The claim-number check: every figure and year in a claim must appear in the source it cites."""

from app.graph.nodes.verifier import verify_claim
from app.graph.schemas import VerdictOutput
from app.tools.quote_match import missing_numbers

IEA = "In 2025, global EV battery demand rose 28% to 1,100 GWh, according to the IEA."


def test_changed_number_is_missing():
    assert missing_numbers("EV battery demand rose 32% in 2025.", IEA) == ["32"]


def test_shifted_year_is_missing():
    assert missing_numbers("EV battery demand rose 28% in 2024.", IEA) == ["2024"]


def test_claim_with_the_sources_figures_passes():
    assert missing_numbers("In 2025, EV battery demand rose 28% to 1,100 GWh.", IEA) == []


def test_number_formatting_differences_pass():
    source = "Demand reached 1100 GWh; the rate was cut to 6% and later 6.50%."
    claim = "Demand reached 1,100 GWh; the rate was cut to 6.0% and later 6.5%."
    assert missing_numbers(claim, source) == []


def test_numbers_inside_codes_are_skipped():
    source = "The RBI projects 6.8% growth for 2026-27; CO emissions and lithium oxide grades vary."
    claim = "The RBI projects 6.8% growth for FY26-27; CO2 emissions and Li2O grades vary in Q1."
    assert missing_numbers(claim, source) == []


def test_units_attached_to_a_number_still_count():
    assert missing_numbers(
        "The repo rate was cut by 50bps.", "The repo rate was cut by 25 bps."
    ) == ["50"]


def test_each_missing_number_reported_once():
    claim = "Prices rose 12% in 2026, then 12% again."
    assert missing_numbers(claim, "Prices rose 10% in 2026.") == ["12"]


async def test_verifier_approval_overruled_when_a_number_is_missing(deps):
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="supported", quote=IEA, reason="same trend"
    )
    out, found = await verify_claim(deps, None, "EV battery demand rose 32% in 2025.", IEA)
    assert out.label == "unsupported" and found is True
    assert "32" in out.reason
