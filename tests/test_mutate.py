import random
import re

from eval.mutate import RewriteOutput, entity_swap, mutate_claim
from tests.fakes import ScriptedLlm


def rng():
    return random.Random(7)


def test_number_changes_value_keeps_unit():
    out = mutate_claim("Demand rose 28% to 1.1 TWh in 2025.", "number", rng())
    assert out != "Demand rose 28% to 1.1 TWh in 2025." and "%" in out and "2025" in out
    assert "TWh" in out and re.search(r"\d+\.\d TWh", out)


def test_year_shifts_only_year():
    out = mutate_claim("Demand rose 28% in 2025.", "year", rng())
    assert "28%" in out and "2025" not in out and re.search(r"20(2[2-8])", out)


def test_negation_flips_direction():
    assert mutate_claim("Battery prices fell in 2024.", "negation", rng()) == (
        "Battery prices rose in 2024."
    )


def test_negation_inserts_and_removes_not():
    assert mutate_claim("Lithium is scarce.", "negation", rng()) == "Lithium is not scarce."
    assert mutate_claim("Lithium is not scarce.", "negation", rng()) == "Lithium is scarce."


def test_not_applicable_returns_none():
    assert mutate_claim("Battery makers face supply risk.", "number", rng()) is None
    assert mutate_claim("Battery makers face supply risk.", "year", rng()) is None
    assert mutate_claim("Battery makers face supply risk.", "negation", rng()) is None


async def test_entity_swap_uses_llm_and_rejects_identity():
    llm = ScriptedLlm({"planner": lambda s, u: RewriteOutput(text="BYD leads the market.")})
    assert await entity_swap("CATL leads the market.", llm) == "BYD leads the market."
    llm.handlers["planner"] = lambda s, u: RewriteOutput(text="CATL leads the market.")
    assert await entity_swap("CATL leads the market.", llm) is None
