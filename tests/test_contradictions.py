import re

from app.graph.runner import create_run, run_research
from app.graph.schemas import Conflict, ContradictionOutput
from app.llm import LlmOutputError
from tests.helpers import event_types, get_run, report_of
from tests.scripts import script_happy_path


def _conflicts_in(letter=None, foreign=False):
    def handler(system, user):
        ids = [int(x) for x in re.findall(r"\[claim:(\d+)\]", user)]
        if foreign:
            return ContradictionOutput(
                conflicts=[Conflict(claim_a=9999, claim_b=ids[0], explanation="x")]
            )
        if letter and f"Sub-question {letter}" in user:
            return ContradictionOutput(
                conflicts=[
                    Conflict(claim_a=ids[0], claim_b=ids[1], explanation="different figures")
                ]
            )
        return ContradictionOutput(conflicts=[])

    return handler


async def test_conflict_flagged_in_report(deps):
    script_happy_path(deps)
    deps.llm.handlers["contradictions"] = _conflicts_in("A")
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    md = report_of(deps, run.id).markdown
    assert "## Conflicting sources" in md and "different figures" in md
    assert "conflict_found" in event_types(deps, run.id)


async def test_conflict_with_foreign_ids_ignored(deps):
    script_happy_path(deps)
    deps.llm.handlers["contradictions"] = _conflicts_in(foreign=True)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert "## Conflicting sources" not in report_of(deps, run.id).markdown


async def test_contradiction_check_failure_does_not_fail_run(deps):
    script_happy_path(deps)
    deps.llm.handlers["contradictions"] = lambda s, u: LlmOutputError("bad json")
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    assert "contradiction_check_failed" in event_types(deps, run.id)


async def test_only_subquestions_with_two_supported_claims_are_checked(deps):
    script_happy_path(deps, thin_in_round0={"B"}, always_thin={"B"})
    deps.llm.handlers["contradictions"] = _conflicts_in()
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    checked = [u for r, u in deps.llm.calls if r == "contradictions"]
    assert len(checked) == 2 and not any("Sub-question B" in u for u in checked)
