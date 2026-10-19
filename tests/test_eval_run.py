import json
import random

from app.graph.schemas import JudgeOutput
from app.models import Claim, EvidenceChunk, Run, Source, SubQuestion, Verdict
from eval.label import sample_for_labelling, write_gold
from eval.run import evaluate, render_results_md, research_questions
from tests.scripts import script_happy_path


async def test_evaluate_end_to_end_with_scripted_agents(deps, tmp_path):
    script_happy_path(deps, thin_in_round0={"B"})
    deps.llm.handlers["judge"] = lambda s, u: JudgeOutput(supported=True, reason="ok")
    run_ids = await research_questions(deps, [{"id": "q01", "question": "EV battery risks?"}])
    results = await evaluate(
        deps,
        run_ids,
        random.Random(1),
        kinds=("number", "year", "negation"),
        gold_path=tmp_path / "none.jsonl",
    )
    assert results["runs"] == {"total": 1, "completed": 1, "failed": 0, "failed_ids": []}
    assert results["claims"]["drafted"] == 8 and results["claims"]["unsupported"] == 1
    assert results["loops"]["runs_with_loop_rate"] == 1.0
    probe = results["verifier_probe"]
    assert probe["n_originals"] == 7 and set(probe["catch_rate_by_kind"]) == {"number"}
    assert results["independent_judge"]["n"] > 0
    assert results["independent_judge"]["groundedness"] == 1.0
    assert results["gold"]["n_labelled"] == 0
    md = render_results_md(results)
    assert "Verifier catch rate" in md and "How to reproduce" in md and "n/a" in md


async def test_failed_runs_are_counted_not_hidden(deps, tmp_path):
    deps.llm.handlers["planner"] = lambda s, u: RuntimeError("mesh down")
    run_ids = await research_questions(deps, [{"id": "q01", "question": "EV battery risks?"}])
    results = await evaluate(deps, run_ids, random.Random(1), kinds=(), gold_path=tmp_path / "x")
    assert results["runs"]["failed"] == 1 and results["runs"]["failed_ids"] == run_ids


def _seed_labelled_run(deps):
    with deps.session_factory() as s:
        run = Run(question="q", status="completed", models={})
        s.add(run)
        s.commit()
        s.refresh(run)
        sq = SubQuestion(run_id=run.id, text="t", query="q")
        s.add(sq)
        s.commit()
        s.refresh(sq)
        src = Source(run_id=run.id, url="https://a.org", domain="a.org")
        s.add(src)
        s.commit()
        s.refresh(src)
        ch = EvidenceChunk(source_id=src.id, run_id=run.id, sub_question_id=sq.id, text="SRC")
        s.add(ch)
        s.commit()
        s.refresh(ch)
        labels = ["supported"] * 3 + ["partial"] + ["unsupported"] * 2
        for i, label in enumerate(labels):
            c = Claim(run_id=run.id, sub_question_id=sq.id, text=f"c{i}", chunk_ids=[ch.id])
            s.add(c)
            s.commit()
            s.refresh(c)
            s.add(Verdict(claim_id=c.id, label=label))
        s.commit()
        return run.id


def test_labelling_sample_is_stratified_and_preserves_labels(deps, tmp_path):
    run_id = _seed_labelled_run(deps)
    rows = sample_for_labelling(deps, [run_id], 4, random.Random(3))
    assert len(rows) == 4 and all(r["source_text"] == "SRC" for r in rows)
    assert {r["verifier_label"] for r in rows} == {"supported", "partial", "unsupported"}
    gold = tmp_path / "gold.jsonl"
    rows[0]["gold_supported"] = True
    write_gold(gold, rows)
    again = sample_for_labelling(deps, [run_id], 4, random.Random(3))
    write_gold(gold, again)
    saved = {json.loads(line)["claim_id"]: json.loads(line) for line in gold.open()}
    assert saved[rows[0]["claim_id"]]["gold_supported"] is True
