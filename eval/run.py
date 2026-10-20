"""Self-evaluation.

    python -m eval.run [--limit N] [--skip-research --runs-file eval/results/runs.json]


1. research every question in eval/questions.jsonl (or reuse stored run ids)
2. run statistics: outcomes, claim labels, loops, coverage, quote validity, cost, latency
3. verifier probe: re-verify supported claims (false-alarm rate) and planted false claims
   (catch rate per alteration kind)
4. independent judge (third model family) re-checks claims cited in final reports
5. agreement with the owner's hand labels (eval/gold/claims_gold.jsonl), when filled in
6. write eval/results/<timestamp>.json and regenerate eval/RESULTS.md
"""

import argparse
import asyncio
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import select

from app.config import get_settings
from app.deps import Deps, build_deps
from app.graph.nodes.verifier import verify_claim
from app.graph.runner import MODEL_ROLES, create_run, run_research
from app.llm import LlmOutputError
from app.models import Event, Report, Run, SubQuestion, Verdict
from eval.common import load_claims
from eval.judge import judge_claim
from eval.label import GOLD_PATH
from eval.metrics import (
    agreement,
    catch_rate,
    quote_validity_rate,
    summarize,
    wrongful_rejection_rate,
)
from eval.mutate import KINDS, MutationKind, entity_swap, mutate_claim

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS = EVAL_DIR / "questions.jsonl"

# USD per 1M tokens (input, output), from the Mesh model list on 2026-10-18 (spec §4.5).
PRICES = {
    "openai/gpt-5-nano": (0.05, 0.40),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
}


def usage_cost(usages: list[dict]) -> float | None:
    total = 0.0
    for u in usages:
        price = PRICES.get(u.get("model", ""))
        if price is None:
            return None
        total += (
            u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[1]
        ) / 1e6
    return total


async def research_questions(deps: Deps, questions: list[dict], concurrency: int = 2) -> list[str]:
    slots = asyncio.Semaphore(concurrency)
    run_ids = []
    for q in questions:
        with deps.session_factory() as session:
            run_ids.append(create_run(session, q["question"], deps.settings).id)

    async def one(run_id: str) -> None:
        async with slots:
            await run_research(run_id, deps)

    await asyncio.gather(*(one(r) for r in run_ids))
    return run_ids


async def retry_failed(deps: Deps, stored: list[dict], questions: list[dict]) -> list[dict]:
    """Re-research only the questions whose stored run did not complete; keep the rest."""
    text = {q["id"]: q["question"] for q in questions}
    with deps.session_factory() as session:
        failed = [
            i for i, s in enumerate(stored) if session.get(Run, s["run_id"]).status != "completed"
        ]
    new_ids = await research_questions(
        deps, [{"question": text[stored[i]["question_id"]]} for i in failed]
    )
    out = [dict(s) for s in stored]
    for i, run_id in zip(failed, new_ids, strict=True):
        out[i]["run_id"] = run_id
    return out


def _seconds(start: datetime, end: datetime | None) -> float | None:
    if end is None:
        return None
    return (end.replace(tzinfo=None) - start.replace(tzinfo=None)).total_seconds()


def run_stats(deps: Deps, run_ids: list[str]) -> dict:
    with deps.session_factory() as session:
        runs = [session.get(Run, r) for r in run_ids]
        reports = [session.get(Report, r) for r in run_ids]
        sqs = session.exec(select(SubQuestion).where(SubQuestion.run_id.in_(run_ids))).all()
        events = session.exec(
            select(Event).where(
                Event.run_id.in_(run_ids), Event.type.in_(["llm_call", "paragraph_dropped"])
            )
        ).all()
    rows = load_claims(deps, run_ids)
    verdicts = [r.verdict for r in rows]
    labels = Counter(v.label for v in verdicts)
    failed = [r.id for r in runs if r.status != "completed"]
    metrics = [rep.metrics for rep in reports if rep]
    usage_by_run: dict[str, list[dict]] = {}
    for e in events:
        if e.type == "llm_call":
            usage_by_run.setdefault(e.run_id, []).append(e.payload)
    costs = [usage_cost(usage_by_run.get(r, [])) for r in run_ids]
    known_costs = [c for c in costs if c is not None]
    drafted = len(verdicts)
    return {
        "runs": {
            "total": len(runs),
            "completed": len(runs) - len(failed),
            "failed": len(failed),
            "failed_ids": failed,
        },
        "claims": {
            "drafted": drafted,
            "supported": labels["supported"],
            "partial": labels["partial"],
            "unsupported": labels["unsupported"],
            "draft_supported_rate": labels["supported"] / drafted if drafted else None,
            "draft_rejected_rate": labels["unsupported"] / drafted if drafted else None,
            "report_paragraphs_dropped": sum(e.type == "paragraph_dropped" for e in events),
        },
        "loops": {
            "runs_with_loop_rate": (
                sum(m.get("rounds", 0) > 0 for m in metrics) / len(metrics) if metrics else None
            ),
            "mean_extra_rounds": (
                sum(m.get("rounds", 0) for m in metrics) / len(metrics) if metrics else None
            ),
        },
        "coverage": {
            "sub_questions": len(sqs),
            "ok_rate": sum(q.status == "ok" for q in sqs) / len(sqs) if sqs else None,
        },
        "quote_validity_rate": quote_validity_rate(verdicts),
        "cost": {
            "per_run_usd": summarize(known_costs),
            "total_usd": sum(known_costs) if known_costs else None,
            "unpriced_runs": len(costs) - len(known_costs),
        },
        "latency_s": summarize(
            [
                s
                for r in runs
                if (s := _seconds(r.started_at or r.created_at, r.finished_at)) is not None
            ]
        ),
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in runs),
            "completion": sum(r.completion_tokens for r in runs),
        },
    }


async def _mutant(row, kinds: tuple[MutationKind, ...], start: int, deps, rng):
    """Try kinds in rotation starting at `start`; return (kind, text) for the first that applies."""
    for k in range(len(kinds)):
        kind = kinds[(start + k) % len(kinds)]
        if kind == "entity":
            try:
                text = await entity_swap(row.claim.text, deps.llm)
            except LlmOutputError:
                text = None
        else:
            text = mutate_claim(row.claim.text, kind, rng)
        if text and text != row.claim.text:
            return kind, text
    return None


async def verifier_probe(
    deps: Deps, run_ids: list[str], rng: random.Random, kinds=KINDS, max_claims: int = 60
) -> dict:
    rows = load_claims(deps, run_ids, labels=("supported",))
    rng.shuffle(rows)
    rows = rows[:max_claims]
    originals = await asyncio.gather(
        *(verify_claim(deps, None, r.claim.text, r.source_text) for r in rows)
    )
    mutants = []
    if kinds:
        for i, r in enumerate(rows):
            m = await _mutant(r, tuple(kinds), i, deps, rng)
            if m:
                mutants.append((m[0], m[1], r.source_text))
    verdicts = await asyncio.gather(*(verify_claim(deps, None, t, src) for _, t, src in mutants))
    by_kind: dict[str, list[str]] = {}
    for (kind, _, _), (out, _) in zip(mutants, verdicts, strict=True):
        by_kind.setdefault(kind, []).append(out.label)
    return {
        "n_originals": len(rows),
        "wrongful_rejection_rate": wrongful_rejection_rate([o.label for o, _ in originals]),
        "n_mutants": len(mutants),
        "catch_rate": catch_rate([out.label for out, _ in verdicts]),
        "catch_rate_by_kind": {k: catch_rate(v) for k, v in sorted(by_kind.items())},
        "n_mutants_by_kind": {k: len(v) for k, v in sorted(by_kind.items())},
        "examples": [
            {"kind": k, "mutant": t, "label": out.label}
            for (k, t, _), (out, _) in list(zip(mutants, verdicts, strict=True))[:5]
        ],
    }


async def judge_reports(deps: Deps, run_ids: list[str], rng: random.Random, max_claims=100) -> dict:
    with deps.session_factory() as session:
        cited: set[int] = set()
        for r in run_ids:
            rep = session.get(Report, r)
            for section in (rep.synthesis if rep else {}).get("sections", []):
                for p in section.get("paragraphs", []):
                    cited.update(p.get("claim_ids", []))
    rows = [r for r in load_claims(deps, run_ids) if r.claim.id in cited]
    rng.shuffle(rows)
    rows = rows[:max_claims]
    outs = await asyncio.gather(*(_judge_or_none(deps, r) for r in rows))
    judged = [o for o in outs if o is not None]
    return {
        "model": deps.settings.model_judge,
        "n": len(judged),
        "errors": len(outs) - len(judged),
        "groundedness": sum(o.supported for o in judged) / len(judged) if judged else None,
    }


async def _judge_or_none(deps: Deps, row):
    try:
        return await judge_claim(deps.llm, row.claim.text, row.source_text)
    except LlmOutputError:
        return None


def gold_stats(deps: Deps, gold_path: Path) -> dict:
    labelled = []
    if gold_path.exists():
        for line in gold_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("gold_supported"), bool):
                    labelled.append(row)
    with deps.session_factory() as session:
        pred, gold = [], []
        for row in labelled:
            v = session.exec(select(Verdict).where(Verdict.claim_id == row["claim_id"])).first()
            if v is not None:
                pred.append(v.label != "unsupported")
                gold.append(row["gold_supported"])
    accuracy, kappa = agreement(pred, gold)
    return {"n_labelled": len(gold), "accuracy": accuracy, "kappa": kappa}


async def evaluate(
    deps: Deps,
    run_ids: list[str],
    rng: random.Random,
    kinds=KINDS,
    gold_path: Path = GOLD_PATH,
    max_probe: int = 60,
    max_judge: int = 100,
) -> dict:
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "n_questions": len(run_ids),
        "models": {role: getattr(deps.settings, f"model_{role}") for role in MODEL_ROLES}
        | {"judge": deps.settings.model_judge},
        **run_stats(deps, run_ids),
        "verifier_probe": await verifier_probe(deps, run_ids, rng, kinds, max_probe),
        "independent_judge": await judge_reports(deps, run_ids, rng, max_judge),
        "gold": gold_stats(deps, gold_path),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x, digits=2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_results_md(r: dict) -> str:
    p, j, g, c = r["verifier_probe"], r["independent_judge"], r["gold"], r["claims"]
    cost = r["cost"]["per_run_usd"]
    rows = [
        ("Runs completed", f"{r['runs']['completed']} / {r['runs']['total']}"),
        (
            "Verifier catch rate (planted false claims)",
            f"{_pct(p['catch_rate'])} of {p['n_mutants']}",
        ),
        *[
            (f"  · catch rate: {k}", f"{_pct(v)} of {p['n_mutants_by_kind'][k]}")
            for k, v in p["catch_rate_by_kind"].items()
        ],
        (
            "Wrongful rejection rate (re-verified supported claims)",
            f"{_pct(p['wrongful_rejection_rate'])} of {p['n_originals']}",
        ),
        (
            "Independent groundedness (judge: " + j["model"] + ")",
            f"{_pct(j['groundedness'])} of {j['n']}",
        ),
        ("Agreement with hand labels (accuracy)", f"{_pct(g['accuracy'])} of {g['n_labelled']}"),
        ("Agreement with hand labels (Cohen's kappa)", _num(g["kappa"])),
        ("Quote validity rate", _pct(r["quote_validity_rate"])),
        (
            "Drafted claims supported / partial / rejected",
            f"{c['supported']} / {c['partial']} / {c['unsupported']} of {c['drafted']}",
        ),
        ("Draft claims rejected by verifier", _pct(c["draft_rejected_rate"])),
        (
            "Report paragraphs dropped for citing unverified claims",
            str(c["report_paragraphs_dropped"]),
        ),
        ("Runs that looped back to research", _pct(r["loops"]["runs_with_loop_rate"])),
        ("Mean extra research rounds", _num(r["loops"]["mean_extra_rounds"])),
        (
            "Sub-questions with enough verified support",
            f"{_pct(r['coverage']['ok_rate'])} of {r['coverage']['sub_questions']}",
        ),
        (
            "Cost per run (mean / p95, USD)",
            f"{_num(cost['mean'], 4)} / {_num(cost['p95'], 4)}",
        ),
        (
            "Latency per run (p50 / p95, s)",
            f"{_num(r['latency_s']['p50'], 0)} / {_num(r['latency_s']['p95'], 0)}",
        ),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    models = "\n".join(f"- {role}: `{m}`" for role, m in r["models"].items())
    failed = ", ".join(r["runs"]["failed_ids"]) or "none"
    return f"""# DeepDive self-evaluation results

Generated {r["generated_at"]} on {r["n_questions"]} questions from `eval/questions.jsonl`.

| Metric | Value |
|---|---|
{table}

## What these numbers mean

- **Catch rate**: we take claims the verifier supported, alter one fact (a number, a year, the
  direction of a change, or the named entity) and verify the altered claim against the same
  source. A good verifier rejects them.
- **Wrongful rejection rate**: the same supported claims verified again, unaltered. A verifier
  that "catches" everything by rejecting everything would score high here.
- **Independent groundedness**: a judge from a third model family (not the drafter's, not the
  verifier's) checks claims cited in final reports against their sources.
- **Agreement with hand labels**: the owner labelled claims by reading the source text; n/a until
  `eval/gold/claims_gold.jsonl` is filled in.

## Models

{models}

Failed runs: {failed}

## How to reproduce

```bash
uv sync
cp .env.example .env   # MESH_API_KEY, TAVILY_API_KEY
uv run python -m eval.run                       # research all questions + evaluate
uv run python -m eval.label --n 50              # build the hand-labelling set
uv run python -m eval.run --skip-research       # re-evaluate stored runs (e.g. after labelling)
```

Raw numbers: the matching JSON file in `eval/results/`.
"""


async def _main(args) -> int:
    settings = get_settings()
    deps = build_deps(settings)
    eval_usage: list[dict] = []
    run_logger = deps.llm.on_usage

    def on_usage(run_id, usage):
        if run_id is None:
            eval_usage.append(usage)
        else:
            run_logger(run_id, usage)

    deps.llm.on_usage = on_usage
    runs_file = Path(args.runs_file)
    questions = [
        json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line
    ][: args.limit]
    if args.skip_research:
        stored = json.loads(runs_file.read_text(encoding="utf-8"))
    elif args.retry_failed:
        print("re-researching failed questions...")
        stored = await retry_failed(
            deps, json.loads(runs_file.read_text(encoding="utf-8")), questions
        )
        runs_file.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    else:
        print(f"researching {len(questions)} questions...")
        ids = await research_questions(deps, questions)
        stored = [
            {"question_id": q["id"], "run_id": r} for q, r in zip(questions, ids, strict=True)
        ]
        runs_file.parent.mkdir(parents=True, exist_ok=True)
        runs_file.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    run_ids = [s["run_id"] for s in stored]

    print("evaluating...")
    results = await evaluate(deps, run_ids, random.Random(args.seed))
    results["eval_cost_usd"] = usage_cost(eval_usage)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (EVAL_DIR / "RESULTS.md").write_text(render_results_md(results), encoding="utf-8")
    print(f"wrote {out} and eval/RESULTS.md")
    print(
        f"runs {results['runs']['completed']}/{results['runs']['total']} · "
        f"catch {_pct(results['verifier_probe']['catch_rate'])} · "
        f"judge {_pct(results['independent_judge']['groundedness'])} · "
        f"research cost ${_num(results['cost']['total_usd'], 3)} · "
        f"eval cost ${_num(results['eval_cost_usd'], 3)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="re-run only failed questions")
    parser.add_argument("--runs-file", default=str(RESULTS_DIR / "runs.json"))
    parser.add_argument("--seed", type=int, default=7)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
