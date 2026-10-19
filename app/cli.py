"""Run one research question from the terminal: uv run python -m app.cli "your question" """

import argparse
import asyncio
import json

from app.config import get_settings
from app.deps import build_deps
from app.graph.runner import create_run, run_research
from app.models import Report, Run


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepDive: self-verifying research run")
    parser.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    deps = build_deps(settings)
    with deps.session_factory() as session:
        run_id = create_run(session, args.question, settings).id
    print(f"run {run_id} started")

    asyncio.run(run_research(run_id, deps))

    with deps.session_factory() as session:
        run = session.get(Run, run_id)
        report = session.get(Report, run_id)
        print(f"status: {run.status}" + (f" ({run.error})" if run.error else ""))
        print(f"tokens: {run.prompt_tokens} in / {run.completion_tokens} out")
        if report:
            out = settings.data_dir / "reports" / f"{run_id}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report.markdown, encoding="utf-8")
            print(f"report: {out}")
            print(json.dumps(report.metrics, indent=2))
    return 0 if run.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
