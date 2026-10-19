"""Build the hand-labelling set: python -m eval.label --runs-file eval/results/runs.json --n 50

Open eval/gold/claims_gold.jsonl, read each `source_text`, and set `gold_supported` to true or
false: does the source text support the claim? Labels already filled in are kept on re-runs.
"""

import argparse
import json
import random
from pathlib import Path

from app.config import get_settings
from app.deps import Deps, build_deps
from eval.common import load_claims

GOLD_PATH = Path(__file__).parent / "gold" / "claims_gold.jsonl"
LABELS = ("supported", "partial", "unsupported")


def sample_for_labelling(deps: Deps, run_ids: list[str], n: int, rng: random.Random) -> list[dict]:
    """Stratified across verifier labels so rejected and partial claims are well represented."""
    by_label = {label: [] for label in LABELS}
    for row in load_claims(deps, run_ids):
        by_label.setdefault(row.verdict.label, []).append(row)
    for rows in by_label.values():
        rng.shuffle(rows)
    picked = []
    while len(picked) < n and any(by_label.values()):
        for label in LABELS:
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop())
    return [
        {
            "claim_id": r.claim.id,
            "claim": r.claim.text,
            "source_text": r.source_text,
            "verifier_label": r.verdict.label,
            "gold_supported": None,
        }
        for r in picked
    ]


def write_gold(path: Path, rows: list[dict]) -> None:
    """Write the labelling file, keeping any gold labels already filled in."""
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["claim_id"]] = row
    for row in rows:
        old = existing.get(row["claim_id"])
        if old is not None and old.get("gold_supported") is not None:
            row["gold_supported"] = old["gold_supported"]
        existing[row["claim_id"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing.values()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", default="eval/results/runs.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_ids = [r["run_id"] for r in json.loads(Path(args.runs_file).read_text(encoding="utf-8"))]
    rows = sample_for_labelling(
        build_deps(get_settings()), run_ids, args.n, random.Random(args.seed)
    )
    write_gold(GOLD_PATH, rows)
    print(f"wrote {len(rows)} claims to {GOLD_PATH}; fill in gold_supported (true/false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
