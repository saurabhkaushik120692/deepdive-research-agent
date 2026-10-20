"""Hand-label the gold set, one claim at a time: uv run python -m eval.annotate

Shows a claim and the source text it cites, and asks whether the source supports it. The
verifier's own answer is never shown, and claims come in shuffled order (the file is stratified
by verifier label, so its order would hint at the answer). Every answer is saved immediately, so
you can quit and resume any time.
"""

import json
import random
import re
import textwrap
from collections.abc import Callable
from pathlib import Path

from eval.label import GOLD_PATH

PROMPT = "Does the source support the claim?  [y]es  [n]o  [s]kip  [b]ack  [q]uit > "
HELP = (
    "Answer 'y' only if the source text states the claim (same numbers, dates, names). "
    "Answer 'n' if it says something different, more, or nothing about it."
)


def load_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def mark_numbers(claim: str, source: str) -> str:
    """Mark the claim's numbers where they occur in the source (>>2023<<) so they're easy to
    find. Derived from the claim only, so it doesn't hint at the verifier's answer."""
    for n in sorted(set(re.findall(r"\d+(?:[.,]\d+)*", claim)), key=len, reverse=True):
        source = re.sub(rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)", f">>{n}<<", source)
    return source


def _screen(row: dict, position: int, total: int, done: int) -> str:
    source = textwrap.fill(mark_numbers(row["claim"], row["source_text"]), width=100)
    rule = "-" * 100
    return (
        f"\n{rule}\nClaim {position} of {total}   ({done} labelled so far)\n{rule}\n"
        f"CLAIM: {row['claim']}\n\nSOURCE TEXT:\n{source}\n{rule}\n{HELP}"
    )


def annotate(
    rows: list[dict],
    ask: Callable[[str], str],
    show: Callable[[str], None],
    save: Callable[[], None],
    seed: int = 11,
) -> None:
    queue = [i for i, r in enumerate(rows) if r.get("gold_supported") is None]
    random.Random(seed).shuffle(queue)
    history: list[int] = []
    pos = 0
    while pos < len(queue):
        i = queue[pos]
        done = sum(r.get("gold_supported") is not None for r in rows)
        show(_screen(rows[i], pos + 1, len(queue), done))
        answer = ask(PROMPT).strip().lower()
        if answer in ("y", "n"):
            rows[i]["gold_supported"] = answer == "y"
            history.append(pos)
            save()
            pos += 1
        elif answer == "s":
            pos += 1
        elif answer == "b" and history:
            pos = history.pop()
            rows[queue[pos]]["gold_supported"] = None
            save()
        elif answer == "q":
            return
        # anything else (typo, empty): show the same claim again


def main() -> int:
    path = GOLD_PATH
    if not path.exists():
        print(f"{path} not found. Build it first: uv run python -m eval.label --n 50")
        return 1
    rows = load_rows(path)
    annotate(rows, ask=input, show=print, save=lambda: save_rows(path, rows))
    labelled = sum(r.get("gold_supported") is not None for r in rows)
    print(f"\n{labelled}/{len(rows)} claims labelled, saved to {path}.")
    if labelled == len(rows):
        print("All done. Update the results: uv run python -m eval.run --skip-research")
    else:
        print("Run the same command again to continue where you left off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
