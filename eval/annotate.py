"""Hand-label the gold set, one claim at a time: uv run python -m eval.annotate

Shows a claim and the source text it cites, and asks whether the source supports it. The
verifier's own answer is never shown, and claims come in shuffled order (the file is stratified
by verifier label, so its order would hint at the answer). Every answer is saved immediately, so
you can quit and resume any time.
"""

import json
import os
import random
import re
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

from eval.label import GOLD_PATH

PROMPT = "Does the source support the claim?  [y]es  [n]o  [f]ull text  [s]kip  [b]ack  [q]uit > "
HELP = (
    "Answer 'y' only if the source text states the claim (same numbers, dates, names). "
    "Answer 'n' if it says something different, more, or nothing about it."
)


def load_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


_NUM = r"\d+(?:[.,]\d+)*"
_STOP = {
    "that", "this", "with", "from", "were", "have", "been", "their", "which", "about", "more",
    "than", "into", "over", "also", "will", "would", "could", "there", "they", "some", "such",
}  # fmt: skip
YELLOW, CYAN, RESET = "\x1b[1;33m", "\x1b[36m", "\x1b[0m"


def _numbers(claim: str) -> list[str]:
    return sorted(set(re.findall(_NUM, claim)), key=len, reverse=True)


def _keywords(claim: str) -> list[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", claim)}
    return sorted(words - _STOP, key=len, reverse=True)


def _num_pattern(n: str) -> str:
    return rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)"


def excerpt(claim: str, text: str, size: int = 700) -> tuple[str, int, int]:
    """The most relevant window of the source: where the claim's numbers (weighted by rarity)
    and key words occur. Returns (snippet, start, end)."""
    if len(text) <= size:
        return text, 0, len(text)
    hits: list[tuple[int, float]] = []
    for n in _numbers(claim):
        found = [m.start() for m in re.finditer(_num_pattern(n), text)]
        hits += [(pos, 3 / len(found)) for pos in found]
    for w in _keywords(claim):
        found = [m.start() for m in re.finditer(rf"\b{re.escape(w)}", text, re.IGNORECASE)]
        hits += [(pos, 1 / len(found)) for pos in found]
    best_start, best_score = 0, -1.0
    for start in range(0, len(text) - size + 50, 50):
        score = sum(w for pos, w in hits if start <= pos < start + size)
        if score > best_score:
            best_start, best_score = start, score
    start = text.rfind(" ", 0, best_start) + 1 if best_start else 0
    end = text.find(" ", min(len(text), best_start + size))
    end = len(text) if end == -1 else end
    return text[start:end], start, end


def layout(text: str) -> str:
    """Put markdown table rows and headings on their own lines; wrap the prose."""
    text = re.sub(r"\|\s+\|", "|\n|", text)
    text = re.sub(r"\s*(?:>\s*)?(#{1,6} )", r"\n\1", text)
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        lines.append(line if line.startswith("|") else textwrap.fill(line, width=100))
    return "\n".join(lines)


def highlight(claim: str, text: str, color: bool) -> str:
    """Mark the claim's numbers (and, in colour, its key words) where they occur in the source.
    Derived from the claim only, so it doesn't hint at the verifier's answer."""
    # One pass over the original text, so a mark can never match inside an earlier mark's codes.
    parts = (
        [f"(?P<num>{'|'.join(_num_pattern(n) for n in _numbers(claim))})"]
        if _numbers(claim)
        else []
    )
    if color and _keywords(claim):
        parts.append(f"(?P<word>\\b(?:{'|'.join(re.escape(w) for w in _keywords(claim))}))")
    if not parts:
        return text

    def mark(m: re.Match) -> str:
        if m.group("num") is not None:
            return f"{YELLOW}{m.group(0)}{RESET}" if color else f">>{m.group(0)}<<"
        return f"{CYAN}{m.group(0)}{RESET}"

    return re.sub("|".join(parts), mark, text, flags=re.IGNORECASE)


def _screen(
    row: dict, position: int, total: int, done: int, full: bool = False, color: bool = False
) -> str:
    source = row["source_text"]
    if full:
        body, note = source, "(full source text)"
    else:
        body, start, end = excerpt(row["claim"], source)
        note = (
            "(whole source shown)"
            if (start, end) == (0, len(source))
            else f"(most relevant part, characters {start}-{end} of {len(source)}; "
            "press f for the full source)"
        )
    rule = "-" * 100
    clear = "\x1b[2J\x1b[H" if color else ""
    claim = highlight(row["claim"], row["claim"], color) if color else row["claim"]
    return (
        f"{clear}\n{rule}\nClaim {position} of {total}   ({done} labelled so far)\n{rule}\n"
        f"CLAIM: {claim}\n\nSOURCE {note}:\n\n{highlight(row['claim'], layout(body), color)}"
        f"\n{rule}\n{HELP}"
    )


def annotate(
    rows: list[dict],
    ask: Callable[[str], str],
    show: Callable[[str], None],
    save: Callable[[], None],
    seed: int = 11,
    color: bool = False,
) -> None:
    queue = [i for i, r in enumerate(rows) if r.get("gold_supported") is None]
    random.Random(seed).shuffle(queue)
    history: list[int] = []
    pos, full = 0, False
    while pos < len(queue):
        i = queue[pos]
        done = sum(r.get("gold_supported") is not None for r in rows)
        show(_screen(rows[i], pos + 1, len(queue), done, full=full, color=color))
        answer = ask(PROMPT).strip().lower()
        if answer == "f":
            full = True
            continue
        full = False
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
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    if color and os.name == "nt":
        os.system("")  # turn on ANSI colours in the Windows console
    annotate(rows, ask=input, show=print, save=lambda: save_rows(path, rows), color=color)
    labelled = sum(r.get("gold_supported") is not None for r in rows)
    print(f"\n{labelled}/{len(rows)} claims labelled, saved to {path}.")
    if labelled == len(rows):
        print("All done. Update the results: uv run python -m eval.run --skip-research")
    else:
        print("Run the same command again to continue where you left off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
