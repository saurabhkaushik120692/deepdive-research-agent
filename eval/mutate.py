"""Plant false claims: take a claim the verifier SUPPORTED and alter one fact.

The verifier should reject the altered claim against the same source. The share it catches is
the verifier's catch rate, measured per kind of alteration.
"""

import random
import re
from typing import Literal

from pydantic import BaseModel

from app.llm import LlmClient

MutationKind = Literal["number", "year", "negation", "entity"]
KINDS: tuple[MutationKind, ...] = ("number", "year", "negation", "entity")

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\w)")
_SWAPS = [
    ("rose", "fell"),
    ("rises", "falls"),
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("increase", "decrease"),
    ("higher", "lower"),
    ("grew", "shrank"),
    ("more", "less"),
]
_AUX = r"\b(is|are|was|were|will|can|has|have)\b"


def _swap_word(text: str) -> str | None:
    for a, b in _SWAPS:
        for src, dst in ((a, b), (b, a)):
            m = re.search(rf"\b{src}\b", text, flags=re.IGNORECASE)
            if m:
                word = dst.capitalize() if m.group(0)[0].isupper() else dst
                return text[: m.start()] + word + text[m.end() :]
    return None


def _negate(text: str) -> str | None:
    swapped = _swap_word(text)
    if swapped:
        return swapped
    m = re.search(_AUX + r"( not)?", text, flags=re.IGNORECASE)
    if not m:
        return None
    if m.group(2):  # remove an existing "not"
        return text[: m.end(1)] + text[m.end() :]
    return text[: m.end(1)] + " not" + text[m.end(1) :]


def _change_number(text: str, rng: random.Random) -> str | None:
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        if _YEAR.fullmatch(raw):
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = float(raw)
        factor = rng.uniform(1.3, 2.0)
        new = value * factor if rng.random() < 0.5 or value == 0 else value / factor
        new_raw = f"{new:.{decimals}f}"
        if new_raw == raw or float(new_raw) == 0:
            new_raw = f"{value * 2 + 1:.{decimals}f}"
        return text[: m.start(1)] + new_raw + text[m.end(1) :]
    return None


def _shift_year(text: str, rng: random.Random) -> str | None:
    m = _YEAR.search(text)
    if not m:
        return None
    year = int(m.group(0)) + rng.choice([-3, -2, -1, 1, 2, 3])
    return text[: m.start()] + str(year) + text[m.end() :]


def mutate_claim(text: str, kind: MutationKind, rng: random.Random) -> str | None:
    """Deterministic alterations (entity swaps need the LLM: see entity_swap)."""
    if kind == "number":
        return _change_number(text, rng)
    if kind == "year":
        return _shift_year(text, rng)
    if kind == "negation":
        return _negate(text)
    raise ValueError("entity mutations use entity_swap()")


class RewriteOutput(BaseModel):
    text: str


ENTITY_PROMPT = (
    "Rewrite the sentence so it names a DIFFERENT but plausible organisation, country, product or "
    "person in place of the main named entity. Change nothing else. If it names no entity, return "
    "it unchanged."
)


async def entity_swap(text: str, llm: LlmClient) -> str | None:
    out = await llm.chat_json(role="planner", system=ENTITY_PROMPT, user=text, schema=RewriteOutput)
    rewritten = out.text.strip()
    return None if rewritten == text.strip() else rewritten
