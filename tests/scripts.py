"""Whole-run scripts for graph tests.

Sub-questions are "Sub-question A/B/C". Each search round returns one page whose text says
"Evidence for X round N ..."; the fake drafter writes two claims "X fact 1/2 round N"; the fake
verifier decides labels from (letter, claim number, round) and quotes the real source sentence.
"""

import re

from app.deps import SearchResult
from app.graph.schemas import (
    DraftClaim,
    DraftOutput,
    Paragraph,
    PlannedSubQuestion,
    PlanOutput,
    RefineOutput,
    Section,
    SynthesisOutput,
    VerdictOutput,
)

LETTERS = ("A", "B", "C")


def _page(letter: str, round_: int) -> str:
    return (
        f"Evidence for {letter} round {round_}: analysts reported two separate findings about "
        f"topic {letter}, the first on supply and the second on prices."
    )


def _install_search(deps, rounds: int = 3) -> None:
    for letter in LETTERS:
        query = f"query {letter}"
        for r in range(rounds):
            url = f"https://src-{letter.lower()}{r}.org/p"
            deps.search.results[query] = [SearchResult(url, f"Src {letter}{r}", _page(letter, r))]
            query = f"refined {query}"


def _planner(system: str, user: str):
    if "Previous search query:" in user:
        old = re.search(r"Previous search query: (.*)", user).group(1)
        return RefineOutput(query=f"refined {old}")
    return PlanOutput(
        sub_questions=[
            PlannedSubQuestion(text=f"Sub-question {x}", query=f"query {x}") for x in LETTERS
        ]
    )


def _drafter(system: str, user: str):
    sq_id, letter = re.search(r"Sub-question \(id (\d+)\): Sub-question (\w)", user).groups()
    chunks = [
        (int(cid), int(r))
        for cid, r in re.findall(r"\[chunk:(\d+)\][^\n]*\nEvidence for \w round (\d+)", user)
    ]
    chunk_id, round_ = max(chunks, key=lambda c: c[1])
    return DraftOutput(
        claims=[
            DraftClaim(
                sub_question_id=int(sq_id),
                text=f"{letter} fact {i} round {round_}",
                chunk_ids=[chunk_id],
            )
            for i in (1, 2)
        ]
    )


def _verifier(thin_in_round0: set, always_thin: set):
    def handler(system: str, user: str):
        letter, i, round_ = re.search(r"Claim:\n(\w) fact (\d) round (\d)", user).groups()
        i, round_ = int(i), int(round_)
        if letter in always_thin:
            ok = False
        elif letter in thin_in_round0 and round_ == 0:
            ok = i == 1
        else:
            ok = True
        if not ok:
            return VerdictOutput(label="unsupported", quote="", reason=f"{letter} fact {i} absent")
        quote = f"Evidence for {letter} round {round_}"
        return VerdictOutput(label="supported", quote=quote, reason="stated")

    return handler


def _synthesizer(cite_rejected: bool):
    def handler(system: str, user: str):
        ids = [int(x) for x in re.findall(r"\[claim:(\d+)\]", user)]
        paragraphs = [
            Paragraph(text=f"Finding backed by claim {cid}.", claim_ids=[cid]) for cid in ids
        ]
        if cite_rejected:
            paragraphs.append(Paragraph(text="B fact 2 round 0", claim_ids=[99999]))
        return SynthesisOutput(
            title="EV battery risks",
            summary="Summary of verified findings.",
            sections=[Section(heading="Findings", paragraphs=paragraphs)],
        )

    return handler


def script_happy_path(
    deps, thin_in_round0=frozenset(), always_thin=frozenset(), synth_cites_rejected=False
):
    _install_search(deps)
    deps.llm.handlers.update(
        planner=_planner,
        drafter=_drafter,
        verifier=_verifier(set(thin_in_round0), set(always_thin)),
        synthesizer=_synthesizer(synth_cites_rejected),
    )


def script_no_results(deps):
    deps.llm.handlers.update(planner=_planner, synthesizer=_synthesizer(False))
