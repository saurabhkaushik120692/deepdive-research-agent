import asyncio
import re

from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.graph.schemas import DraftOutput
from app.graph.state import ResearchState
from app.llm import LlmOutputError
from app.models import Claim, EvidenceChunk, Source, SubQuestion
from app.prompts import load_prompt


def _evidence_block(rows: list[tuple[EvidenceChunk, Source]]) -> str:
    return "\n\n".join(
        f"[chunk:{ch.id}] (source: {src.domain}, tier {src.tier})\n{ch.text}" for ch, src in rows
    )


def claim_key(text: str) -> str:
    """Identity of a claim for de-duplication: case, spacing and punctuation ignored."""
    return " ".join(re.sub(r"[^a-z0-9%]+", " ", text.lower()).split())


def _previous_block(previous: list[str]) -> str:
    if not previous:
        return ""
    lines = "\n".join(f"- {t}" for t in previous)
    return f"\n\nAlready drafted for this sub-question (do NOT repeat these):\n{lines}"


async def _draft_one(deps, state: ResearchState, sq_id: int) -> None:
    run_id, round_ = state["run_id"], state["round"]
    with deps.session_factory() as session:
        sq_text = session.get(SubQuestion, sq_id).text
        previous = session.exec(
            select(Claim.text).where(Claim.sub_question_id == sq_id).order_by(Claim.id)
        ).all()
    seen = {claim_key(t) for t in previous}
    chunk_ids = await deps.vectors.query(run_id, sq_id, sq_text, deps.settings.top_k_chunks)
    with deps.session_factory() as session:
        rows = session.exec(
            select(EvidenceChunk, Source)
            .join(Source, Source.id == EvidenceChunk.source_id)
            .where(EvidenceChunk.id.in_(chunk_ids))
        ).all()
        if not rows:
            log_event(session, run_id, "drafter", "no_evidence", {"sub_question_id": sq_id})
            return
    rows = sorted(rows, key=lambda r: (r[1].tier, chunk_ids.index(r[0].id)))
    given = {ch.id for ch, _ in rows}

    try:
        draft = await deps.llm.chat_json(
            role="drafter",
            system=load_prompt("drafter"),
            user=(
                f"Research question: {state['question']}\n"
                f"Sub-question (id {sq_id}): {sq_text}\n\n"
                f"Evidence:\n\n{_evidence_block(rows)}" + _previous_block(previous)
            ),
            schema=DraftOutput,
            run_id=run_id,
        )
    except LlmOutputError as e:
        # One unparseable draft leaves this sub-question thin (the loop rule handles it);
        # it must not fail the whole run.
        with deps.session_factory() as session:
            log_event(
                session,
                run_id,
                "drafter",
                "draft_failed",
                {"sub_question_id": sq_id, "error": str(e)},
            )
        return

    with deps.session_factory() as session:
        kept = 0
        for c in draft.claims:
            if not set(c.chunk_ids) <= given:
                log_event(
                    session,
                    run_id,
                    "drafter",
                    "claim_dropped_bad_citation",
                    {"sub_question_id": sq_id, "text": c.text, "chunk_ids": c.chunk_ids},
                )
                continue
            if claim_key(c.text) in seen:
                # A re-research round must not re-count facts already drafted.
                log_event(
                    session,
                    run_id,
                    "drafter",
                    "claim_dropped_duplicate",
                    {"sub_question_id": sq_id, "text": c.text},
                )
                continue
            seen.add(claim_key(c.text))
            session.add(
                Claim(
                    run_id=run_id,
                    sub_question_id=sq_id,
                    text=c.text,
                    chunk_ids=c.chunk_ids,
                    round=round_,
                )
            )
            kept += 1
        session.commit()
        log_event(
            session,
            run_id,
            "drafter",
            "claims_drafted",
            {"sub_question_id": sq_id, "n_claims": kept},
        )


async def draft_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Turn retrieved evidence into atomic claims, each citing the chunk(s) it came from.
    Sub-questions are drafted concurrently (each call is an independent LLM request)."""
    deps = get_deps(config)
    with node_events(deps, state["run_id"], "drafter", round=state["round"]):
        # TaskGroup (not gather): if one sub-question fails hard, the others are cancelled
        # instead of spending budget and writing into a run that is already failed.
        async with asyncio.TaskGroup() as tg:
            for sq in state["pending_sq_ids"]:
                tg.create_task(_draft_one(deps, state, sq))
    return {}
