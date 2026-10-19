from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.graph.schemas import ContradictionOutput
from app.graph.state import ResearchState
from app.llm import LlmOutputError
from app.models import Claim, EvidenceChunk, Source, SubQuestion, Verdict
from app.prompts import load_prompt


async def contradictions_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Flag verified claims that disagree across sources instead of silently picking one."""
    deps = get_deps(config)
    run_id = state["run_id"]
    with node_events(deps, run_id, "contradictions"):
        with deps.session_factory() as session:
            sqs = session.exec(
                select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.id)
            ).all()
            supported = session.exec(
                select(Claim)
                .join(Verdict, Verdict.claim_id == Claim.id)
                .where(Claim.run_id == run_id, Verdict.label == "supported")
                .order_by(Claim.id)
            ).all()
            domains = {
                ch.id: src.domain
                for ch, src in session.exec(
                    select(EvidenceChunk, Source)
                    .join(Source, Source.id == EvidenceChunk.source_id)
                    .where(EvidenceChunk.run_id == run_id)
                ).all()
            }

        for sq in sqs:
            claims = [c for c in supported if c.sub_question_id == sq.id]
            if len(claims) < 2:
                continue
            lines = [
                f"[claim:{c.id}] (source: {', '.join(domains.get(i, '?') for i in c.chunk_ids)}) "
                f"{c.text}"
                for c in claims
            ]
            try:
                out = await deps.llm.chat_json(
                    role="contradictions",
                    system=load_prompt("contradictions"),
                    user=f"Sub-question: {sq.text}\n\n" + "\n".join(lines),
                    schema=ContradictionOutput,
                    run_id=run_id,
                )
            except LlmOutputError as e:
                # A bonus check must not sink the run: record it and move on.
                with deps.session_factory() as session:
                    log_event(
                        session,
                        run_id,
                        "contradictions",
                        "contradiction_check_failed",
                        {"sub_question_id": sq.id, "error": str(e)},
                    )
                continue
            allowed = {c.id for c in claims}
            with deps.session_factory() as session:
                for conflict in out.conflicts:
                    if {conflict.claim_a, conflict.claim_b} <= allowed and (
                        conflict.claim_a != conflict.claim_b
                    ):
                        log_event(
                            session,
                            run_id,
                            "contradictions",
                            "conflict_found",
                            conflict.model_dump(),
                        )
    return {}
