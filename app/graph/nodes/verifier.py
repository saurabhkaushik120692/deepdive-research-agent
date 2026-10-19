import asyncio

from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.deps import Deps
from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.graph.schemas import VerdictOutput
from app.graph.state import ResearchState
from app.llm import LlmOutputError
from app.models import Claim, EvidenceChunk, Verdict
from app.prompts import load_prompt
from app.tools.quote_match import quote_in_chunk

QUOTE_NOT_FOUND = "verifier quote not found in source"


async def _judge(deps: Deps, run_id: str, claim: Claim, source_text: str) -> VerdictOutput:
    try:
        return await deps.llm.chat_json(
            role="verifier",
            system=load_prompt("verifier"),
            user=f"Claim:\n{claim.text}\n\nSource text:\n{source_text}",
            schema=VerdictOutput,
            run_id=run_id,
        )
    except LlmOutputError as e:
        # Unparseable verdict: fail closed for this claim instead of failing the whole run.
        return VerdictOutput(label="unsupported", quote="", reason=f"verifier error: {e}")


async def verify_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Check every not-yet-verified claim against ONLY the chunk(s) it cites."""
    deps = get_deps(config)
    run_id = state["run_id"]
    with node_events(deps, run_id, "verifier", round=state["round"]):
        with deps.session_factory() as session:
            claims = session.exec(
                select(Claim)
                .outerjoin(Verdict, Verdict.claim_id == Claim.id)
                .where(Claim.run_id == run_id, Verdict.id.is_(None))
                .order_by(Claim.id)
            ).all()
            texts = {}
            for c in claims:
                chunks = session.exec(
                    select(EvidenceChunk).where(EvidenceChunk.id.in_(c.chunk_ids))
                ).all()
                texts[c.id] = "\n\n".join(ch.text for ch in chunks)

        outs = await asyncio.gather(*(_judge(deps, run_id, c, texts[c.id]) for c in claims))

        with deps.session_factory() as session:
            for claim, out in zip(claims, outs, strict=True):
                found = quote_in_chunk(
                    out.quote, texts[claim.id], deps.settings.quote_match_threshold
                )
                label, reason = out.label, out.reason
                if label != "unsupported" and not found:
                    label, reason = "unsupported", QUOTE_NOT_FOUND
                session.add(
                    Verdict(
                        claim_id=claim.id,
                        label=label,
                        quote=out.quote,
                        quote_found=found,
                        reason=reason,
                        model=deps.settings.model_verifier,
                    )
                )
                session.commit()
                log_event(
                    session,
                    run_id,
                    "verifier",
                    "claim_verified",
                    {
                        "claim_id": claim.id,
                        "sub_question_id": claim.sub_question_id,
                        "label": label,
                        "quote_found": found,
                        "reason": reason,
                    },
                )
    return {}
