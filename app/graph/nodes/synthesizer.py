from collections import Counter

import openai
from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.graph.schemas import Conflict, Paragraph, Section, SynthesisOutput
from app.graph.state import ResearchState
from app.llm import LlmOutputError
from app.models import Claim, Event, EvidenceChunk, Report, Source, SubQuestion, Verdict
from app.prompts import load_prompt
from app.report import render_report
from app.tools.quote_match import NUMBER, normalize

VERIFIED = ("supported", "partial")


def _claims_prompt(question, sub_questions, verified, verdicts, chunk_sources) -> str:
    parts = [f"Research question: {question}", "", "Verified claims:"]
    for sq in sub_questions:
        rows = [c for c in verified if c.sub_question_id == sq.id]
        if not rows:
            continue
        parts += ["", f"Sub-question: {sq.text}"]
        for c in rows:
            tier = min(
                (chunk_sources[ch].tier for ch in c.chunk_ids if ch in chunk_sources), default=3
            )
            partial = ", only partly supported" if verdicts[c.id].label == "partial" else ""
            parts.append(f"[claim:{c.id}] (source tier {tier}{partial}) {c.text}")
    return "\n".join(parts)


def _numbers(text: str) -> set[str]:
    return set(NUMBER.findall(normalize(text)))


def _paragraph_problem(p, allowed: set[int], claims: dict[int, Claim]) -> str | None:
    """Why a paragraph can't go in the report, or None. Prose may reword its claims, but it may
    only cite verified claims, and every figure it states must come from those claims."""
    if not set(p.claim_ids) <= allowed:
        return "cites unverified claims"
    cited_numbers = set().union(*(_numbers(claims[i].text) for i in p.claim_ids))
    extra = _numbers(p.text) - cited_numbers
    if extra:
        return f"numbers not in cited claims: {sorted(extra)}"
    return None


def _fallback_synthesis(question, sub_questions, verified) -> SynthesisOutput:
    sections = []
    for sq in sub_questions:
        rows = [c for c in verified if c.sub_question_id == sq.id]
        if rows:
            paragraphs = [Paragraph(text=c.text, claim_ids=[c.id]) for c in rows]
            sections.append(Section(heading=sq.text, paragraphs=paragraphs))
    return SynthesisOutput(
        title=question,
        summary="The verified findings are listed below, grouped by sub-question.",
        sections=sections,
    )


async def synthesize_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Write the report from verified claims only; paragraphs citing anything else are dropped."""
    deps = get_deps(config)
    run_id, question = state["run_id"], state["question"]
    with node_events(deps, run_id, "synthesizer"):
        with deps.session_factory() as session:
            sub_questions = session.exec(
                select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.id)
            ).all()
            claims = {
                c.id: c for c in session.exec(select(Claim).where(Claim.run_id == run_id)).all()
            }
            verdicts = {
                v.claim_id: v
                for v in session.exec(
                    select(Verdict).where(Verdict.claim_id.in_(list(claims)))
                ).all()
            }
            chunk_sources = {
                ch.id: src
                for ch, src in session.exec(
                    select(EvidenceChunk, Source)
                    .join(Source, Source.id == EvidenceChunk.source_id)
                    .where(EvidenceChunk.run_id == run_id)
                ).all()
            }
            all_sources = session.exec(select(Source).where(Source.run_id == run_id)).all()
            conflicts = [
                Conflict(**e.payload)
                for e in session.exec(
                    select(Event).where(Event.run_id == run_id, Event.type == "conflict_found")
                ).all()
            ]

        verified = [
            c for c in claims.values() if c.id in verdicts and verdicts[c.id].label in VERIFIED
        ]
        allowed = {c.id for c in verified}

        fallback = None
        if verified:
            try:
                synthesis = await deps.llm.chat_json(
                    role="synthesizer",
                    system=load_prompt("synthesizer"),
                    user=_claims_prompt(question, sub_questions, verified, verdicts, chunk_sources),
                    schema=SynthesisOutput,
                    run_id=run_id,
                )
            except (LlmOutputError, openai.APIError) as e:
                # Bad output, or Mesh timed out / errored even after retries. The claims are
                # already verified: list them rather than lose the whole run at its last step.
                synthesis = _fallback_synthesis(question, sub_questions, verified)
                fallback = f"{type(e).__name__}: {e}"
        else:
            synthesis = SynthesisOutput(
                title=question,
                summary="No claim could be verified against its sources, so this report "
                "contains no findings.",
                sections=[],
            )

        with deps.session_factory() as session:
            if fallback:
                log_event(
                    session, run_id, "synthesizer", "synthesis_fallback", {"reason": fallback}
                )
            all_numbers = set().union(*(_numbers(c.text) for c in verified))
            if not _numbers(synthesis.summary) <= all_numbers:
                log_event(
                    session,
                    run_id,
                    "synthesizer",
                    "summary_replaced",
                    {"text": synthesis.summary, "reason": "numbers not in verified claims"},
                )
                synthesis.summary = "Summary withheld: it stated figures that no verified claim "
                synthesis.summary += "supports. See the findings below."
            for section in synthesis.sections:
                kept = []
                for p in section.paragraphs:
                    reason = _paragraph_problem(p, allowed, claims)
                    if reason is None:
                        kept.append(p)
                    else:
                        log_event(
                            session,
                            run_id,
                            "synthesizer",
                            "paragraph_dropped",
                            {"text": p.text, "claim_ids": p.claim_ids, "reason": reason},
                        )
                section.paragraphs = kept
            synthesis.sections = [s for s in synthesis.sections if s.paragraphs]

            thin_texts = [sq.text for sq in sub_questions if sq.status == "thin"]
            markdown = render_report(
                question, synthesis, claims, verdicts, chunk_sources, thin_texts, conflicts
            )
            labels = Counter(v.label for v in verdicts.values())
            metrics = {
                "claims_drafted": len(claims),
                "supported": labels["supported"],
                "partial": labels["partial"],
                "unsupported": labels["unsupported"],
                "rounds": state["round"],
                "sources": len(all_sources),
                "tier_counts": {str(t): n for t, n in Counter(s.tier for s in all_sources).items()},
            }
            session.merge(
                Report(
                    run_id=run_id,
                    markdown=markdown,
                    metrics=metrics,
                    synthesis=synthesis.model_dump(),
                )
            )
            session.commit()
            log_event(session, run_id, "synthesizer", "report_written", metrics)
    return {}
