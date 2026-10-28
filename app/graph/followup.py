"""Follow-up questions under a finished report: a second, smaller LangGraph graph.

START -> scope -> END (off-topic: declined)
              -> draft -> verify -> route
route -> search -> draft -> verify   (too few new verified claims; the web is searched once)
route -> answer -> END

The follow-up reuses the report's research: saved chunks are searched first, and the report's
verified claims can be cited as they are. New claims go through the same verifier and quote check,
and the answer may only state what its cited claims say.
"""

import asyncio
from datetime import UTC, datetime
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from sqlmodel import select

from app.deps import Deps
from app.events import current_followup, log_event
from app.graph.nodes import get_deps, node_events
from app.graph.nodes.drafter import _evidence_block, claim_key
from app.graph.nodes.researcher import research_node
from app.graph.nodes.synthesizer import VERIFIED, _paragraph_problem
from app.graph.nodes.verifier import verify_claim
from app.graph.runner import _first_leaf
from app.graph.schemas import DraftOutput, FollowUpAnswer, Paragraph, ScopeOutput
from app.llm import LlmOutputError
from app.models import Claim, EvidenceChunk, FollowUp, Report, Run, Source, SubQuestion, Verdict
from app.prompts import load_prompt

EARLIER_FOLLOWUPS = 3  # answered follow-ups shown to the scope check, for short questions


class FollowUpState(TypedDict):
    followup_id: int
    run_id: str
    sq_id: int
    known_ids: list[int]  # the report's verified claims that bear on this follow-up
    searched: bool
    declined: bool


def _update(deps: Deps, followup_id: int, **fields) -> None:
    with deps.session_factory() as session:
        f = session.get(FollowUp, followup_id)
        for name, value in fields.items():
            setattr(f, name, value)
        session.add(f)
        session.commit()


def _report_outline(session, run_id: str) -> str:
    report = session.get(Report, run_id)
    synthesis = (report.synthesis if report else None) or {}
    headings = [s.get("heading", "") for s in synthesis.get("sections", [])]
    return (
        f"Report title: {synthesis.get('title', '')}\n"
        f"Report summary: {synthesis.get('summary', '')}\n"
        "Report sections: " + "; ".join(h for h in headings if h)
    )


def _earlier_block(session, f: FollowUp) -> str:
    rows = session.exec(
        select(FollowUp)
        .where(FollowUp.run_id == f.run_id, FollowUp.status == "answered", FollowUp.id < f.id)
        .order_by(FollowUp.id.desc())
        .limit(EARLIER_FOLLOWUPS)
    ).all()
    if not rows:
        return "Earlier follow-ups: none"
    lines = []
    for r in reversed(rows):
        paragraphs = (r.answer or {}).get("paragraphs") or [{"text": "(no supported answer)"}]
        lines.append(f"- Q: {r.question}\n  A: {paragraphs[0]['text']}")
    return "Earlier follow-ups:\n" + "\n".join(lines)


async def scope_node(state: FollowUpState, config: RunnableConfig) -> dict:
    """Is the follow-up within the report's subject? If so, rewrite it to stand on its own."""
    deps = get_deps(config)
    fid, run_id = state["followup_id"], state["run_id"]
    _update(deps, fid, stage="scope")
    with node_events(deps, run_id, "followup_scope", followup_id=fid):
        with deps.session_factory() as session:
            f = session.get(FollowUp, fid)
            run = session.get(Run, run_id)
            user = (
                f"Original research question: {run.question}\n"
                f"{_report_outline(session, run_id)}\n\n"
                f"{_earlier_block(session, f)}\n\n"
                f"New follow-up: {f.question}"
            )
        out = await deps.llm.chat_json(
            role="scope",
            system=load_prompt("scope"),
            user=user,
            schema=ScopeOutput,
            run_id=run_id,
        )
        with deps.session_factory() as session:
            log_event(
                session,
                run_id,
                "followup_scope",
                "scope_checked",
                {
                    "followup_id": fid,
                    "in_scope": out.in_scope,
                    "reason": out.reason,
                    "standalone_question": out.standalone_question,
                },
            )
            f = session.get(FollowUp, fid)
            if not out.in_scope:
                f.status, f.decline_reason = "declined", out.reason
                session.add(f)
                session.commit()
                return {"declined": True}
            standalone = out.standalone_question.strip() or f.question
            sq = SubQuestion(
                run_id=run_id,
                text=standalone,
                query=out.search_query.strip() or standalone,
                status="followup",
                followup_id=fid,
            )
            session.add(sq)
            f.standalone_question = standalone
            session.add(f)
            session.commit()
            session.refresh(sq)
            return {"declined": False, "sq_id": sq.id}


def _known_block(texts: list[str]) -> str:
    if not texts:
        return ""
    lines = "\n".join(f"- {t}" for t in texts)
    return f"\n\nAlready established by the report (do NOT repeat these):\n{lines}"


async def draft_node(state: FollowUpState, config: RunnableConfig) -> dict:
    """Draft new claims for the follow-up: first from the run's saved evidence, after a web
    search from the new evidence only. Facts the report already has are never drafted again."""
    deps = get_deps(config)
    fid, run_id, sq_id, searched = (
        state["followup_id"],
        state["run_id"],
        state["sq_id"],
        state["searched"],
    )
    _update(deps, fid, stage="drafting")
    with node_events(deps, run_id, "followup_drafter", followup_id=fid, searched=searched):
        with deps.session_factory() as session:
            question = session.get(SubQuestion, sq_id).text
            run_question = session.get(Run, run_id).question
        k = deps.settings.top_k_chunks
        if searched:
            chunk_ids = await deps.vectors.query(run_id, sq_id, question, k)
        else:
            chunk_ids = await deps.vectors.query_run(run_id, question, k)

        with deps.session_factory() as session:
            rows = session.exec(
                select(EvidenceChunk, Source)
                .join(Source, Source.id == EvidenceChunk.source_id)
                .where(EvidenceChunk.id.in_(chunk_ids))
            ).all()
            report_claims = session.exec(
                select(Claim, Verdict)
                .outerjoin(Verdict, Verdict.claim_id == Claim.id)
                .where(Claim.run_id == run_id, Claim.followup_id.is_(None))
            ).all()
            own = session.exec(select(Claim.text).where(Claim.followup_id == fid)).all()
        if searched:
            known_ids = state["known_ids"]
        else:
            retrieved = set(chunk_ids)
            known_ids = [
                c.id
                for c, v in report_claims
                if v is not None and v.label in VERIFIED and retrieved & set(c.chunk_ids)
            ]
        known_texts = [c.text for c, _ in report_claims if c.id in set(known_ids)]
        taken = {claim_key(c.text) for c, _ in report_claims} | {claim_key(t) for t in own}

        if not rows:
            with deps.session_factory() as session:
                log_event(session, run_id, "followup_drafter", "no_evidence", {"followup_id": fid})
            return {"known_ids": known_ids}
        rows = sorted(rows, key=lambda r: (r[1].tier, chunk_ids.index(r[0].id)))
        given = {ch.id for ch, _ in rows}
        try:
            draft = await deps.llm.chat_json(
                role="followup_drafter",
                system=load_prompt("drafter"),
                user=(
                    f"Research question: {run_question}\n"
                    f"Sub-question (id {sq_id}): {question}\n\n"
                    f"Evidence:\n\n{_evidence_block(rows)}" + _known_block(known_texts)
                ),
                schema=DraftOutput,
                run_id=run_id,
            )
        except LlmOutputError as e:
            with deps.session_factory() as session:
                log_event(
                    session,
                    run_id,
                    "followup_drafter",
                    "draft_failed",
                    {"followup_id": fid, "error": str(e)},
                )
            return {"known_ids": known_ids}

        with deps.session_factory() as session:
            kept = 0
            for c in draft.claims:
                if not set(c.chunk_ids) <= given:
                    log_event(
                        session,
                        run_id,
                        "followup_drafter",
                        "claim_dropped_bad_citation",
                        {"followup_id": fid, "text": c.text, "chunk_ids": c.chunk_ids},
                    )
                    continue
                if claim_key(c.text) in taken:
                    # Includes claims the report rejected: a follow-up can't bring them back.
                    log_event(
                        session,
                        run_id,
                        "followup_drafter",
                        "claim_dropped_duplicate",
                        {"followup_id": fid, "text": c.text},
                    )
                    continue
                taken.add(claim_key(c.text))
                session.add(
                    Claim(
                        run_id=run_id,
                        sub_question_id=sq_id,
                        text=c.text,
                        chunk_ids=c.chunk_ids,
                        round=1 if searched else 0,
                        followup_id=fid,
                    )
                )
                kept += 1
            session.commit()
            log_event(
                session,
                run_id,
                "followup_drafter",
                "claims_drafted",
                {"followup_id": fid, "n_claims": kept, "n_known": len(known_ids)},
            )
    return {"known_ids": known_ids}


async def verify_node(state: FollowUpState, config: RunnableConfig) -> dict:
    """The report's verifier and quote check, on this follow-up's unverified claims."""
    deps = get_deps(config)
    fid, run_id = state["followup_id"], state["run_id"]
    with node_events(deps, run_id, "followup_verifier", followup_id=fid):
        with deps.session_factory() as session:
            claims = session.exec(
                select(Claim)
                .outerjoin(Verdict, Verdict.claim_id == Claim.id)
                .where(Claim.followup_id == fid, Verdict.id.is_(None))
                .order_by(Claim.id)
            ).all()
            texts = {}
            for c in claims:
                chunks = session.exec(
                    select(EvidenceChunk).where(EvidenceChunk.id.in_(c.chunk_ids))
                ).all()
                texts[c.id] = "\n\n".join(ch.text for ch in chunks)
        outs = await asyncio.gather(
            *(verify_claim(deps, run_id, c.text, texts[c.id]) for c in claims)
        )
        with deps.session_factory() as session:
            for claim, (out, found) in zip(claims, outs, strict=True):
                session.add(
                    Verdict(
                        claim_id=claim.id,
                        label=out.label,
                        quote=out.quote,
                        quote_found=found,
                        reason=out.reason,
                        model=deps.settings.model_verifier,
                    )
                )
                session.commit()
                log_event(
                    session,
                    run_id,
                    "followup_verifier",
                    "claim_verified",
                    {
                        "followup_id": fid,
                        "claim_id": claim.id,
                        "label": out.label,
                        "quote_found": found,
                        "reason": out.reason,
                    },
                )
    return {}


def route_after_verify(state: FollowUpState, config: RunnableConfig) -> Literal["search", "answer"]:
    """Plain-code rule: too few NEW verified facts and no web search yet -> search the web.
    The report's own claims don't count: the user is asking for more than the report says."""
    deps = get_deps(config)
    fid, run_id = state["followup_id"], state["run_id"]
    with deps.session_factory() as session:
        new = session.exec(
            select(Claim.id)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.followup_id == fid, Verdict.label.in_(VERIFIED))
        ).all()
        searching = len(new) < deps.settings.min_supported_claims and not state["searched"]
        log_event(
            session,
            run_id,
            "followup_router",
            "loop_decision",
            {"followup_id": fid, "new_verified": len(new), "searching": searching},
        )
    return "search" if searching else "answer"


async def search_node(state: FollowUpState, config: RunnableConfig) -> dict:
    """One web search for the follow-up, with the run's existing researcher."""
    deps = get_deps(config)
    fid = state["followup_id"]
    _update(deps, fid, stage="searching", searched_web=True)
    payload = {
        "run_id": state["run_id"],
        "sub_question_id": state["sq_id"],
        "round": 1,
        "followup_id": fid,
    }
    await research_node(payload, config)
    return {"searched": True}


NONE_FOUND = {"paragraphs": [], "none_found": True}


async def answer_node(state: FollowUpState, config: RunnableConfig) -> dict:
    """Answer from verified claims only; paragraphs stating more than their claims are dropped.

    Unlike the report, claims the answer doesn't use are left out: a follow-up answers one
    question, and the report already lists everything else."""
    deps = get_deps(config)
    fid, run_id = state["followup_id"], state["run_id"]
    _update(deps, fid, stage="writing")
    with node_events(deps, run_id, "followup_answer", followup_id=fid):
        with deps.session_factory() as session:
            question = session.get(SubQuestion, state["sq_id"]).text
            rows = session.exec(
                select(Claim, Verdict)
                .join(Verdict, Verdict.claim_id == Claim.id)
                .where(
                    (Claim.id.in_(state["known_ids"])) | (Claim.followup_id == fid),
                    Verdict.label.in_(VERIFIED),
                )
                .order_by(Claim.id)
            ).all()
        known = set(state["known_ids"])
        allowed = [c for c, _ in rows if c.id in known] + [c for c, _ in rows if c.id not in known]
        partial = {c.id for c, v in rows if v.label == "partial"}
        if not allowed:
            _update(deps, fid, answer=NONE_FOUND)
            return {}

        claims = {c.id: c for c in allowed}
        lines = [
            f"[claim:{c.id}]{' (only partly supported)' if c.id in partial else ''} {c.text}"
            for c in allowed
        ]
        try:
            out = await deps.llm.chat_json(
                role="followup_answer",
                system=load_prompt("followup_answer"),
                user=f"Follow-up question: {question}\n\nVerified claims:\n" + "\n".join(lines),
                schema=FollowUpAnswer,
                run_id=run_id,
            )
            paragraphs = out.paragraphs
        except LlmOutputError as e:
            # The claims are verified already: list them rather than lose the answer.
            paragraphs = []
            with deps.session_factory() as session:
                log_event(
                    session,
                    run_id,
                    "followup_answer",
                    "answer_fallback",
                    {"followup_id": fid, "reason": str(e)},
                )

        kept: list[Paragraph] = []
        with deps.session_factory() as session:
            for p in paragraphs:
                reason = _paragraph_problem(p, set(claims), claims)
                if reason is None:
                    kept.append(p)
                else:
                    log_event(
                        session,
                        run_id,
                        "followup_answer",
                        "paragraph_dropped",
                        {
                            "followup_id": fid,
                            "text": p.text,
                            "claim_ids": p.claim_ids,
                            "reason": reason,
                        },
                    )
        if not kept:
            # Nothing written survived the checks: list the verified claims rather than say nothing.
            kept = [Paragraph(text=c.text, claim_ids=[c.id]) for c in allowed]
        answer = {"paragraphs": [p.model_dump() for p in kept], "none_found": False}
        _update(deps, fid, answer=answer)
    return {}


def build_followup_graph():
    g = StateGraph(FollowUpState)
    g.add_node("scope", scope_node)
    g.add_node("draft", draft_node)
    g.add_node("verify", verify_node)
    g.add_node("search", search_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "scope")
    g.add_conditional_edges(
        "scope", lambda s: END if s["declined"] else "draft", {END: END, "draft": "draft"}
    )
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"search": "search", "answer": "answer"})
    g.add_edge("search", "draft")
    g.add_edge("answer", END)
    return g.compile()


async def run_followup(followup_id: int, deps: Deps) -> None:
    """Answer one follow-up end to end. Never raises: failures are recorded on the follow-up."""
    with deps.session_factory() as session:
        f = session.get(FollowUp, followup_id)
        f.status = "running"
        session.add(f)
        session.commit()
        run_id = f.run_id
        log_event(
            session,
            run_id,
            "followup",
            "followup_started",
            {"followup_id": followup_id, "question": f.question},
        )
    status, error = "answered", None
    token = current_followup.set(followup_id)  # tags this follow-up's llm_call events too
    try:
        final = await build_followup_graph().ainvoke(
            {
                "followup_id": followup_id,
                "run_id": run_id,
                "sq_id": 0,
                "known_ids": [],
                "searched": False,
                "declined": False,
            },
            {"configurable": {"deps": deps}, "recursion_limit": 25},
        )
        if final.get("declined"):
            status = "declined"
    except asyncio.CancelledError:
        _finish(deps, followup_id, "failed", "follow-up cancelled: server shutting down")
        raise
    except Exception as e:  # recorded, not re-raised: follow-ups run as background tasks
        root = _first_leaf(e)
        status, error = "failed", f"{type(root).__name__}: {root}"
    finally:
        current_followup.reset(token)
    _finish(deps, followup_id, status, error)


def _finish(deps: Deps, followup_id: int, status: str, error: str | None) -> None:
    with deps.session_factory() as session:
        f = session.get(FollowUp, followup_id)
        f.status, f.error = status, error
        f.finished_at = datetime.now(UTC)
        session.add(f)
        session.commit()
        log_event(
            session,
            f.run_id,
            "followup",
            "followup_failed" if status == "failed" else "followup_finished",
            {"followup_id": followup_id, "status": status, "error": error},
        )


def fail_interrupted_followups(deps: Deps) -> int:
    """On startup: follow-ups left queued/running by a restart can never finish, so close them."""
    with deps.session_factory() as session:
        stuck = session.exec(
            select(FollowUp.id).where(FollowUp.status.in_(["queued", "running"]))
        ).all()
    for followup_id in stuck:
        _finish(deps, followup_id, "failed", "interrupted by server restart")
    return len(stuck)
