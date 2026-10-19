from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.graph.schemas import PlanOutput, RefineOutput
from app.graph.state import ResearchState
from app.models import Claim, SubQuestion, Verdict
from app.prompts import load_prompt


async def plan_node(state: ResearchState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    run_id = state["run_id"]
    with node_events(deps, run_id, "planner"):
        plan = await deps.llm.chat_json(
            role="planner",
            system=load_prompt("planner"),
            user=f"Research question: {state['question']}",
            schema=PlanOutput,
            run_id=run_id,
        )
        with deps.session_factory() as session:
            rows = [
                SubQuestion(run_id=run_id, text=p.text, query=p.query) for p in plan.sub_questions
            ]
            session.add_all(rows)
            session.commit()
            for r in rows:
                session.refresh(r)
            planned = [{"id": r.id, "text": r.text, "query": r.query} for r in rows]
            log_event(
                session, run_id, "planner", "sub_questions_planned", {"sub_questions": planned}
            )
    return {"pending_sq_ids": [p["id"] for p in planned]}


def _rejections(session, sq_id: int) -> list[str]:
    rows = session.exec(
        select(Claim, Verdict)
        .join(Verdict, Verdict.claim_id == Claim.id)
        .where(Claim.sub_question_id == sq_id, Verdict.label == "unsupported")
    ).all()
    return [f"- {c.text} (rejected: {v.reason})" for c, v in rows]


async def refine_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Write a better search query for every sub-question the loop rule marked thin."""
    deps = get_deps(config)
    run_id = state["run_id"]
    next_round = state["round"] + 1
    with node_events(deps, run_id, "refine", round=next_round):
        with deps.session_factory() as session:
            thin = session.exec(
                select(SubQuestion)
                .where(SubQuestion.run_id == run_id, SubQuestion.status == "thin")
                .order_by(SubQuestion.id)
            ).all()
            contexts = {sq.id: (sq.text, sq.query, _rejections(session, sq.id)) for sq in thin}

        for sq_id, (text, old_query, rejected) in contexts.items():
            rejected_text = "\n".join(rejected) or "- (no usable evidence was found)"
            user = (
                f"Original research question: {state['question']}\n"
                f"Sub-question: {text}\n"
                f"Previous search query: {old_query}\n"
                f"Rejected claims and why:\n{rejected_text}"
            )
            out = await deps.llm.chat_json(
                role="planner",
                system=load_prompt("refine"),
                user=user,
                schema=RefineOutput,
                run_id=run_id,
            )
            with deps.session_factory() as session:
                sq = session.get(SubQuestion, sq_id)
                sq.query, sq.round, sq.status = out.query, next_round, "pending"
                session.add(sq)
                session.commit()
                log_event(
                    session,
                    run_id,
                    "refine",
                    "query_refined",
                    {"sub_question_id": sq_id, "old_query": old_query, "new_query": out.query},
                )
    return {"round": next_round, "pending_sq_ids": list(contexts)}
