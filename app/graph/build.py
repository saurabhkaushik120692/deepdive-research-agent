"""The coordinator: a LangGraph StateGraph wiring the agents together, with a real loop.

START → plan →(Send per sub-question) research → draft → verify →(route) refine | contradictions
refine →(Send per thin sub-question) research   ·   contradictions → synthesize → END
"""

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sqlmodel import select

from app.events import log_event
from app.graph.nodes import get_deps
from app.graph.nodes.contradictions import contradictions_node
from app.graph.nodes.drafter import draft_node
from app.graph.nodes.planner import plan_node, refine_node
from app.graph.nodes.researcher import research_node
from app.graph.nodes.synthesizer import synthesize_node
from app.graph.nodes.verifier import verify_node
from app.graph.rules import should_loop, thin_sub_questions
from app.graph.state import ResearchState
from app.models import Claim, SubQuestion, Verdict


def dispatch_research(state: ResearchState) -> list[Send]:
    return [
        Send(
            "research",
            {"run_id": state["run_id"], "sub_question_id": sq_id, "round": state["round"]},
        )
        for sq_id in state["pending_sq_ids"]
    ]


def route_after_verify(
    state: ResearchState, config: RunnableConfig
) -> Literal["refine", "contradictions"]:
    """Plain-code loop rule: re-research sub-questions with too few supported claims."""
    deps = get_deps(config)
    run_id, round_ = state["run_id"], state["round"]
    with deps.session_factory() as session:
        sqs = session.exec(select(SubQuestion).where(SubQuestion.run_id == run_id)).all()
        counts = {sq.id: 0 for sq in sqs}
        supported = session.exec(
            select(Claim.sub_question_id)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id, Verdict.label == "supported")
        ).all()
        for sq_id in supported:
            counts[sq_id] += 1
        thin = thin_sub_questions(counts, deps.settings.min_supported_claims)
        looping = should_loop(thin, round_, deps.settings.max_extra_rounds)
        for sq in sqs:
            sq.status = "thin" if sq.id in thin else "ok"
            session.add(sq)
        session.commit()
        log_event(
            session,
            run_id,
            "router",
            "loop_decision",
            {"thin_ids": thin, "round": round_, "looping": looping, "supported_counts": counts},
        )
    return "refine" if looping else "contradictions"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("research", research_node)
    g.add_node("draft", draft_node)
    g.add_node("verify", verify_node)
    g.add_node("refine", refine_node)
    g.add_node("contradictions", contradictions_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch_research, ["research"])
    g.add_edge("research", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify, {"refine": "refine", "contradictions": "contradictions"}
    )
    g.add_conditional_edges("refine", dispatch_research, ["research"])
    g.add_edge("contradictions", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
