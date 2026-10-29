from app.deps import SearchResult
from app.graph.nodes.planner import plan_node, refine_node
from app.graph.nodes.researcher import research_node
from app.graph.schemas import PlannedSubQuestion, PlanOutput, RefineOutput
from app.models import Claim, Verdict
from tests.helpers import all_sources, cfg, chunk_count, event_types, state, sub_questions


async def test_plan_creates_sub_questions(deps, run):
    deps.llm.handlers["planner"] = lambda s, u: PlanOutput(
        sub_questions=[PlannedSubQuestion(text=f"q{i}", query=f"query {i}") for i in range(4)]
    )
    out = await plan_node(state(run), cfg(deps))
    assert len(out["pending_sq_ids"]) == 4
    assert [q.query for q in sub_questions(deps, run.id)] == [f"query {i}" for i in range(4)]
    assert event_types(deps, run.id) >= {"node_started", "sub_questions_planned", "node_finished"}
    role, user = deps.llm.calls[0]
    assert role == "planner" and run.question in user


async def test_research_stores_sources_with_tiers_and_chunks(deps, run, sq):
    deps.search.results[sq.query] = [
        SearchResult("https://www.iea.org/r", "IEA", "word " * 900),
        SearchResult("https://blog.example.com/p", "Blog", None),
    ]
    deps.fetcher.pages["https://blog.example.com/p"] = "blog page " * 300
    out = await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    assert out == {"researched": [sq.id]}
    tiers = sorted(s.tier for s in all_sources(deps, run.id))
    assert tiers == [1, 3] and chunk_count(deps, sq.id) >= 3
    assert len(deps.vectors.store[(run.id, sq.id)]) == chunk_count(deps, sq.id)
    assert "search_done" in event_types(deps, run.id)


async def test_research_survives_all_fetch_failures(deps, run, sq):
    deps.search.results[sq.query] = [SearchResult("https://x.com/a", "A", None)]
    deps.fetcher.pages["https://x.com/a"] = None
    out = await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    assert out == {"researched": [sq.id]} and chunk_count(deps, sq.id) == 0
    assert "source_skipped" in event_types(deps, run.id)


async def test_research_skips_url_already_stored_for_subquestion(deps, run, sq):
    deps.search.results[sq.query] = [SearchResult("https://www.iea.org/r", "IEA", "word " * 100)]
    payload = {"run_id": run.id, "sub_question_id": sq.id, "round": 0}
    await research_node(payload, cfg(deps))
    await research_node({**payload, "round": 1}, cfg(deps))
    assert chunk_count(deps, sq.id) == 1 and len(all_sources(deps, run.id)) == 1


async def test_research_caps_chunks_per_source(deps, run, sq):
    deps.settings.max_chunks_per_source = 2
    deps.search.results[sq.query] = [SearchResult("https://a.org/x", "A", "word " * 5000)]
    await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    assert chunk_count(deps, sq.id) == 2


async def test_refine_updates_query_with_rejection_context(deps, run, sq):
    with deps.session_factory() as s:
        q = s.get(type(sq), sq.id)
        q.status = "thin"
        s.add(q)
        c = Claim(run_id=run.id, sub_question_id=sq.id, text="Market is $5T.", chunk_ids=[1])
        s.add(c)
        s.commit()
        s.add(Verdict(claim_id=c.id, label="unsupported", reason="number not in source"))
        s.commit()
    deps.llm.handlers["planner"] = lambda s, u: RefineOutput(
        query="EV battery market size 2026 IEA"
    )
    out = await refine_node(state(run, round_=0), cfg(deps))
    assert out == {"round": 1, "pending_sq_ids": [sq.id]}
    [q] = sub_questions(deps, run.id)
    assert q.query == "EV battery market size 2026 IEA" and q.round == 1
    _, user = deps.llm.calls[-1]
    assert "Market is $5T." in user and "number not in source" in user
    assert "query_refined" in event_types(deps, run.id)


async def test_research_cleans_links_and_caps_chunk_size(deps, run, sq):
    page = " ".join(f"word{i} [link](https://example.com/{'x' * 60}/{i})" for i in range(800))
    deps.search.results[sq.query] = [SearchResult("https://a.org/x", "A", page)]
    await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    from sqlmodel import select

    from app.models import EvidenceChunk

    with deps.session_factory() as s:
        texts = s.exec(select(EvidenceChunk.text)).all()
    assert texts and all(len(t) <= deps.settings.chunk_max_chars for t in texts)
    assert not any("https://" in t for t in texts) and "word0 link" in texts[0]
