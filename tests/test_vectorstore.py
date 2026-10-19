from sqlmodel import select

from app.models import Event, Run
from app.tools.vectorstore import ChromaStore


async def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.05] if "battery" in t else [0.05, 1.0] for t in texts]


async def test_query_filters_by_run_and_subq(tmp_path):
    store = ChromaStore(tmp_path, embed=fake_embed)
    await store.add("r1", 1, [10, 11], ["battery prices fell", "football scores"])
    await store.add("r2", 1, [20], ["battery prices fell"])
    await store.add("r1", 2, [30], ["battery recycling"])
    assert await store.query("r1", 1, "battery cost", k=1) == [10]
    got = await store.query("r1", 1, "battery cost", k=5)
    assert set(got) == {10, 11} and 20 not in got and 30 not in got


async def test_query_with_no_chunks_returns_empty(tmp_path):
    store = ChromaStore(tmp_path, embed=fake_embed)
    assert await store.query("nope", 1, "battery", k=3) == []


async def test_build_deps_logs_llm_usage_as_events(tmp_path):
    from app.config import Settings
    from app.deps import build_deps

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'dd.db').as_posix()}",
    )
    deps = build_deps(settings)
    with deps.session_factory() as s:
        run = Run(question="q", models={})
        s.add(run)
        s.commit()
        run_id = run.id
    deps.llm.on_usage(run_id, {"role": "planner", "prompt_tokens": 5})
    deps.llm.on_usage(None, {"role": "planner"})  # no run → not logged
    with deps.session_factory() as s:
        events = s.exec(select(Event).where(Event.type == "llm_call")).all()
    assert len(events) == 1 and events[0].payload["prompt_tokens"] == 5
