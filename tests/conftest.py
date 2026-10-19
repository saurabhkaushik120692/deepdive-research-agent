import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from app.db import init_db


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Keep tests independent of a developer's local .env and data folder."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def settings():
    from app.config import Settings

    return Settings(_env_file=None, mesh_api_key="rsk_test", tavily_api_key="tvly_test")


@pytest.fixture
def deps(settings, engine):
    from sqlmodel import Session as _Session

    from app.deps import Deps
    from tests.fakes import FakeFetcher, FakeSearch, MemoryVectors, ScriptedLlm

    return Deps(
        settings=settings,
        llm=ScriptedLlm(),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
        vectors=MemoryVectors(),
        session_factory=lambda: _Session(engine),
    )


@pytest.fixture
def session_factory(engine):
    from sqlmodel import Session as _Session

    return lambda: _Session(engine)


@pytest.fixture
def run(deps):
    from app.models import Run

    with deps.session_factory() as s:
        r = Run(question="What are the main risks of investing in EV batteries in 2026?", models={})
        s.add(r)
        s.commit()
        s.refresh(r)
        return r


@pytest.fixture
def sq(deps, run):
    from app.models import SubQuestion

    with deps.session_factory() as s:
        q = SubQuestion(
            run_id=run.id, text="How big is the market?", query="EV battery market size"
        )
        s.add(q)
        s.commit()
        s.refresh(q)
        return q
