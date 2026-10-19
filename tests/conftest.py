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
