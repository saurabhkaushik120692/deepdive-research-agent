from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app import models  # noqa: F401  (registers tables on SQLModel.metadata)

_SQLITE_FILE_PREFIX = "sqlite:///"


def make_engine(database_url: str) -> Engine:
    if database_url.startswith(_SQLITE_FILE_PREFIX):
        db_path = Path(database_url.removeprefix(_SQLITE_FILE_PREFIX))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency; the engine is created once in create_app()."""
    with Session(request.app.state.engine) as session:
        yield session
