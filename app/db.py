from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app import models  # noqa: F401  (registers tables on SQLModel.metadata)

_SQLITE_FILE_PREFIX = "sqlite:///"


def make_engine(database_url: str) -> Engine:
    if database_url.startswith(_SQLITE_FILE_PREFIX):
        db_path = Path(database_url.removeprefix(_SQLITE_FILE_PREFIX))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, connect_args={"check_same_thread": False})


# Columns added after the first release; create_all() doesn't alter existing tables.
_ADDED_COLUMNS = {
    "run": {
        "user_id": "INTEGER REFERENCES user (id)",
        "is_example": "BOOLEAN NOT NULL DEFAULT 0",
    },
    "user": {
        "blocked": "BOOLEAN NOT NULL DEFAULT 0",
        "quota_reset_at": "DATETIME",
    },
}


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            have = {c["name"] for c in inspect(conn).get_columns(table)}
            for name, ddl in columns.items():
                if name not in have:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_run_user_id ON run (user_id)")


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency; the engine is created once by build_deps() in create_app()."""
    with request.app.state.deps.session_factory() as session:
        yield session
