"""Mark runs as public example reports (visible to everyone, listed on the home page).

    python -m app.examples list
    python -m app.examples add <run_id>
    python -m app.examples remove <run_id>

On the server: docker compose -f docker-compose.prod.yml exec app python -m app.examples ..."""

import sys

from sqlmodel import Session, select

from app.config import get_settings
from app.db import init_db, make_engine
from app.models import Run


def main(argv: list[str], session_factory=None) -> int:
    if session_factory is None:
        engine = make_engine(get_settings().database_url)
        init_db(engine)
        session_factory = lambda: Session(engine)  # noqa: E731
    if (
        not argv
        or argv[0] not in ("list", "add", "remove")
        or (argv[0] != "list" and len(argv) != 2)
    ):
        print(__doc__)
        return 2
    with session_factory() as session:
        if argv[0] == "list":
            for run in session.exec(select(Run).where(Run.is_example)).all():
                print(f"{run.id}  {run.status:<9}  {run.question}")
            return 0
        run = session.get(Run, argv[1])
        if run is None:
            print(f"no run with id {argv[1]}")
            return 1
        run.is_example = argv[0] == "add"
        session.add(run)
        session.commit()
        print(f"{'marked' if run.is_example else 'unmarked'} as example: {run.question}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
