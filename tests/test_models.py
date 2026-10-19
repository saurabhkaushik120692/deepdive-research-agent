from app.events import log_event
from app.models import Claim, Run, SubQuestion, Verdict


def test_run_roundtrip_with_claim_and_verdict(session):
    run = Run(question="EV battery risks 2026?", status="queued", models={})
    session.add(run)
    session.commit()
    sq = SubQuestion(run_id=run.id, text="Market size?", query="EV battery market size 2026")
    session.add(sq)
    session.commit()
    c = Claim(
        run_id=run.id, sub_question_id=sq.id, text="Demand rose 28%.", chunk_ids=[1, 2], round=0
    )
    session.add(c)
    session.commit()
    session.add(
        Verdict(
            claim_id=c.id,
            label="supported",
            quote="rose 28%",
            quote_found=True,
            reason="ok",
            model="m",
        )
    )
    session.commit()
    got = session.get(Claim, c.id)
    assert got.chunk_ids == [1, 2]


def test_log_event_is_ordered(session):
    run = Run(question="q", status="running", models={})
    session.add(run)
    session.commit()
    e1 = log_event(session, run.id, "planner", "node_started")
    e2 = log_event(session, run.id, "planner", "node_finished", {"n": 3})
    assert e2.id > e1.id and e2.payload == {"n": 3}


def test_make_engine_creates_sqlite_parent_dir(tmp_path):
    from app.db import make_engine

    db_file = tmp_path / "nested" / "deepdive.db"
    make_engine(f"sqlite:///{db_file.as_posix()}")
    assert db_file.parent.is_dir()
