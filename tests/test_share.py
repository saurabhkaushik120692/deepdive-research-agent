"""Share link: the owner can let anyone with the link read a finished report."""

import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlmodel import create_engine

from app.db import init_db
from app.main import create_app
from app.models import FollowUp, Run
from tests.helpers import sign_in
from tests.test_followups import seed_report


@pytest.fixture
def owner(deps):
    deps.settings.google_client_id, deps.settings.google_client_secret = "cid", "s"
    client = TestClient(create_app(deps.settings, deps=deps))
    client.user_id = sign_in(client)
    return client


def _visitor(owner):
    public = TestClient(owner.app)
    public.cookies.clear()
    return public


def _share(client, run_id, make="1"):
    return client.post(f"/runs/{run_id}/share", data={"make": make}, follow_redirects=False)


def test_private_report_needs_sign_in(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    r = _visitor(owner).get(f"/runs/{ids['run']}/report", follow_redirects=False)
    assert r.status_code == 303 and "/auth/login" in r.headers["location"]


def test_shared_report_opens_without_sign_in(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    r = _share(owner, ids["run"])
    assert r.status_code == 303 and r.headers["location"] == f"/runs/{ids['run']}/report"
    visitor = _visitor(owner)
    page = visitor.get(f"/runs/{ids['run']}/report")
    assert page.status_code == 200 and "RBI repo rate in 2025" in page.text
    assert visitor.get(f"/runs/{ids['run']}/trail").status_code == 200
    assert "Stop sharing" not in page.text  # only the owner can change it


def test_stop_sharing_makes_it_private_again(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    _share(owner, ids["run"])
    _share(owner, ids["run"], make="0")
    r = _visitor(owner).get(f"/runs/{ids['run']}/report", follow_redirects=False)
    assert r.status_code == 303


def test_owner_sees_share_controls(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    page = owner.get(f"/runs/{ids['run']}/report").text
    assert "Share link" in page and "Stop sharing" not in page
    _share(owner, ids["run"])
    page = owner.get(f"/runs/{ids['run']}/report").text
    assert "Stop sharing" in page and "Copy link" in page


def test_only_the_owner_can_share(deps, owner):
    ids = seed_report(deps)  # someone else's report
    assert _share(owner, ids["run"]).status_code == 404
    with deps.session_factory() as s:
        assert s.get(Run, ids["run"]).is_shared is False


def test_unfinished_report_cannot_be_shared(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id, status="running")
    assert _share(owner, ids["run"]).status_code == 400


def test_shared_link_hides_followups(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    with deps.session_factory() as s:
        s.add(
            FollowUp(
                run_id=ids["run"],
                user_id=owner.user_id,
                question="A private follow-up question?",
                status="declined",
            )
        )
        s.commit()
    _share(owner, ids["run"])
    visitor = _visitor(owner)
    page = visitor.get(f"/runs/{ids['run']}/report").text
    assert "A private follow-up question?" not in page and 'id="followups"' not in page
    assert visitor.get(f"/api/runs/{ids['run']}/trail").json()["followups"] == []


def test_is_shared_column_migrates(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE run (id VARCHAR PRIMARY KEY, question VARCHAR NOT NULL)")
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    init_db(engine)
    assert "is_shared" in {c["name"] for c in inspect(engine).get_columns("run")}
