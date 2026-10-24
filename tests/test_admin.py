"""Admin area: who is admin, seeing every question, reset, block, and example marking."""

import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlmodel import create_engine, select

from app.db import init_db
from app.main import create_app
from app.models import Run, User
from tests.helpers import sign_in

Q = "What is the EV battery market size in 2026?"
ADMIN = "owner@example.com"  # the email sign_in() uses by default


async def _fake_run_research(run_id, deps):
    pass


@pytest.fixture
def make_client(deps, monkeypatch):
    monkeypatch.setattr("app.api.runs.run_research", _fake_run_research)

    def _make(**overrides):
        settings = {"google_client_id": "cid", "google_client_secret": "s", **overrides}
        for k, v in settings.items():
            setattr(deps.settings, k, v)
        client = TestClient(create_app(deps.settings, deps=deps))
        client.user_id = sign_in(client)
        return client

    return _make


@pytest.fixture
def admin(make_client):
    return make_client(contact_email="Owner@Example.com")  # matching ignores case


def _other_user(client):
    """Create a second account (signs the client in as them), then sign back in as owner."""
    other_id = sign_in(client, sub="google-other", email="other@example.com", name="Other")
    sign_in(client)
    return other_id


def _seed(deps, user_id, *statuses, question=Q):
    with deps.session_factory() as s:
        runs = [Run(question=question, status=st, models={}, user_id=user_id) for st in statuses]
        s.add_all(runs)
        s.commit()
        return [r.id for r in runs]


def _user(deps, user_id):
    with deps.session_factory() as s:
        return s.get(User, user_id)


def _start(client):
    return client.post("/api/runs", json={"question": Q})


# --- who is admin ----------------------------------------------------------------------------


def test_admin_page_is_hidden_from_other_users(make_client):
    client = make_client(contact_email="someone-else@example.com")
    assert client.get("/admin").status_code == 404
    home = client.get("/").text
    assert 'href="/admin"' not in home
    assert 'href="/docs"' not in home


def test_api_docs_link_is_hidden_when_signed_out(admin):
    admin.post("/auth/logout")
    assert 'href="/docs"' not in admin.get("/").text


def test_nobody_is_admin_without_a_contact_email(make_client):
    client = make_client(contact_email="")
    assert client.get("/admin").status_code == 404


def test_admin_page_is_hidden_when_signed_out(admin):
    admin.post("/auth/logout")
    assert admin.get("/admin").status_code == 404


def test_admin_sees_the_admin_link_and_page(admin):
    home = admin.get("/").text
    assert 'href="/admin"' in home
    assert 'href="/docs"' in home
    assert admin.get("/admin").status_code == 200


# --- reports tab -----------------------------------------------------------------------------


def test_reports_tab_lists_every_users_questions(deps, admin):
    other = _other_user(admin)
    _seed(deps, other, "completed", question="Who makes the most solar panels in 2025?")
    _seed(deps, None, "failed", question="An eval question asked from the command line?")
    html = admin.get("/admin").text
    assert "Who makes the most solar panels in 2025?" in html
    assert "other@example.com" in html
    assert "An eval question asked from the command line?" in html


def test_reports_tab_search_matches_question_or_email(deps, admin):
    other = _other_user(admin)
    _seed(deps, other, "completed", question="Who makes the most solar panels in 2025?")
    _seed(deps, admin.user_id, "completed", question="How big is the lithium market?")
    by_email = admin.get("/admin", params={"q": "OTHER@"}).text
    assert "solar panels" in by_email and "lithium" not in by_email
    by_question = admin.get("/admin", params={"q": "lithium"}).text
    assert "lithium" in by_question and "solar panels" not in by_question


def test_admin_can_open_any_users_report(deps, admin):
    other = _other_user(admin)
    (run_id,) = _seed(deps, other, "completed")
    assert admin.get(f"/runs/{run_id}").status_code == 200
    assert admin.get(f"/api/runs/{run_id}").status_code == 200
    assert admin.get(f"/runs/{run_id}/trail").status_code == 200


def test_summary_counts(deps, admin):
    other = _other_user(admin)
    _seed(deps, other, "completed", "failed")
    html = admin.get("/admin").text
    assert "2 users" in html and "2 reports" in html and "1 failed this week" in html


# --- users tab, reset, block -----------------------------------------------------------------


def test_users_tab_shows_reports_used(deps, admin):
    other = _other_user(admin)
    _seed(deps, other, "completed", "completed", "failed")
    html = admin.get("/admin", params={"tab": "users"}).text
    assert "other@example.com" in html
    assert "2 / 5" in html


def test_reset_lets_the_user_start_again(deps, admin):
    other = _other_user(admin)
    old_ids = _seed(deps, other, *["completed"] * 5)
    r = admin.post(f"/admin/users/{other}/reset", follow_redirects=False)
    assert r.status_code == 303
    assert _user(deps, other).quota_reset_at is not None
    sign_in(admin, sub="google-other", email="other@example.com", name="Other")
    assert "5 of 5 free reports left" in admin.get("/").text
    assert _start(admin).status_code == 202
    assert admin.get(f"/runs/{old_ids[0]}").status_code == 200  # old reports stay visible


def test_blocked_user_keeps_old_reports_but_cannot_start(deps, admin):
    other = _other_user(admin)
    (run_id,) = _seed(deps, other, "completed")
    assert admin.post(f"/admin/users/{other}/block", follow_redirects=False).status_code == 303
    sign_in(admin, sub="google-other", email="other@example.com", name="Other")
    home = admin.get("/").text
    assert 'id="ask-btn" disabled' in home
    assert "Your account can&#39;t start new reports." in home
    r = _start(admin)
    assert r.status_code == 403
    assert r.json()["detail"].startswith("Your account can't start new reports.")
    assert admin.get(f"/runs/{run_id}").status_code == 200


def test_unblock_restores_the_user(deps, admin):
    other = _other_user(admin)
    admin.post(f"/admin/users/{other}/block")
    admin.post(f"/admin/users/{other}/unblock")
    assert not _user(deps, other).blocked
    sign_in(admin, sub="google-other", email="other@example.com", name="Other")
    assert _start(admin).status_code == 202


def test_admin_cannot_block_themselves(deps, admin):
    assert admin.post(f"/admin/users/{admin.user_id}/block").status_code == 400
    assert not _user(deps, admin.user_id).blocked
    html = admin.get("/admin", params={"tab": "users"}).text
    assert f'action="/admin/users/{admin.user_id}/block"' not in html


def test_admin_has_no_report_limit(deps, admin):
    _seed(deps, admin.user_id, *["completed"] * 5)
    assert _start(admin).status_code == 202
    assert "free reports left" not in admin.get("/").text


def test_non_admins_cannot_use_admin_actions(deps, make_client):
    client = make_client(contact_email="someone-else@example.com")
    other = _other_user(client)
    (run_id,) = _seed(deps, other, "completed")
    assert client.post(f"/admin/users/{other}/block").status_code == 404
    assert client.post(f"/admin/users/{other}/reset").status_code == 404
    assert client.post(f"/admin/runs/{run_id}/example", data={"make": "1"}).status_code == 404
    assert not _user(deps, other).blocked
    with deps.session_factory() as s:
        assert not s.get(Run, run_id).is_example


def test_unknown_user_or_run_is_404(admin):
    assert admin.post("/admin/users/999/block").status_code == 404
    assert admin.post("/admin/runs/nope/example", data={"make": "1"}).status_code == 404


# --- examples --------------------------------------------------------------------------------


def test_marking_an_example_makes_it_public(deps, admin):
    other = _other_user(admin)
    (run_id,) = _seed(deps, other, "completed")
    r = admin.post(f"/admin/runs/{run_id}/example", data={"make": "1"}, follow_redirects=False)
    assert r.status_code == 303
    admin.post("/auth/logout")
    assert admin.get(f"/runs/{run_id}").status_code == 200  # signed out, still visible
    sign_in(admin)
    admin.post(f"/admin/runs/{run_id}/example", data={"make": "0"})
    with deps.session_factory() as s:
        assert not s.get(Run, run_id).is_example


def test_actions_return_to_a_safe_page_only(deps, admin):
    other = _other_user(admin)
    ok = admin.post(
        f"/admin/users/{other}/reset", data={"back": "/admin?tab=users&q=x"}, follow_redirects=False
    )
    assert ok.headers["location"] == "/admin?tab=users&q=x"
    bad = admin.post(
        f"/admin/users/{other}/reset", data={"back": "https://evil.example"}, follow_redirects=False
    )
    assert bad.headers["location"] == "/admin"


# --- existing databases ----------------------------------------------------------------------


def test_init_db_adds_admin_columns_to_an_old_user_table(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE user (id INTEGER PRIMARY KEY, google_sub VARCHAR NOT NULL, email VARCHAR"
        " NOT NULL, name VARCHAR, picture VARCHAR, created_at DATETIME, last_login_at DATETIME)"
    )
    con.execute("INSERT INTO user (id, google_sub, email) VALUES (1, 'g1', 'a@b.c')")
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    init_db(engine)
    init_db(engine)  # idempotent
    cols = {c["name"] for c in inspect(engine).get_columns("user")}
    assert {"blocked", "quota_reset_at"} <= cols
    from sqlmodel import Session

    with Session(engine) as s:
        user = s.exec(select(User)).one()
    assert (user.blocked, user.quota_reset_at) == (False, None)


# --- featured report on the home page -------------------------------------------------------


def _feature(client, run_id, make="1"):
    return client.post(f"/admin/runs/{run_id}/feature", data={"make": make}, follow_redirects=False)


def _flags(deps, run_id):
    with deps.session_factory() as s:
        run = s.get(Run, run_id)
        return run.is_example, run.is_featured


def test_featuring_picks_one_example_for_the_home_page(deps, admin):
    a, b = _seed(deps, None, "completed", "completed")
    for run_id in (a, b):
        admin.post(f"/admin/runs/{run_id}/example", data={"make": "1"})
    assert _feature(admin, a).status_code == 303
    assert _flags(deps, a) == (True, True)
    _feature(admin, b)
    assert _flags(deps, a) == (True, False)  # only one featured at a time
    assert _flags(deps, b) == (True, True)
    _feature(admin, b, make="0")
    assert _flags(deps, b) == (True, False)


def test_only_example_reports_can_be_featured(deps, admin):
    (run_id,) = _seed(deps, None, "completed")
    assert _feature(admin, run_id).status_code == 400
    assert _flags(deps, run_id) == (False, False)


def test_removing_an_example_also_removes_it_from_the_home_page(deps, admin):
    (run_id,) = _seed(deps, None, "completed")
    admin.post(f"/admin/runs/{run_id}/example", data={"make": "1"})
    _feature(admin, run_id)
    admin.post(f"/admin/runs/{run_id}/example", data={"make": "0"})
    assert _flags(deps, run_id) == (False, False)


def test_non_admins_cannot_feature(deps, make_client):
    client = make_client(contact_email="someone-else@example.com")
    (run_id,) = _seed(deps, None, "completed")
    with deps.session_factory() as s:
        run = s.get(Run, run_id)
        run.is_example = True
        s.add(run)
        s.commit()
    assert _feature(client, run_id).status_code == 404
    assert _flags(deps, run_id) == (True, False)


def test_reports_tab_offers_the_home_page_toggle_on_examples_only(deps, admin):
    example, plain = _seed(deps, None, "completed", "completed")
    admin.post(f"/admin/runs/{example}/example", data={"make": "1"})
    html = admin.get("/admin").text
    assert f'action="/admin/runs/{example}/feature"' in html
    assert f'action="/admin/runs/{plain}/feature"' not in html
