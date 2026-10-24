"""Google sign-in, sessions, and who may see which run."""

import sqlite3

import pytest
from authlib.integrations.starlette_client import OAuthError
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlmodel import create_engine, select

from app.db import init_db
from app.events import log_event
from app.main import create_app
from app.models import Run, User
from tests.helpers import FakeGoogle, sign_in

Q = "What is the EV battery market size in 2026?"


async def _noop_research(run_id, deps):
    return None


@pytest.fixture
def make_client(deps, monkeypatch):
    monkeypatch.setattr("app.api.runs.run_research", _noop_research)

    def _make(**overrides):
        for k, v in overrides.items():
            setattr(deps.settings, k, v)
        return TestClient(create_app(deps.settings, deps=deps))

    return _make


@pytest.fixture
def client(make_client):
    return make_client(google_client_id="cid", google_client_secret="csecret")


def _run(deps, user_id=None, example=False, question=Q):
    with deps.session_factory() as s:
        run = Run(question=question, status="completed", models={}, user_id=user_id)
        run.is_example = example
        s.add(run)
        s.commit()
        s.refresh(run)
        log_event(s, run.id, "planner", "node_started")
        return run.id


# --- signing in and out -------------------------------------------------------------------


def test_login_redirects_to_google_with_our_callback(client):
    client.app.state.google = google = FakeGoogle()
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("https://accounts.google.com")
    assert google.redirect_uris == ["http://testserver/auth/callback"]


def test_callback_uses_https_site_host_when_deployed(make_client):
    client = make_client(google_client_id="cid", google_client_secret="s", site_host="dd.example")
    client.app.state.google = google = FakeGoogle()
    client.get("/auth/login", follow_redirects=False)
    assert google.redirect_uris == ["https://dd.example/auth/callback"]


def test_login_without_google_config_is_503(make_client):
    assert make_client().get("/auth/login").status_code == 503


def test_first_sign_in_creates_user_and_shows_name(client, deps):
    user_id = sign_in(client, name="Asha Rao")
    with deps.session_factory() as s:
        user = s.get(User, user_id)
        assert user.email == "owner@example.com" and user.name == "Asha Rao"
    html = client.get("/").text
    assert "Asha Rao" in html and 'action="/auth/logout"' in html


def test_second_sign_in_reuses_the_account(client, deps):
    first = sign_in(client, name="Old name")
    second = sign_in(client, name="New name")
    with deps.session_factory() as s:
        assert first == second and len(s.exec(select(User)).all()) == 1
        assert s.get(User, first).name == "New name"


def test_unverified_email_is_refused(client):
    client.app.state.google = FakeGoogle(
        {"sub": "x", "email": "x@example.com", "email_verified": False, "name": "X"}
    )
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?signin=failed"
    assert client.post("/api/runs", json={"question": Q}).status_code == 401


def test_cancelled_google_sign_in_returns_home(client):
    client.app.state.google = FakeGoogle(error=OAuthError(error="access_denied"))
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?signin=failed"


def test_sign_in_returns_to_the_page_it_started_from(client):
    client.app.state.google = FakeGoogle()
    client.get("/auth/login?next=/runs/abc", follow_redirects=False)
    client.app.state.google = FakeGoogle(
        {"sub": "s", "email": "s@example.com", "email_verified": True, "name": "S"}
    )
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.headers["location"] == "/runs/abc"


@pytest.mark.parametrize("bad", ["//evil.example", "https://evil.example/x", "evil", "/\\evil"])
def test_next_cannot_point_off_site(client, bad):
    client.app.state.google = FakeGoogle()
    client.get("/auth/login", params={"next": bad}, follow_redirects=False)
    client.app.state.google = FakeGoogle(
        {"sub": "s", "email": "s@example.com", "email_verified": True, "name": "S"}
    )
    assert client.get("/auth/callback", follow_redirects=False).headers["location"] == "/"


def test_logout_ends_the_session(client):
    sign_in(client)
    r = client.post("/auth/logout", follow_redirects=False)
    assert r.status_code == 303
    assert client.post("/api/runs", json={"question": Q}).status_code == 401


def test_session_cookie_is_secure_and_httponly_when_deployed(make_client):
    client = make_client(google_client_id="cid", google_client_secret="s", site_host="dd.example")
    client.app.state.google = FakeGoogle(
        {"sub": "s", "email": "s@example.com", "email_verified": True, "name": "S"}
    )
    cookie = client.get("/auth/callback", follow_redirects=False).headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=lax" in cookie


# --- starting runs ------------------------------------------------------------------------


def test_starting_a_run_requires_sign_in(client):
    assert client.post("/api/runs", json={"question": Q}).status_code == 401


def test_run_belongs_to_whoever_started_it(client, deps):
    user_id = sign_in(client)
    run_id = client.post("/api/runs", json={"question": Q}).json()["run_id"]
    with deps.session_factory() as s:
        assert s.get(Run, run_id).user_id == user_id


# --- who sees which run -------------------------------------------------------------------

VIEWS = ["/runs/{id}", "/runs/{id}/report", "/runs/{id}/trail"]
API_VIEWS = ["/api/runs/{id}", "/api/runs/{id}/trail", "/api/runs/{id}/events"]


@pytest.mark.parametrize("path", VIEWS + API_VIEWS)
def test_owner_sees_their_run(client, deps, path):
    run_id = _run(deps, user_id=sign_in(client))
    assert client.get(path.format(id=run_id)).status_code == 200


@pytest.mark.parametrize("path", VIEWS + API_VIEWS)
def test_other_users_get_404(client, deps, path):
    run_id = _run(deps, user_id=sign_in(client, sub="alice", email="a@example.com"))
    sign_in(client, sub="bob", email="b@example.com")
    assert client.get(path.format(id=run_id), follow_redirects=False).status_code == 404


@pytest.mark.parametrize("path", VIEWS)
def test_signed_out_visitors_are_sent_to_sign_in(client, deps, path):
    run_id = _run(deps, user_id=sign_in(client))
    client.post("/auth/logout")
    r = client.get(path.format(id=run_id), follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/auth/login?next=/runs/")


@pytest.mark.parametrize("path", API_VIEWS)
def test_signed_out_api_gets_404_for_private_runs(client, deps, path):
    run_id = _run(deps, user_id=sign_in(client))
    client.post("/auth/logout")
    assert client.get(path.format(id=run_id)).status_code == 404


@pytest.mark.parametrize("path", VIEWS + API_VIEWS)
def test_example_reports_are_public(client, deps, path):
    run_id = _run(deps, example=True)
    assert client.get(path.format(id=run_id)).status_code == 200


def test_home_signed_out_shows_examples_and_google_button(client, deps):
    _run(deps, example=True, question="Example: RBI repo rate in 2025?")
    _run(deps, user_id=None, question="Someone's private question?")
    html = client.get("/").text
    assert ">Examples</button>" in html and "Example: RBI repo rate in 2025?" in html
    assert "Someone&#39;s private question?" not in html and "private question" not in html
    assert "Continue with Google" in html and 'href="/auth/login' in html


def test_home_signed_in_lists_only_your_runs(client, deps):
    mine = sign_in(client)
    _run(deps, user_id=mine, question="My own question about lithium?")
    _run(deps, user_id=None, question="A stranger's question about gold?")
    html = client.get("/").text
    assert "Your runs" in html and "My own question about lithium?" in html
    assert "question about gold" not in html


# --- existing databases and example marking -----------------------------------------------


def test_init_db_adds_owner_columns_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE run (id VARCHAR PRIMARY KEY, question VARCHAR NOT NULL, status VARCHAR,"
        " models JSON, error VARCHAR, created_at DATETIME, started_at DATETIME,"
        " finished_at DATETIME, prompt_tokens INTEGER, completion_tokens INTEGER)"
    )
    con.execute(
        "INSERT INTO run (id, question, status) VALUES ('r1', 'old question?', 'completed')"
    )
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    init_db(engine)
    init_db(engine)  # idempotent
    cols = {c["name"] for c in inspect(engine).get_columns("run")}
    assert {"user_id", "is_example", "is_featured"} <= cols
    with engine.connect() as c:
        assert c.exec_driver_sql("SELECT is_example, user_id FROM run").one() == (0, None)


def test_examples_command_marks_and_unmarks(deps, capsys):
    from app.examples import main

    run_id = _run(deps)
    factory = deps.session_factory
    assert main(["add", run_id], session_factory=factory) == 0
    with factory() as s:
        assert s.get(Run, run_id).is_example
    main(["list"], session_factory=factory)
    assert run_id in capsys.readouterr().out
    assert main(["remove", run_id], session_factory=factory) == 0
    with factory() as s:
        assert not s.get(Run, run_id).is_example
    assert main(["add", "missing"], session_factory=factory) == 1


# --- privacy and terms --------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, heading", [("/privacy", "Privacy policy"), ("/terms", "Terms of service")]
)
def test_legal_pages_are_public(client, path, heading):
    r = client.get(path)
    assert r.status_code == 200 and heading in r.text


def test_privacy_names_what_is_stored_and_who_processes_it(client):
    html = client.get("/privacy").text
    for fact in ["email", "Google", "Mesh", "Tavily", "Amazon Web Services", "cookie"]:
        assert fact in html


def test_legal_pages_show_contact_email_when_set(make_client):
    client = make_client(contact_email="help@deepdive.example")
    assert "help@deepdive.example" in client.get("/privacy").text
    assert "help@deepdive.example" in client.get("/terms").text


def test_every_page_links_privacy_and_terms(client):
    html = client.get("/").text
    assert 'href="/privacy"' in html and 'href="/terms"' in html
