import pytest
from fastapi.testclient import TestClient

from app.events import log_event
from app.main import create_app
from app.models import Run

Q = "What is the EV battery market size in 2026?"


async def _fake_run_research(run_id, deps):
    with deps.session_factory() as s:
        log_event(s, run_id, "planner", "node_started")
        log_event(s, run_id, "planner", "node_finished")
        run = s.get(Run, run_id)
        run.status = "completed"
        s.add(run)
        s.commit()


@pytest.fixture
def make_client(deps, monkeypatch):
    monkeypatch.setattr("app.api.runs.run_research", _fake_run_research)

    def _make(**settings_overrides):
        for k, v in settings_overrides.items():
            setattr(deps.settings, k, v)
        return TestClient(create_app(deps.settings, deps=deps))

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def client_with_code(make_client):
    return make_client(demo_access_code="secret")


@pytest.fixture
def client_rate_1(make_client):
    return make_client(runs_per_hour=1)


def _seed_run(deps, status, error=None, n_events=2):
    with deps.session_factory() as s:
        run = Run(question=Q, status=status, error=error, models={})
        s.add(run)
        s.commit()
        s.refresh(run)
        ids = []
        for i in range(n_events):
            ids.append(log_event(s, run.id, "planner", "node_started" if i == 0 else "x").id)
        return run.id, ids


@pytest.fixture
def finished_run_with_events(deps):
    return _seed_run(deps, "completed")[0]


@pytest.fixture
def first_event_id(deps, finished_run_with_events):
    from tests.helpers import events

    return events(deps, finished_run_with_events)[0].id


@pytest.fixture
def failed_run(deps):
    return _seed_run(deps, "failed", error="RuntimeError: boom")[0]


def test_gate_blocks_without_cookie(client_with_code):
    assert client_with_code.post("/api/runs", json={"question": Q}).status_code == 401


def test_access_sets_cookie_and_unlocks(client_with_code):
    bad = client_with_code.post("/access", data={"code": "nope"}, follow_redirects=False)
    assert bad.status_code == 401
    ok = client_with_code.post("/access", data={"code": "secret"}, follow_redirects=False)
    assert ok.status_code == 303 and "dd_access" in ok.cookies
    assert client_with_code.post("/api/runs", json={"question": Q}).status_code == 202


def test_create_run_returns_202(client, deps):
    r = client.post("/api/runs", json={"question": Q})
    assert r.status_code == 202 and r.json()["run_id"]
    with deps.session_factory() as s:
        assert s.get(Run, r.json()["run_id"]).question == Q


def test_question_length_validated(client):
    assert client.post("/api/runs", json={"question": "short"}).status_code == 422


def test_moderation_rejects(client, deps):
    deps.llm.flagged = True
    r = client.post("/api/runs", json={"question": "something harmful here please"})
    assert r.status_code == 400 and r.json()["detail"] == "question rejected by safety screen"


def test_rate_limit(client_rate_1):
    assert client_rate_1.post("/api/runs", json={"question": Q}).status_code == 202
    assert client_rate_1.post("/api/runs", json={"question": Q + " again"}).status_code == 429


def test_sse_replays_finished_run_then_done(client, finished_run_with_events):
    with client.stream("GET", f"/api/runs/{finished_run_with_events}/events") as r:
        body = "".join(r.iter_text())
    assert body.index("event: node_started") < body.index("event: done")


def test_sse_resumes_after_last_event_id(client, finished_run_with_events, first_event_id):
    headers = {"Last-Event-ID": str(first_event_id)}
    with client.stream("GET", f"/api/runs/{finished_run_with_events}/events", headers=headers) as r:
        body = "".join(r.iter_text())
    assert f"id: {first_event_id}\r\n" not in body and f"id: {first_event_id}\n" not in body
    assert "event: done" in body


def test_failed_run_status_exposes_error(client, failed_run):
    body = client.get(f"/api/runs/{failed_run}").json()
    assert body["status"] == "failed" and "boom" in body["error"]


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_trail_shape(client, finished_run_with_events):
    t = client.get(f"/api/runs/{finished_run_with_events}/trail").json()
    assert set(t) == {"run", "sub_questions", "searches", "sources", "claims", "events"}
    assert t["run"]["id"] == finished_run_with_events and len(t["events"]) == 2


def test_rate_limit_ignores_runs_older_than_an_hour(client_rate_1, deps):
    from datetime import UTC, datetime, timedelta

    with deps.session_factory() as s:
        s.add(Run(question=Q, models={}, created_at=datetime.now(UTC) - timedelta(hours=2)))
        s.commit()
    assert client_rate_1.post("/api/runs", json={"question": Q}).status_code == 202


def test_non_ascii_access_code_is_rejected_cleanly(client_with_code):
    r = client_with_code.post("/access", data={"code": "pässwörd-ü"}, follow_redirects=False)
    assert r.status_code == 401


def test_access_cookie_is_secure_behind_https_proxy(client_with_code):
    r = client_with_code.post(
        "/access",
        data={"code": "secret"},
        headers={"X-Forwarded-Proto": "https"},
        follow_redirects=False,
    )
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=lax" in cookie


def test_access_cookie_not_secure_on_plain_http_dev(client_with_code):
    r = client_with_code.post("/access", data={"code": "secret"}, follow_redirects=False)
    assert "secure" not in r.headers["set-cookie"].lower()
