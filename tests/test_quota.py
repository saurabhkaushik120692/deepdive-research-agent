"""Free reports per account: who is limited, what counts, and what the home page shows."""

import pytest
from fastapi.testclient import TestClient

from app.api.quota import limit_message, quota_for
from app.config import Settings
from app.main import create_app
from app.models import Run
from tests.helpers import sign_in

Q = "What is the EV battery market size in 2026?"


async def _fake_run_research(run_id, deps):
    pass


@pytest.fixture
def make_client(deps, monkeypatch):
    monkeypatch.setattr("app.api.runs.run_research", _fake_run_research)

    def _make(**overrides):
        for k, v in {"google_client_id": "cid", "google_client_secret": "s", **overrides}.items():
            setattr(deps.settings, k, v)
        client = TestClient(create_app(deps.settings, deps=deps))
        client.user_id = sign_in(client)
        return client

    return _make


def _seed(deps, user_id, *statuses):
    with deps.session_factory() as s:
        for status in statuses:
            s.add(Run(question=Q, status=status, models={}, user_id=user_id))
        s.commit()


def _start(client):
    return client.post("/api/runs", json={"question": Q})


def test_default_is_five_free_reports():
    assert Settings(_env_file=None).free_reports_per_user == 5


def test_failed_runs_do_not_count(deps, make_client):
    client = make_client()
    _seed(deps, client.user_id, "completed", "running", "queued", "failed", "failed")
    with deps.session_factory() as s:
        quota = quota_for(s, client.user_id, deps.settings)
    assert (quota.used, quota.limit, quota.left, quota.exhausted) == (3, 5, 2, False)


def test_other_users_runs_do_not_count(deps, make_client):
    client = make_client()
    _seed(deps, None, *["completed"] * 5)  # CLI / eval runs have no user
    assert _start(client).status_code == 202


def test_sixth_report_is_refused_with_a_clear_message(deps, make_client):
    client = make_client(contact_email="help@example.com")
    _seed(deps, client.user_id, *["completed"] * 4)
    assert _start(client).status_code == 202
    r = _start(client)
    assert r.status_code == 403
    assert r.json()["detail"] == (
        "You've used your 5 free reports. Paid plans are coming soon. "
        "Need more? Write to help@example.com."
    )


def test_failed_runs_give_the_report_back(deps, make_client):
    client = make_client()
    _seed(deps, client.user_id, *["completed"] * 4, "failed")
    assert _start(client).status_code == 202


def test_limit_comes_from_settings(deps, make_client):
    client = make_client(free_reports_per_user=1)
    assert _start(client).status_code == 202
    assert _start(client).status_code == 403


def test_unlimited_emails_have_no_limit(deps, make_client):
    # sign_in() uses owner@example.com; matching ignores case and spaces
    client = make_client(unlimited_emails=" someone@else.com , Owner@Example.com")
    _seed(deps, client.user_id, *["completed"] * 5)
    assert _start(client).status_code == 202
    with deps.session_factory() as s:
        quota = quota_for(s, client.user_id, deps.settings)
    assert (quota.limit, quota.left, quota.exhausted) == (None, None, False)


def test_message_without_contact_email():
    settings = Settings(_env_file=None, contact_email="", free_reports_per_user=5)
    assert limit_message(settings) == (
        "You've used your 5 free reports. Paid plans are coming soon."
    )


def test_home_page_shows_reports_left(deps, make_client):
    client = make_client()
    _seed(deps, client.user_id, "completed", "completed")
    html = client.get("/").text
    assert "3 of 5 free reports left" in html
    assert 'id="ask-btn" disabled' not in html


def test_home_page_at_the_limit_disables_start(deps, make_client):
    client = make_client()
    _seed(deps, client.user_id, *["completed"] * 5)
    html = client.get("/").text
    assert 'id="ask-btn" disabled' in html
    assert "You&#39;ve used your 5 free reports." in html


def test_home_page_shows_no_counter_for_unlimited_accounts(deps, make_client):
    client = make_client(unlimited_emails="owner@example.com")
    _seed(deps, client.user_id, *["completed"] * 5)
    html = client.get("/").text
    assert "free reports left" not in html
    assert 'id="ask-btn" disabled' not in html


def test_signed_out_home_page_has_no_counter(deps, make_client):
    client = make_client()
    client.post("/auth/logout")
    assert "free reports left" not in client.get("/").text
