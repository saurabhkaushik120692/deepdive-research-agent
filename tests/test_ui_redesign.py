"""The three-column UI: home page numbers, run page with report, quote popovers, live feed."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import select

import app.api.pages as pages
from app.api.pages import time_ago
from app.events import log_event
from app.models import Claim, Run
from tests.test_pages import client, make_client, seeded_run  # noqa: F401  (fixtures)

EVAL = {
    "generated_at": "2026-10-21 16:48 UTC",
    "n_questions": 24,
    "runs": {"total": 24, "completed": 24},
    "verifier_probe": {
        "catch_rate": 0.945,
        "n_mutants": 55,
        "wrongful_rejection_rate": 0.0,
        "n_originals": 60,
        "catch_rate_by_kind": {},
        "n_mutants_by_kind": {},
    },
    "independent_judge": {"model": "google/x", "n": 100, "groundedness": 0.94},
    "gold": {"n_labelled": 50, "accuracy": 0.86, "kappa": 0.70},
    "cost": {"per_run_usd": {"mean": 0.0398}},
    "latency_s": {"p50": 250.7},
}


@pytest.fixture
def eval_results(tmp_path, monkeypatch):
    (tmp_path / "20261021T165231Z.json").write_text(json.dumps(EVAL))
    monkeypatch.setattr(pages, "EVAL_RESULTS", tmp_path)


def test_home_shows_how_it_works_examples_and_eval_numbers(client, eval_results):  # noqa: F811
    html = client.get("/").text
    assert "How it works" in html and 'name="question"' in html
    assert html.count("data-example=") >= 3
    assert "94.5%" in html and "0.0%" in html and "94.0%" in html and "86.0%" in html
    assert 'href="/eval"' in html


def test_home_without_eval_results_still_renders(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pages, "EVAL_RESULTS", tmp_path)
    html = client.get("/").text
    assert 'name="question"' in html and "How it works" in html


def test_home_lists_recent_runs_with_relative_time(client, seeded_run):  # noqa: F811
    html = client.get("/").text
    assert f'href="/runs/{seeded_run.id}"' in html and "just now" in html


@pytest.mark.parametrize(
    "delta, expected",
    [
        (timedelta(seconds=20), "just now"),
        (timedelta(seconds=90), "1 min ago"),
        (timedelta(hours=3, minutes=5), "3 h ago"),
        (timedelta(days=1, hours=2), "1 day ago"),
        (timedelta(days=4), "4 days ago"),
    ],
)
def test_time_ago(delta, expected):
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    assert time_ago(now - delta, now) == expected


def test_time_ago_treats_naive_timestamps_as_utc():
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    assert time_ago(datetime(2026, 10, 22, 11, 0), now) == "1 h ago"


def test_completed_run_page_shows_report_and_collapsed_timeline(client, seeded_run):  # noqa: F811
    html = client.get(f"/runs/{seeded_run.id}").text
    assert "EV battery risks" in html  # the report title, rendered server-side
    assert 'id="live"' in html and 'data-stage="verify"' in html
    assert '<details id="live" open' not in html  # collapsed under the report
    assert 'id="replay"' in html


def test_running_run_page_has_open_live_view_and_no_report(client, deps):  # noqa: F811
    with deps.session_factory() as s:
        run = Run(
            question="Still researching?", status="running", models={}, user_id=client.user_id
        )
        s.add(run)
        s.commit()
        s.refresh(run)
    html = client.get(f"/runs/{run.id}").text
    assert '<details id="live" open' in html and 'class="report"' not in html


def test_report_has_quote_popover_contents_and_summary_card(client, seeded_run):  # noqa: F811
    html = client.get(f"/runs/{seeded_run.id}/report").text
    assert 'class="cite-pop"' in html and "iea.org" in html and "official" in html
    assert 'href="#sec-1"' in html and 'id="sec-1"' in html
    assert 'href="#rejected"' in html and 'id="rejected"' in html
    assert "1 of 3" in html  # supported of drafted, on the summary card
    assert 'class="report-side' in html


def test_event_stream_adds_claim_text_to_verdicts(client, deps, seeded_run):  # noqa: F811
    with deps.session_factory() as s:
        claim = s.exec(select(Claim).where(Claim.run_id == seeded_run.id)).first()
        log_event(
            s,
            seeded_run.id,
            "verifier",
            "claim_verified",
            {"claim_id": claim.id, "label": "supported", "reason": "ok"},
        )
        text = claim.text
    body = client.get(f"/api/runs/{seeded_run.id}/events").text
    assert json.dumps(text)[1:-1] in body
