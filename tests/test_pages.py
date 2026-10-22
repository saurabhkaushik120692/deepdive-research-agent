from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.events import log_event
from app.graph.schemas import Paragraph, Section, SynthesisOutput
from app.main import create_app
from app.models import Claim, EvidenceChunk, Report, Run, Source, SubQuestion, Verdict
from tests.helpers import sign_in

SUPPORTED_QUOTE = "global EV battery demand rose 28% to 1.1 TWh"
REJECTED_REASON = "the $5T figure does not appear in the source"


@pytest.fixture
def make_client(deps):
    def _make(**overrides):
        overrides = {"google_client_id": "cid", "google_client_secret": "s", **overrides}
        for k, v in overrides.items():
            setattr(deps.settings, k, v)
        client = TestClient(create_app(deps.settings, deps=deps))
        client.user_id = sign_in(client)  # pages here are viewed by the run's owner
        return client

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def seeded_run(deps, client):
    with deps.session_factory() as s:
        run = Run(
            question="EV battery risks 2026?", status="completed", models={}, user_id=client.user_id
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        sq = SubQuestion(run_id=run.id, text="Market size?", query="q", status="ok")
        s.add(sq)
        s.commit()
        s.refresh(sq)
        iea = Source(
            run_id=run.id, url="https://www.iea.org/r", domain="iea.org", title="IEA", tier=1
        )
        s.add(iea)
        s.commit()
        s.refresh(iea)
        ch = EvidenceChunk(source_id=iea.id, run_id=run.id, sub_question_id=sq.id, text="x")
        s.add(ch)
        s.commit()
        s.refresh(ch)
        texts = [
            ("Demand rose 28%.", "supported"),
            ("Prices fell.", "partial"),
            ("$5T.", "unsupported"),
        ]
        ids = []
        for text, label in texts:
            c = Claim(run_id=run.id, sub_question_id=sq.id, text=text, chunk_ids=[ch.id])
            s.add(c)
            s.commit()
            s.refresh(c)
            ids.append(c.id)
            quote = SUPPORTED_QUOTE if label == "supported" else ""
            reason = REJECTED_REASON if label == "unsupported" else "ok"
            s.add(
                Verdict(
                    claim_id=c.id, label=label, quote=quote, quote_found=bool(quote), reason=reason
                )
            )
        synthesis = SynthesisOutput(
            title="EV battery risks",
            summary="Demand is growing.",
            sections=[
                Section(
                    heading="Market",
                    paragraphs=[
                        Paragraph(text="Demand grew.", claim_ids=[ids[0]]),
                        Paragraph(text="Prices eased.", claim_ids=[ids[1]]),
                    ],
                )
            ],
        )
        s.add(Report(run_id=run.id, markdown="# EV", metrics={}, synthesis=synthesis.model_dump()))
        s.commit()
        log_event(s, run.id, "verifier", "claim_verified", {"label": "unsupported"})
        return SimpleNamespace(
            id=run.id, supported_quote=SUPPORTED_QUOTE, rejected_reason=REJECTED_REASON
        )


def test_ask_page(client):
    assert 'name="question"' in client.get("/").text


def test_report_shows_quote_on_hover_and_rejected(client, seeded_run):
    html = client.get(f"/runs/{seeded_run.id}/report").text
    assert 'class="cite"' in html and seeded_run.supported_quote in html
    assert "Rejected claims" in html and seeded_run.rejected_reason in html
    assert "partially verified" in html and "tier 1" in html
    assert "window.print()" in html


def test_trail_page_lists_verdicts(client, seeded_run):
    html = client.get(f"/runs/{seeded_run.id}/trail").text
    assert "unsupported" in html and f"/api/runs/{seeded_run.id}/trail" in html


def test_run_page_follows_event_stream(client, seeded_run):
    html = client.get(f"/runs/{seeded_run.id}").text
    assert f"/api/runs/{seeded_run.id}/events" in html


def test_unknown_run_page_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_eval_page_without_results(client, tmp_path, monkeypatch):
    import app.api.pages as pages

    monkeypatch.setattr(pages, "EVAL_RESULTS", tmp_path)  # independent of committed results
    assert "No eval results yet" in client.get("/eval").text


def test_report_fragment_is_bare(client, seeded_run):
    html = client.get(f"/runs/{seeded_run.id}/report?fragment=1").text
    assert "<html" not in html and "<title>" not in html and 'class="cite"' in html


def test_eval_page_shows_tiles_and_kind_bars(client, tmp_path, monkeypatch):
    import json

    import app.api.pages as pages

    results = {
        "generated_at": "2026-10-27 10:00 UTC",
        "n_questions": 24,
        "runs": {"total": 24, "completed": 23, "failed": 1, "failed_ids": ["x"]},
        "verifier_probe": {
            "catch_rate": 0.9,
            "n_mutants": 40,
            "wrongful_rejection_rate": 0.05,
            "n_originals": 40,
            "catch_rate_by_kind": {"number": 0.95, "negation": 0.8},
            "n_mutants_by_kind": {"number": 20, "negation": 20},
        },
        "independent_judge": {"model": "google/x", "n": 100, "groundedness": 0.93},
        "gold": {"n_labelled": 50, "accuracy": 0.88, "kappa": 0.74},
        "quote_validity_rate": 0.97,
    }
    (tmp_path / "20261027T100000Z.json").write_text(json.dumps(results))
    (tmp_path / "runs.json").write_text("[]")
    monkeypatch.setattr(pages, "EVAL_RESULTS", tmp_path)
    html = client.get("/eval").text
    assert "90.0%" in html and "93.0%" in html and "0.74" in html
    assert 'data-kind="number"' in html and 'data-kind="negation"' in html
