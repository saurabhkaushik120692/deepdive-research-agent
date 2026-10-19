from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.events import log_event
from app.graph.schemas import Paragraph, Section, SynthesisOutput
from app.main import create_app
from app.models import Claim, EvidenceChunk, Report, Run, Source, SubQuestion, Verdict

SUPPORTED_QUOTE = "global EV battery demand rose 28% to 1.1 TWh"
REJECTED_REASON = "the $5T figure does not appear in the source"


@pytest.fixture
def make_client(deps):
    def _make(**overrides):
        for k, v in overrides.items():
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
def seeded_run(deps):
    with deps.session_factory() as s:
        run = Run(question="EV battery risks 2026?", status="completed", models={})
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


def test_gate_page_when_code_set(client_with_code):
    html = client_with_code.get("/").text
    assert 'name="code"' in html and 'name="question"' not in html


def test_pages_gated_without_cookie(client_with_code, seeded_run):
    r = client_with_code.get(f"/runs/{seeded_run.id}/report", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"


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


def test_eval_page_without_results(client):
    assert "No eval results yet" in client.get("/eval").text


def test_report_fragment_is_bare(client, seeded_run):
    html = client.get(f"/runs/{seeded_run.id}/report?fragment=1").text
    assert "<html" not in html and "<title>" not in html and 'class="cite"' in html
