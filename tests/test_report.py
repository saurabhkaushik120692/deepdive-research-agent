from app.graph.schemas import Paragraph, Section, SynthesisOutput
from app.models import Claim, Source, Verdict
from app.report import render_report


def _fixture():
    claims = {
        1: Claim(id=1, run_id="r", sub_question_id=1, text="Demand rose 28%.", chunk_ids=[10]),
        2: Claim(id=2, run_id="r", sub_question_id=1, text="Prices fell sharply.", chunk_ids=[20]),
        3: Claim(id=3, run_id="r", sub_question_id=1, text="Market is $5T.", chunk_ids=[20]),
    }
    verdicts = {
        1: Verdict(claim_id=1, label="supported", quote="rose 28%", quote_found=True),
        2: Verdict(claim_id=2, label="partial", quote="prices fell", quote_found=True),
        3: Verdict(claim_id=3, label="unsupported", reason="figure not in source"),
    }
    iea = Source(
        id=1, run_id="r", url="https://www.iea.org/r", domain="iea.org", title="IEA", tier=1
    )
    blog = Source(id=2, run_id="r", url="https://blog.x.com/p", domain="blog.x.com", tier=3)
    blog.title = "Blog"
    sources = {10: iea, 20: blog}
    synthesis = SynthesisOutput(
        title="EV battery risks",
        summary="Short summary.",
        sections=[
            Section(
                heading="Market",
                paragraphs=[
                    Paragraph(text="Demand grew strongly.", claim_ids=[1]),
                    Paragraph(text="Prices dropped.", claim_ids=[2]),
                ],
            )
        ],
    )
    return synthesis, claims, verdicts, sources


def test_report_sections_and_markers():
    synthesis, claims, verdicts, sources = _fixture()
    md = render_report("EV risks?", synthesis, claims, verdicts, sources, [], [])
    assert md.startswith("# EV battery risks")
    assert "Demand grew strongly. [1]" in md
    assert "Prices dropped. [2] ⚠ partially verified" in md
    rejected = md.split("## Rejected claims")[1].split("## Sources")[0]
    assert "Market is $5T." in rejected and "figure not in source" in rejected
    sources_part = md.split("## Sources")[1]
    assert "[1] IEA (https://www.iea.org/r), tier 1" in sources_part
    assert "## Limited evidence" not in md


def test_report_lists_limited_evidence():
    synthesis, claims, verdicts, sources = _fixture()
    md = render_report("EV risks?", synthesis, claims, verdicts, sources, ["Who regulates?"], [])
    assert "## Limited evidence" in md and "Who regulates?" in md
