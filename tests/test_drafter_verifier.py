import pytest

from app.graph.nodes.drafter import draft_node
from app.graph.nodes.verifier import verify_node
from app.graph.schemas import DraftClaim, DraftOutput, VerdictOutput
from app.models import Claim, EvidenceChunk, Source
from tests.helpers import cfg, claims_of, event_types, state, verdict_of

CHUNK = "In 2025, global EV battery demand rose 28% to 1.1 TWh, according to the IEA."
OTHER = "Football: the home team won 3-1 on Saturday after a late penalty."


def _add_chunk(deps, run, sq, text, url, tier):
    with deps.session_factory() as s:
        src = Source(run_id=run.id, url=url, domain=url.split("/")[2], tier=tier)
        s.add(src)
        s.commit()
        s.refresh(src)
        ch = EvidenceChunk(source_id=src.id, run_id=run.id, sub_question_id=sq.id, text=text)
        s.add(ch)
        s.commit()
        s.refresh(ch)
        return ch


@pytest.fixture
async def chunks(deps, run, sq):
    blog = _add_chunk(deps, run, sq, OTHER, "https://blog.example.com/a", tier=3)
    iea = _add_chunk(deps, run, sq, CHUNK, "https://www.iea.org/r", tier=1)
    await deps.vectors.add(run.id, sq.id, [blog.id, iea.id], [blog.text, iea.text])
    return {"blog": blog, "iea": iea}


@pytest.fixture
def claim_on_chunk(deps, run, sq, chunks):
    with deps.session_factory() as s:
        c = Claim(
            run_id=run.id,
            sub_question_id=sq.id,
            text="EV battery demand rose 28% in 2025.",
            chunk_ids=[chunks["iea"].id],
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return c


async def test_drafter_drops_claims_with_unknown_chunk_ids(deps, run, sq, chunks):
    iea = chunks["iea"].id
    deps.llm.handlers["drafter"] = lambda s, u: DraftOutput(
        claims=[
            DraftClaim(sub_question_id=sq.id, text="Demand rose 28%.", chunk_ids=[iea]),
            DraftClaim(sub_question_id=sq.id, text="Invented.", chunk_ids=[999]),
            DraftClaim(sub_question_id=sq.id, text="Half invented.", chunk_ids=[iea, 998]),
        ]
    )
    await draft_node(state(run, pending=[sq.id]), cfg(deps))
    assert [c.text for c in claims_of(deps, run.id)] == ["Demand rose 28%."]
    assert "claim_dropped_bad_citation" in event_types(deps, run.id)


async def test_drafter_sees_tier1_chunks_first_with_ids(deps, run, sq, chunks):
    deps.llm.handlers["drafter"] = lambda s, u: DraftOutput(claims=[])
    await draft_node(state(run, pending=[sq.id]), cfg(deps))
    _, user = deps.llm.calls[-1]
    iea_pos = user.index(f"[chunk:{chunks['iea'].id}]")
    blog_pos = user.index(f"[chunk:{chunks['blog'].id}]")
    assert iea_pos < blog_pos and sq.text in user


async def test_drafter_uses_current_round_and_subquestion(deps, run, sq, chunks):
    iea = chunks["iea"].id
    deps.llm.handlers["drafter"] = lambda s, u: DraftOutput(
        claims=[DraftClaim(sub_question_id=12345, text="Demand rose 28%.", chunk_ids=[iea])]
    )
    await draft_node(state(run, pending=[sq.id], round_=1), cfg(deps))
    [c] = claims_of(deps, run.id)
    assert c.sub_question_id == sq.id and c.round == 1


async def test_verifier_downgrades_fabricated_quote(deps, run, claim_on_chunk):
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="supported", quote="demand fell 12% to 0.9 TWh", reason="r"
    )
    await verify_node(state(run), cfg(deps))
    v = verdict_of(deps, claim_on_chunk.id)
    assert v.label == "unsupported" and v.quote_found is False
    assert v.reason == "verifier quote not found in source"
    assert v.model == deps.settings.model_verifier


async def test_verifier_keeps_real_quote(deps, run, claim_on_chunk):
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="supported", quote="demand rose 28% to 1.1 TWh", reason="r"
    )
    await verify_node(state(run), cfg(deps))
    v = verdict_of(deps, claim_on_chunk.id)
    assert v.label == "supported" and v.quote_found is True
    assert "claim_verified" in event_types(deps, run.id)


async def test_verifier_sees_only_cited_chunk(deps, run, claim_on_chunk, chunks):
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="unsupported", quote="", reason="r"
    )
    await verify_node(state(run), cfg(deps))
    role, user = deps.llm.calls[-1]
    assert role == "verifier"
    assert CHUNK in user and OTHER not in user and claim_on_chunk.text in user


async def test_verifier_skips_already_verified(deps, run, claim_on_chunk):
    deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(
        label="unsupported", quote="", reason="r"
    )
    await verify_node(state(run), cfg(deps))
    await verify_node(state(run), cfg(deps))
    assert sum(1 for r, _ in deps.llm.calls if r == "verifier") == 1


async def test_verifier_invalid_output_fails_closed(deps, run, claim_on_chunk):
    from app.llm import LlmOutputError

    deps.llm.handlers["verifier"] = lambda s, u: LlmOutputError("bad json twice")
    await verify_node(state(run), cfg(deps))
    v = verdict_of(deps, claim_on_chunk.id)
    assert v.label == "unsupported" and v.reason.startswith("verifier error")
