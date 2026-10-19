import pytest
from pydantic import ValidationError

from app.graph.schemas import DraftClaim, Paragraph, PlannedSubQuestion, PlanOutput


def test_plan_requires_3_to_6():
    sq = PlannedSubQuestion(text="a", query="a")
    with pytest.raises(ValidationError):
        PlanOutput(sub_questions=[sq] * 2)
    with pytest.raises(ValidationError):
        PlanOutput(sub_questions=[sq] * 7)
    assert len(PlanOutput(sub_questions=[sq] * 3).sub_questions) == 3


def test_draft_claim_needs_1_or_2_chunks():
    with pytest.raises(ValidationError):
        DraftClaim(sub_question_id=1, text="x", chunk_ids=[])
    with pytest.raises(ValidationError):
        DraftClaim(sub_question_id=1, text="x", chunk_ids=[1, 2, 3])


def test_paragraph_needs_a_claim():
    with pytest.raises(ValidationError):
        Paragraph(text="x", claim_ids=[])
