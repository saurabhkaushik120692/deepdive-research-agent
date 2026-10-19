"""Pydantic I/O contracts for every agent. The LLM client validates replies against these."""

from typing import Literal

from pydantic import BaseModel, Field


class PlannedSubQuestion(BaseModel):
    text: str
    query: str = Field(description="web search query for this sub-question")


class PlanOutput(BaseModel):
    sub_questions: list[PlannedSubQuestion] = Field(min_length=3, max_length=6)


class RefineOutput(BaseModel):
    query: str


class DraftClaim(BaseModel):
    sub_question_id: int
    text: str = Field(description="one atomic, checkable fact")
    chunk_ids: list[int] = Field(min_length=1, max_length=2)


class DraftOutput(BaseModel):
    claims: list[DraftClaim]


class VerdictOutput(BaseModel):
    label: Literal["supported", "partial", "unsupported"]
    quote: str = Field(description="exact sentence copied verbatim from the source text")
    reason: str


class Paragraph(BaseModel):
    text: str
    claim_ids: list[int] = Field(min_length=1)


class Section(BaseModel):
    heading: str
    paragraphs: list[Paragraph]


class SynthesisOutput(BaseModel):
    title: str
    summary: str
    sections: list[Section]


class Conflict(BaseModel):
    claim_a: int
    claim_b: int
    explanation: str


class ContradictionOutput(BaseModel):
    conflicts: list[Conflict]


class JudgeOutput(BaseModel):
    supported: bool
    reason: str
