"""Independent judge: a third model family (not the drafter's, not the verifier's) re-checks
final-report claims, so the groundedness number isn't the verifier grading its own homework."""

from app.graph.schemas import JudgeOutput
from app.llm import LlmClient
from app.prompts import load_prompt


async def judge_claim(llm: LlmClient, claim_text: str, chunk_text: str) -> JudgeOutput:
    return await llm.chat_json(
        role="judge",
        system=load_prompt("judge"),
        user=f"Claim:\n{claim_text}\n\nSource text:\n{chunk_text}",
        schema=JudgeOutput,
    )
