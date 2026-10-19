import httpx
import openai
import pytest
from pydantic import BaseModel

from app.llm import LlmOutputError, MeshLlm
from tests.fakes import FakeOpenAI


class Out(BaseModel):
    answer: str


def _llm(settings, fake, **kw):
    return MeshLlm(settings, client=fake, retry_base_delay=0, **kw)


async def test_chat_json_parses_and_uses_role_model(settings):
    fake = FakeOpenAI(chat=['{"answer": "hi"}'])
    out = await _llm(settings, fake).chat_json(role="verifier", system="s", user="u", schema=Out)
    assert out.answer == "hi"
    assert fake.chat_calls[0]["model"] == settings.model_verifier
    assert fake.chat_calls[0]["response_format"] == {"type": "json_object"}


async def test_repair_retry_once_then_succeeds(settings):
    fake = FakeOpenAI(chat=["not json", '{"answer": "fixed"}'])
    out = await _llm(settings, fake).chat_json(role="planner", system="s", user="u", schema=Out)
    assert out.answer == "fixed" and len(fake.chat_calls) == 2


async def test_two_bad_outputs_raise(settings):
    fake = FakeOpenAI(chat=["nope", '{"wrong": 1}'])
    with pytest.raises(LlmOutputError):
        await _llm(settings, fake).chat_json(role="planner", system="s", user="u", schema=Out)


async def test_usage_callback_receives_tokens(settings):
    seen = []
    fake = FakeOpenAI(chat=['{"answer": "x"}'], usage=(120, 30))
    llm = _llm(settings, fake, on_usage=lambda rid, u: seen.append((rid, u)))
    await llm.chat_json(role="drafter", system="s", user="u", schema=Out, run_id="r1")
    assert seen[0][0] == "r1" and seen[0][1]["prompt_tokens"] == 120
    assert seen[0][1]["role"] == "drafter" and seen[0][1]["model"] == settings.model_drafter


async def test_transient_error_is_retried(settings):
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://api.meshapi.ai/v1"))
    fake = FakeOpenAI(chat=[timeout, '{"answer": "ok"}'])
    out = await _llm(settings, fake).chat_json(role="planner", system="s", user="u", schema=Out)
    assert out.answer == "ok" and len(fake.chat_calls) == 2


async def test_moderate_returns_flag(settings):
    fake = FakeOpenAI(moderation_flagged=True)
    assert await _llm(settings, fake).moderate("bad text") is True
    assert fake.moderation_calls[0]["model"] == settings.model_moderation


async def test_embed_uses_embed_model(settings):
    fake = FakeOpenAI()
    vecs = await _llm(settings, fake).embed(["ab", "abcd"])
    assert vecs == [[2.0, 1.0], [4.0, 1.0]]
    assert fake.embed_calls[0]["model"] == settings.model_embed
