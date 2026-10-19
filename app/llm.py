"""The ONLY place that talks to models. Every call goes through Mesh (hackathon rule)."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

Role = Literal["planner", "drafter", "verifier", "synthesizer", "judge", "contradictions"]
T = TypeVar("T", bound=BaseModel)
UsageCallback = Callable[[str | None, dict], None]

_TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LlmOutputError(Exception):
    """The model did not return valid JSON for the schema, even after one repair attempt."""


class LlmClient(Protocol):
    async def chat_json(
        self, *, role: Role, system: str, user: str, schema: type[T], run_id: str | None = None
    ) -> T: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def moderate(self, text: str) -> bool: ...


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "\n\nRespond with a single JSON object (no prose, no code fences) that matches this "
        f"JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )


class MeshLlm:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        on_usage: UsageCallback | None = None,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.settings = settings
        # A missing key must not crash startup (health checks, tests); calls fail with 401 instead.
        self.client = client or AsyncOpenAI(
            base_url=settings.mesh_base_url, api_key=settings.mesh_api_key or "MESH_API_KEY-not-set"
        )
        self.on_usage = on_usage
        self.retry_base_delay = retry_base_delay
        self._slots = asyncio.Semaphore(settings.llm_concurrency)

    async def _call(self, fn, **kwargs):
        """Run one API call with bounded concurrency and exponential backoff on transient errors."""
        attempt = 0
        while True:
            try:
                async with self._slots:
                    return await fn(**kwargs)
            except _TRANSIENT:
                if attempt >= self.settings.llm_max_retries:
                    raise
                await asyncio.sleep(self.retry_base_delay * 2**attempt)
                attempt += 1

    async def chat_json(
        self, *, role: Role, system: str, user: str, schema: type[T], run_id: str | None = None
    ) -> T:
        model = getattr(self.settings, f"model_{role}")
        messages = [
            {"role": "system", "content": system + _schema_instructions(schema)},
            {"role": "user", "content": user},
        ]
        last_error = ""
        for attempt in range(2):  # first try + one repair
            started = time.perf_counter()
            resp = await self._call(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            self._report_usage(run_id, role, model, resp, started)
            content = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(content)
            except ValidationError as e:
                last_error = str(e)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Your previous reply was not valid for the schema:\n"
                        f"{last_error}\nReply again with ONLY the corrected JSON object.",
                    },
                ]
        raise LlmOutputError(f"{role} ({model}) returned invalid JSON twice: {last_error}")

    def _report_usage(self, run_id, role, model, resp, started) -> None:
        if self.on_usage is None:
            return
        usage = getattr(resp, "usage", None)
        self.on_usage(
            run_id,
            {
                "role": role,
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._call(
            self.client.embeddings.create, model=self.settings.model_embed, input=texts
        )
        return [item.embedding for item in resp.data]

    async def moderate(self, text: str) -> bool:
        resp = await self._call(
            self.client.moderations.create, model=self.settings.model_moderation, input=text
        )
        return bool(resp.results[0].flagged)
