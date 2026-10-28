"""The only place that talks to models. Every call goes through Mesh (hackathon rule)."""

import asyncio
import json
import re
import time
from collections import deque
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

Role = Literal[
    "planner",
    "drafter",
    "verifier",
    "synthesizer",
    "judge",
    "contradictions",
    "scope",
    "followup_drafter",
    "followup_answer",
]
T = TypeVar("T", bound=BaseModel)
UsageCallback = Callable[[str | None, dict], None]

_TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


EMBED_BATCH = 64
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fences(content: str) -> str:
    """Some models wrap JSON in ```json fences even in JSON mode."""
    m = _FENCE.match(content)
    return m.group(1) if m else content


class RateLimiter:
    """Sliding one-minute window: at most `rpm` requests start in any 60 s."""

    def __init__(self, rpm: int, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.rpm, self.clock, self.sleep = rpm, clock, sleep
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self.clock()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                await self.sleep(60 - (now - self._starts[0]))


def _retry_after(e: openai.RateLimitError) -> float | None:
    try:
        return float(e.response.headers.get("retry-after", ""))
    except (AttributeError, ValueError):
        return None


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
        rate_limit_wait: float = 15.0,
    ) -> None:
        self.settings = settings
        # A missing key must not crash startup (health checks, tests); calls fail with 401 instead.
        # SDK retries are off: _call() owns the retry policy, and the timeout bounds a hung call.
        self.client = client or AsyncOpenAI(
            base_url=settings.mesh_base_url,
            api_key=settings.mesh_api_key or "MESH_API_KEY-not-set",
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        self.on_usage = on_usage
        self.retry_base_delay = retry_base_delay
        self.rate_limit_wait = rate_limit_wait
        self._slots = asyncio.Semaphore(settings.llm_concurrency)
        # One limiter per client, shared by every run in the process (Mesh limits per account).
        self._limiter = RateLimiter(settings.llm_rpm)

    async def _call(self, fn, **kwargs):
        """One API call: requests/minute limit, bounded concurrency, backoff on transient errors.

        A 429 means the per-minute window is full, so it waits long enough for the window to move
        (Retry-After if Mesh sends it) instead of the short exponential backoff."""
        attempt = 0
        while True:
            try:
                async with self._slots:
                    # Take window budget only once a slot is held, i.e. when the request is sent.
                    await self._limiter.acquire()
                    return await fn(**kwargs)
            except _TRANSIENT as e:
                # A hung call already cost up to llm_timeout_s; don't repeat that many times.
                limit = (
                    self.settings.llm_timeout_retries
                    if isinstance(e, openai.APITimeoutError)
                    else self.settings.llm_max_retries
                )
                if attempt >= limit:
                    raise
                delay = self.retry_base_delay * 2**attempt
                if isinstance(e, openai.RateLimitError):
                    delay = max(delay, _retry_after(e) or self.rate_limit_wait * (attempt + 1))
                await asyncio.sleep(delay)
                attempt += 1

    async def chat_json(
        self, *, role: Role, system: str, user: str, schema: type[T], run_id: str | None = None
    ) -> T:
        model = getattr(self.settings, f"model_{role}")
        effort = getattr(self.settings, f"reasoning_effort_{role}", "")
        extra = {"reasoning_effort": effort} if effort else {}
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
                **extra,
            )
            self._report_usage(run_id, role, model, resp, started)
            content = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_strip_fences(content))
            except ValidationError as e:
                last_error = str(e)
            if attempt == 0:
                messages = messages + [
                    # Anthropic rejects empty content blocks, so an empty reply gets a placeholder.
                    {"role": "assistant", "content": content.strip() or "(empty reply)"},
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
        """Embed in batches: embedding endpoints cap inputs/tokens per request.

        Inputs are cut to embed_max_chars (most of a chunk). Unusually dense text can still exceed
        the model's 2048-token input limit; that batch is retried once at embed_fallback_chars."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            try:
                resp = await self._embed_batch(batch, self.settings.embed_max_chars)
            except openai.BadRequestError:
                resp = await self._embed_batch(batch, self.settings.embed_fallback_chars)
            vectors += [item.embedding for item in resp.data]
        return vectors

    async def _embed_batch(self, batch: list[str], max_chars: int):
        return await self._call(
            self.client.embeddings.create,
            model=self.settings.model_embed,
            input=[t[:max_chars] for t in batch],
        )

    async def moderate(self, text: str) -> bool:
        resp = await self._call(
            self.client.moderations.create, model=self.settings.model_moderation, input=text
        )
        return bool(resp.results[0].flagged)
