"""Test doubles. Tests never touch the network."""

import hashlib
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import openai
from pydantic import BaseModel

from app.deps import SearchResult


class _FakeChatCompletions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        o = self._owner
        o.chat_calls.append(kwargs)
        item = o._chat.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = SimpleNamespace(prompt_tokens=o._usage[0], completion_tokens=o._usage[1])
        message = SimpleNamespace(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeEmbeddings:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.embed_calls.append(kwargs)
        limit = self._owner.embed_max_len
        if limit is not None and any(len(t) > limit for t in kwargs["input"]):
            req = httpx.Request("POST", "https://api.meshapi.ai/v1/embeddings")
            raise openai.BadRequestError(
                "input too long for the model's context length",
                response=httpx.Response(400, request=req),
                body=None,
            )
        data = [SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in kwargs["input"]]
        return SimpleNamespace(data=data)


class _FakeModerations:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.moderation_calls.append(kwargs)
        return SimpleNamespace(results=[SimpleNamespace(flagged=self._owner._flagged)])


class FakeOpenAI:
    """Stands in for openai.AsyncOpenAI: scripted chat replies (str or Exception)."""

    def __init__(self, chat=None, usage=(10, 5), moderation_flagged=False, embed_max_len=None):
        self._chat = list(chat or [])
        self.embed_max_len = embed_max_len
        self._usage = usage
        self._flagged = moderation_flagged
        self.chat_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.moderation_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(self))
        self.embeddings = _FakeEmbeddings(self)
        self.moderations = _FakeModerations(self)


# --- graph-level fakes -------------------------------------------------------


class ScriptedLlm:
    """LlmClient double: per-role handler(system, user) -> schema instance (or Exception)."""

    def __init__(self, handlers: dict[str, Callable[[str, str], object]] | None = None):
        self.handlers = handlers or {}
        self.calls: list[tuple[str, str]] = []
        self.flagged = False

    async def chat_json(self, *, role, system, user, schema, run_id=None):
        self.calls.append((role, user))
        if role not in self.handlers:
            raise AssertionError(f"ScriptedLlm has no handler for role {role!r}")
        out = self.handlers[role](system, user)
        if isinstance(out, Exception):
            raise out
        assert isinstance(out, BaseModel)
        return out

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            vecs.append([b / 255 for b in digest[:8]])
        return vecs

    async def moderate(self, text: str) -> bool:
        return self.flagged


class FakeSearch:
    def __init__(self, results: dict[str, list[SearchResult]] | None = None):
        self.results = results or {}
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        return list(self.results.get(query, []))[:max_results]


class FakeFetcher:
    def __init__(self, pages: dict[str, str | None] | None = None):
        self.pages = pages or {}

    async def fetch_text(self, url: str) -> str | None:
        return self.pages.get(url)


class MemoryVectors:
    """Returns the first k chunk ids stored for (run, sub-question), in insertion order."""

    def __init__(self):
        self.store: dict[tuple[str, int], list[int]] = {}

    async def add(self, run_id, sub_question_id, chunk_ids, texts):
        self.store.setdefault((run_id, sub_question_id), []).extend(chunk_ids)

    async def query(self, run_id, sub_question_id, text, k):
        return self.store.get((run_id, sub_question_id), [])[:k]

    async def query_run(self, run_id, text, k):
        ids = [i for (r, _), chunk_ids in self.store.items() if r == run_id for i in chunk_ids]
        return ids[:k]
