"""Test doubles. Tests never touch the network."""

from types import SimpleNamespace


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

    def __init__(self, chat=None, usage=(10, 5), moderation_flagged=False):
        self._chat = list(chat or [])
        self._usage = usage
        self._flagged = moderation_flagged
        self.chat_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.moderation_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(self))
        self.embeddings = _FakeEmbeddings(self)
        self.moderations = _FakeModerations(self)
