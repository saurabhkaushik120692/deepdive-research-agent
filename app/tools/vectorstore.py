import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import chromadb

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class ChromaStore:
    """Evidence chunks embedded via Mesh. Embeddings are always passed explicitly, so
    Chroma's built-in local embedding model is never used (every AI call goes via Mesh)."""

    def __init__(self, path: Path, embed: EmbedFn) -> None:
        self.embed = embed
        client = chromadb.PersistentClient(path=str(path))
        self.collection = client.get_or_create_collection(
            "evidence", metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    async def add(
        self, run_id: str, sub_question_id: int, chunk_ids: list[int], texts: list[str]
    ) -> None:
        if not chunk_ids:
            return
        vectors = await self.embed(texts)
        await asyncio.to_thread(
            self.collection.upsert,
            ids=[str(c) for c in chunk_ids],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"run_id": run_id, "sub_question_id": sub_question_id}] * len(chunk_ids),
        )

    async def query(self, run_id: str, sub_question_id: int, text: str, k: int) -> list[int]:
        where = {"$and": [{"run_id": run_id}, {"sub_question_id": sub_question_id}]}
        n = await asyncio.to_thread(lambda: len(self.collection.get(where=where)["ids"]))
        if n == 0:
            return []
        [vector] = await self.embed([text])
        res = await asyncio.to_thread(
            self.collection.query, query_embeddings=[vector], n_results=min(k, n), where=where
        )
        return [int(i) for i in res["ids"][0]]
