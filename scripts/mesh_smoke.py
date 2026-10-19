"""Day-1 check that every configured Mesh model answers. Run: uv run python scripts/mesh_smoke.py

Uses the real MESH_API_KEY from .env and spends a few cents at most.
"""

import asyncio
import sys
import time
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import MeshLlm  # noqa: E402

ROLES = ["planner", "drafter", "verifier", "synthesizer", "judge", "contradictions"]


class Ping(BaseModel):
    reply: str


async def main() -> int:
    settings = Settings()
    if not settings.mesh_api_key:
        print("MESH_API_KEY is empty: copy .env.example to .env and fill it in.")
        return 1
    usage: list[dict] = []
    llm = MeshLlm(settings, on_usage=lambda _rid, u: usage.append(u))
    failures = 0

    for role in ROLES:
        try:
            out = await llm.chat_json(
                role=role,
                system="You are a test.",
                user='Reply with {"reply": "pong"}.',
                schema=Ping,
            )
            u = usage[-1]
            print(
                f"OK   {role:<15} {u['model']:<35} {u['latency_ms']:>6} ms "
                f"in={u['prompt_tokens']} out={u['completion_tokens']} reply={out.reply!r}"
            )
        except Exception as e:  # report every failure, keep going
            failures += 1
            model = getattr(settings, f"model_{role}")
            print(f"FAIL {role:<15} {model:<35} {type(e).__name__}: {e}")

    for name, coro in (
        ("embed", lambda: llm.embed(["hello world"])),
        ("moderate", lambda: llm.moderate("hello world")),
    ):
        model = settings.model_embed if name == "embed" else settings.model_moderation
        started = time.perf_counter()
        try:
            result = await coro()
            detail = f"dim={len(result[0])}" if name == "embed" else f"flagged={result}"
            ms = int((time.perf_counter() - started) * 1000)
            print(f"OK   {name:<15} {model:<35} {ms:>6} ms {detail}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name:<15} {model:<35} {type(e).__name__}: {e}")

    print("\nall OK" if failures == 0 else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
