import asyncio

from fastapi import FastAPI

from app.api import guard, runs
from app.config import Settings, get_settings
from app.deps import Deps, build_deps


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    settings = settings or get_settings()
    deps = deps or build_deps(settings)

    app = FastAPI(title="DeepDive Research Agent", version="0.1.0")
    app.state.settings = settings
    app.state.deps = deps
    app.state.run_slots = asyncio.Semaphore(settings.max_concurrent_runs)
    app.state.tasks = set()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(guard.router)
    app.include_router(runs.router)
    return app
