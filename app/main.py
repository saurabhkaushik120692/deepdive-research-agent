import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.api import auth, pages, runs
from app.config import Settings, get_settings
from app.deps import Deps, build_deps
from app.graph.runner import fail_interrupted_runs


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    settings = settings or get_settings()
    deps = deps or build_deps(settings)
    fail_interrupted_runs(deps)  # in-process runs don't survive a restart or redeploy

    app = FastAPI(title="DeepDive Research Agent", version="0.1.0")
    app.state.settings = settings
    app.state.deps = deps
    app.state.run_slots = asyncio.Semaphore(settings.max_concurrent_runs)
    app.state.tasks = set()
    app.state.google = auth.build_google(settings)
    app.add_middleware(
        SessionMiddleware,
        secret_key=auth.session_secret(settings),
        session_cookie="dd_session",
        max_age=30 * 24 * 3600,
        same_site="lax",
        https_only=bool(settings.site_host),
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth.router)
    app.include_router(runs.router)
    app.include_router(pages.router)
    app.add_exception_handler(pages.SignInRequired, pages.sign_in_redirect)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
