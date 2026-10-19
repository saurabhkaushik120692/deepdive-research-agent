"""Protects the public demo: an access code (cookie) gate. Rate limiting lives in runs.py."""

import hmac

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.deps import Deps

COOKIE = "dd_access"

router = APIRouter()


def _code(request: Request) -> str:
    deps: Deps = request.app.state.deps
    return deps.settings.demo_access_code


def has_access(request: Request) -> bool:
    code = _code(request)
    return not code or hmac.compare_digest(request.cookies.get(COOKIE, ""), code)


def require_access(request: Request) -> None:
    if not has_access(request):
        raise HTTPException(status_code=401, detail="access code required")


@router.post("/access")
def access(request: Request, code: str = Form(...)):
    expected = _code(request)
    if expected and not hmac.compare_digest(code, expected):
        raise HTTPException(status_code=401, detail="wrong access code")
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, code, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
    return resp
