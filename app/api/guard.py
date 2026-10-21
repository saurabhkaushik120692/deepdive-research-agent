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


def _same(a: str, b: str) -> bool:
    """Constant-time comparison that also accepts non-ASCII input (str compare_digest raises)."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def has_access(request: Request) -> bool:
    code = _code(request)
    return not code or _same(request.cookies.get(COOKIE, ""), code)


def require_access(request: Request) -> None:
    if not has_access(request):
        raise HTTPException(status_code=401, detail="access code required")


@router.post("/access")
def access(request: Request, code: str = Form(...)):
    expected = _code(request)
    if expected and not _same(code, expected):
        raise HTTPException(status_code=401, detail="wrong access code")
    resp = RedirectResponse("/", status_code=303)
    # Behind Caddy the app sees plain HTTP; the proxy says the visitor used HTTPS.
    https = request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
    resp.set_cookie(
        COOKIE, code, httponly=True, samesite="lax", secure=https, max_age=7 * 24 * 3600
    )
    return resp
