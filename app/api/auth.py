"""Google sign-in (OpenID Connect via Authlib), the signed session cookie, and run visibility.

Signing in for the first time creates the account. The session cookie (Starlette's
SessionMiddleware, signed with SESSION_SECRET) holds the user's id, name and picture.
A run is visible to its owner, or to everyone when it is marked as an example report."""

import logging
import secrets

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from app.config import Settings
from app.deps import Deps
from app.models import Run, User, _now

log = logging.getLogger(__name__)
GOOGLE_METADATA = "https://accounts.google.com/.well-known/openid-configuration"

router = APIRouter(prefix="/auth")


def build_google(settings: Settings):
    """The Authlib Google client, or None when sign-in isn't configured."""
    if not (settings.google_client_id and settings.google_client_secret):
        return None
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url=GOOGLE_METADATA,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth.google


def session_secret(settings: Settings) -> str:
    if settings.session_secret:
        return settings.session_secret
    log.warning("SESSION_SECRET is not set: using a random key, so sign-ins end on restart")
    return secrets.token_urlsafe(32)


def session_user(request: Request) -> dict | None:
    return request.session.get("user")


def require_user(request: Request) -> dict:
    user = session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="sign in to start research")
    return user


def is_admin_email(settings: Settings, email: str) -> bool:
    """The admin is whoever signs in with CONTACT_EMAIL; with it unset there is no admin."""
    admin = settings.contact_email.strip().lower()
    return bool(admin) and email.strip().lower() == admin


def admin_user(request: Request) -> User | None:
    """The signed-in user if they are the admin, else None. Checked against the database."""
    user = session_user(request)
    if user is None:
        return None
    deps: Deps = request.app.state.deps
    with deps.session_factory() as session:
        row = session.get(User, user["id"])
    return row if row is not None and is_admin_email(deps.settings, row.email) else None


def can_view(run: Run, user: dict | None, admin: bool = False) -> bool:
    if admin or run.is_example or run.is_shared:
        return True
    return user is not None and run.user_id == user["id"]


def safe_next(path: str | None) -> str:
    """Only same-site paths: '/runs/x' yes; '//evil', 'https://evil', '/\\evil' no."""
    if not path or not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/"
    return path


def _callback_url(request: Request, settings: Settings) -> str:
    if settings.site_host:  # behind Caddy the app itself only sees plain HTTP
        return f"https://{settings.site_host}/auth/callback"
    return str(request.url_for("auth_callback"))


@router.get("/login")
async def login(request: Request, next: str | None = None):
    google = request.app.state.google
    if google is None:
        return HTMLResponse(
            "Google sign-in isn't configured on this server "
            "(set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET).",
            status_code=503,
        )
    request.session["next"] = safe_next(next)
    return await google.authorize_redirect(
        request, _callback_url(request, request.app.state.settings)
    )


@router.get("/callback", name="auth_callback")
async def callback(request: Request):
    next_path = safe_next(request.session.pop("next", None))
    try:
        token = await request.app.state.google.authorize_access_token(request)
    except (OAuthError, AttributeError) as e:
        log.info("Google sign-in failed: %s", e)
        return RedirectResponse("/?signin=failed", status_code=303)
    info = token.get("userinfo") or {}
    if not info.get("sub") or not info.get("email") or not info.get("email_verified"):
        return RedirectResponse("/?signin=failed", status_code=303)

    deps: Deps = request.app.state.deps
    with deps.session_factory() as session:
        user = session.exec(select(User).where(User.google_sub == info["sub"])).first()
        if user is None:
            user = User(google_sub=info["sub"], email=info["email"])
        user.email = info["email"]
        user.name = info.get("name") or info["email"]
        user.picture = info.get("picture") or ""
        user.last_login_at = _now()
        session.add(user)
        session.commit()
        session.refresh(user)
        request.session["user"] = {"id": user.id, "name": user.name, "picture": user.picture}
    return RedirectResponse(next_path, status_code=303)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)
