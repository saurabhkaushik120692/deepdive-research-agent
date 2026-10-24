"""Admin area: every question, every account, and the actions on them. Only the account signed
in with CONTACT_EMAIL gets in; for everyone else /admin answers 404, as if it didn't exist."""

import math
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import func, or_, select

from app.api.auth import admin_user, safe_next
from app.api.pages import templates
from app.api.quota import quota_for
from app.models import Run, User, _now

router = APIRouter(prefix="/admin")
PAGE_SIZE = 50


def require_admin(request: Request) -> User:
    admin = admin_user(request)
    if admin is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return admin


Admin = Annotated[User, Depends(require_admin)]
Back = Annotated[str | None, Form()]


def _back(back: str | None) -> RedirectResponse:
    """Return to the admin page the action came from (only ever an /admin page)."""
    path = safe_next(back)
    return RedirectResponse(path if path.startswith("/admin") else "/admin", status_code=303)


def _count(session, query) -> int:
    return session.exec(select(func.count()).select_from(query.subquery())).one()


@router.get("", response_class=HTMLResponse)
def admin_page(request: Request, admin: Admin, tab: str = "reports", q: str = "", page: int = 1):
    deps = request.app.state.deps
    tab = "users" if tab == "users" else "reports"
    q, page = q.strip(), max(page, 1)
    like = f"%{q}%"
    with deps.session_factory() as session:
        week_ago = datetime.now(UTC) - timedelta(days=7)
        summary = {
            "users": session.exec(select(func.count()).select_from(User)).one(),
            "reports": session.exec(select(func.count()).select_from(Run)).one(),
            "failed": session.exec(
                select(func.count()).where(Run.status == "failed", Run.created_at >= week_ago)
            ).one(),
        }
        if tab == "users":
            query = select(User)
            if q:
                query = query.where(or_(User.email.ilike(like), User.name.ilike(like)))
            total = _count(session, query)
            users = session.exec(
                query.order_by(User.last_login_at.desc())
                .offset((page - 1) * PAGE_SIZE)
                .limit(PAGE_SIZE)
            ).all()
            rows = [(u, quota_for(session, u.id, deps.settings)) for u in users]
        else:
            query = select(Run, User).outerjoin(User, User.id == Run.user_id)
            if q:
                query = query.where(or_(Run.question.ilike(like), User.email.ilike(like)))
            total = _count(session, query)
            rows = session.exec(
                query.order_by(Run.created_at.desc())
                .offset((page - 1) * PAGE_SIZE)
                .limit(PAGE_SIZE)
            ).all()
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "admin": admin,
            "tab": tab,
            "q": q,
            "page": page,
            "pages": max(1, math.ceil(total / PAGE_SIZE)),
            "total": total,
            "rows": rows,
            "summary": summary,
        },
    )


def _update_user(request: Request, user_id: int, **fields) -> None:
    with request.app.state.deps.session_factory() as session:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="user not found")
        for name, value in fields.items():
            setattr(user, name, value)
        session.add(user)
        session.commit()


@router.post("/users/{user_id}/reset")
def reset_free_reports(user_id: int, request: Request, admin: Admin, back: Back = None):
    _update_user(request, user_id, quota_reset_at=_now())
    return _back(back)


@router.post("/users/{user_id}/block")
def block_user(user_id: int, request: Request, admin: Admin, back: Back = None):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="you can't block your own account")
    _update_user(request, user_id, blocked=True)
    return _back(back)


@router.post("/users/{user_id}/unblock")
def unblock_user(user_id: int, request: Request, admin: Admin, back: Back = None):
    _update_user(request, user_id, blocked=False)
    return _back(back)


@router.post("/runs/{run_id}/example")
def set_example(
    run_id: str,
    request: Request,
    admin: Admin,
    make: Annotated[str, Form()] = "1",
    back: Back = None,
):
    with request.app.state.deps.session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        if make == "1" and run.status != "completed":
            raise HTTPException(status_code=400, detail="only finished reports can be examples")
        run.is_example = make == "1"
        session.add(run)
        session.commit()
    return _back(back)
