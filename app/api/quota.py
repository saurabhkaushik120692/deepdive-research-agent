"""Free reports per account. Queued, running and completed runs count; failed runs don't, so a
run that dies gives its report back. An admin reset counts again from the reset time, and a
blocked account can't start new runs at all."""

from dataclasses import dataclass

from sqlmodel import func, select

from app.api.auth import is_admin_email
from app.config import Settings
from app.models import Run, User


@dataclass(frozen=True)
class Quota:
    used: int
    limit: int | None  # None: this account has no limit
    blocked: bool = False

    @property
    def left(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit


def _is_unlimited(settings: Settings, email: str) -> bool:
    allowed = {e.strip().lower() for e in settings.unlimited_emails.split(",") if e.strip()}
    return email.lower() in allowed or is_admin_email(settings, email)


def quota_for(session, user_id: int, settings: Settings) -> Quota:
    user = session.get(User, user_id)
    query = select(func.count()).where(Run.user_id == user_id, Run.status != "failed")
    if user is not None and user.quota_reset_at is not None:
        query = query.where(Run.created_at >= user.quota_reset_at)
    used = session.exec(query).one()
    if user is None:
        return Quota(used, settings.free_reports_per_user)
    limit = None if _is_unlimited(settings, user.email) else settings.free_reports_per_user
    return Quota(used, limit, user.blocked)


def _contact(settings: Settings, prefix: str) -> str:
    return f" {prefix} Write to {settings.contact_email}." if settings.contact_email else ""


def limit_message(settings: Settings) -> str:
    return (
        f"You've used your {settings.free_reports_per_user} free reports. "
        "Paid plans are coming soon." + _contact(settings, "Need more?")
    )


def blocked_message(settings: Settings) -> str:
    return "Your account can't start new reports." + _contact(settings, "Questions?")


def start_refusal(quota: Quota, settings: Settings) -> str | None:
    """Why this account may not start a run now, or None if it may."""
    if quota.blocked:
        return blocked_message(settings)
    if quota.exhausted:
        return limit_message(settings)
    return None
