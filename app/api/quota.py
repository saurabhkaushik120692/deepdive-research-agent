"""Free reports per account. Queued, running and completed runs count; failed runs don't, so a
run that dies gives its report back."""

from dataclasses import dataclass

from sqlmodel import func, select

from app.config import Settings
from app.models import Run, User


@dataclass(frozen=True)
class Quota:
    used: int
    limit: int | None  # None: this account has no limit

    @property
    def left(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit


def _is_unlimited(settings: Settings, email: str) -> bool:
    allowed = {e.strip().lower() for e in settings.unlimited_emails.split(",") if e.strip()}
    return email.lower() in allowed


def quota_for(session, user_id: int, settings: Settings) -> Quota:
    used = session.exec(
        select(func.count()).where(Run.user_id == user_id, Run.status != "failed")
    ).one()
    user = session.get(User, user_id)
    if user is not None and _is_unlimited(settings, user.email):
        return Quota(used, None)
    return Quota(used, settings.free_reports_per_user)


def limit_message(settings: Settings) -> str:
    msg = (
        f"You've used your {settings.free_reports_per_user} free reports. "
        "Paid plans are coming soon."
    )
    if settings.contact_email:
        msg += f" Need more? Write to {settings.contact_email}."
    return msg
