"""SQLAlchemy declarative base + shared column helpers."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy.orm import DeclarativeBase


def new_uuid() -> str:
    """Generate a string UUID primary key (SQLite has no native UUID type)."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


__all__ = ["Base", "new_uuid", "utcnow", "date"]
