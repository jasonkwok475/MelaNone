"""Database engine + session management."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _ensure_sqlite_parent(db_url: str) -> None:
    """Create the parent directory for a local SQLite file if needed."""
    prefix = "sqlite:///"
    if db_url.startswith(prefix):
        path = Path(db_url[len(prefix):])
        path.parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(db_url: str | None = None) -> Engine:
    settings = get_settings()
    url = db_url or settings.db_url
    _ensure_sqlite_parent(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):
        # Enforce foreign keys (SQLite has them off by default) so ON DELETE CASCADE works.
        @event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_conn, _record) -> None:  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


# Process-wide engine + session factory.
engine: Engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
