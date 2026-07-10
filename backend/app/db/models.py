"""ORM models — patients, scans, lesions, and the model registry (docs/05).

Design notes:
- UUID primary keys are stored as 36-char strings (SQLite has no UUID type).
- Timestamps are timezone-aware UTC.
- ``scans`` carries job status so a scan is a first-class, queryable job (no global
  "current analysis" singleton). Denormalized ``concerning_count`` / ``total_lesions``
  keep history lists fast.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Boolean, Date, DateTime

from app.db.base import Base, new_uuid, utcnow


class ScanStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    # Human-facing label / de-identified code. Prefer this over a real name.
    display_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Acknowledged the research-use disclaimer.
    consent_ack: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    scans: Mapped[list[Scan]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # e.g. "left_forearm" — must match across scans to be comparable.
    body_site: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[ScanStatus] = mapped_column(
        Enum(ScanStatus, native_enum=False, length=16),
        default=ScanStatus.queued,
        nullable=False,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Which detector/classifier produced results — keeps history interpretable.
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    mesh_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    texture_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    vertex_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    concerning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_lesions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="scans")
    lesions: Mapped[list[Lesion]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )


class Lesion(Base):
    __tablename__ = "lesions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    scan_id: Mapped[str] = mapped_column(
        ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Position on the UV texture.
    uv_x: Mapped[float] = mapped_column(Float, nullable=False)
    uv_y: Mapped[float] = mapped_column(Float, nullable=False)
    # Mapped 3D surface coordinate (for the marker on the mesh).
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)

    bbox_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    area: Mapped[float | None] = mapped_column(Float, nullable=True)

    # e.g. melanoma | nevus | benign | keratosis | unknown
    classification: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Links the "same" physical lesion across scans (longitudinal tracking).
    track_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    scan: Mapped[Scan] = relationship(back_populates="lesions")


class ModelRegistry(Base):
    """Registry of detector/classifier versions used to produce results."""

    __tablename__ = "models"

    version: Mapped[str] = mapped_column(String(128), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # detector | classifier
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


__all__ = ["Patient", "Scan", "Lesion", "ModelRegistry", "ScanStatus"]
