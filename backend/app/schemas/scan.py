"""Scan request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScanCreate(BaseModel):
    patient_id: str
    body_site: str = Field(min_length=1, max_length=64)
    notes: str = ""


class ScanAck(BaseModel):
    """Returned immediately from POST /scans (the job runs in the background)."""

    scan_id: str
    status: str


class LesionRead(BaseModel):
    id: str
    uv_x: float
    uv_y: float
    x: float
    y: float
    z: float
    bbox_w: float | None
    bbox_h: float | None
    area: float | None
    classification: str
    confidence: float
    track_id: str | None


class ScanSummary(BaseModel):
    id: str
    patient_id: str
    body_site: str
    status: str
    progress: int
    current_step: str | None
    concerning_count: int
    total_lesions: int
    model_version: str | None
    failure_stage: str | None
    failure_reason: str | None
    thumbnail_url: str | None
    created_at: datetime
    completed_at: datetime | None


class ScanRead(ScanSummary):
    notes: str
    vertex_count: int | None
    mesh_url: str | None
    texture_url: str | None
    started_at: datetime | None
    lesions: list[LesionRead]
