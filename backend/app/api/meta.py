"""Meta / health routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import get_settings

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: str
    version: str
    demo_mode: bool


class SafeConfig(BaseModel):
    """A non-sensitive subset of settings, safe to expose to the frontend."""

    app_name: str
    demo_mode: bool
    rotation_steps: int
    angle_step_deg: float
    camera_indices: list[int]
    serial_baud: int


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns ok when the API process is up."""
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, demo_mode=settings.demo_mode)


@router.get("/config", response_model=SafeConfig)
def config() -> SafeConfig:
    """Expose the safe subset of configuration the dashboard needs."""
    s = get_settings()
    return SafeConfig(
        app_name=s.app_name,
        demo_mode=s.demo_mode,
        rotation_steps=s.rotation_steps,
        angle_step_deg=s.angle_step_deg,
        camera_indices=s.camera_indices,
        serial_baud=s.serial_baud,
    )
