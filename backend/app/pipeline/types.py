"""Shared pipeline types: errors, stages, and dataclasses."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Stage(str, enum.Enum):
    """Ordered pipeline stages (also used as progress step labels)."""

    acquire = "acquire"
    capture = "capture"
    reconstruct = "reconstruct"
    detect = "detect"
    map = "map"
    persist = "persist"
    complete = "complete"


class PipelineError(Exception):
    """Raised by a stage on failure. Carries the stage and a human-readable reason.

    The job runner turns this into a ``failed`` scan — never a fake success.
    """

    def __init__(self, stage: Stage | str, reason: str) -> None:
        self.stage = stage.value if isinstance(stage, Stage) else str(stage)
        self.reason = reason
        super().__init__(f"[{self.stage}] {reason}")


@dataclass
class CaptureResult:
    """Raw images captured for a scan, one entry per (camera, rotation-step)."""

    image_paths: list[str]
    rotation_steps: int
    camera_count: int


@dataclass
class MeshResult:
    """Reconstructed mesh + texture artifacts (relative artifact-store paths)."""

    obj_path: str
    texture_path: str
    vertex_count: int
    mtl_path: str | None = None
    thumbnail_path: str | None = None
    # Geometry parameters needed to map UV coords onto this mesh's surface.
    radius: float = 1.0
    height: float = 1.0


@dataclass
class DetectedLesion:
    """A localized, classified lesion. UV coords come from detect; xyz from mapping."""

    uv_x: float
    uv_y: float
    classification: str
    confidence: float
    bbox_w: float | None = None
    bbox_h: float | None = None
    area: float | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
