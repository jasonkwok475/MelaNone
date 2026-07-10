"""Reconstruct stage: images -> mesh + texture.

The real classical-CV reconstruction (SfM -> point cloud -> Poisson mesh) lands in
Milestone 5, ported from v1 and stripped of all blocking GUI (``draw_geometries``) calls.
For DEMO_MODE / tests, MockReconstructor writes a UV-mapped cylinder + synthetic texture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from app.pipeline import geometry
from app.pipeline.types import CaptureResult, MeshResult
from app.services.artifacts import (
    ArtifactStore,
    MESH_MTL,
    MESH_OBJ,
    THUMBNAIL_PNG,
    TEXTURE_PNG,
)

ProgressFn = Callable[[float, str], None]


class Reconstructor(Protocol):
    def reconstruct(
        self, scan_id: str, capture: CaptureResult, store: ArtifactStore, progress: ProgressFn
    ) -> MeshResult: ...


class MockReconstructor:
    """Synthetic reconstruction: a capped cylinder standing in for a limb."""

    def __init__(self, radius: float = 1.0, height: float = 3.0) -> None:
        self.radius = radius
        self.height = height

    def reconstruct(
        self, scan_id: str, capture: CaptureResult, store: ArtifactStore, progress: ProgressFn
    ) -> MeshResult:
        progress(0.2, "Building surface")
        texture_path = store.path_for(scan_id, TEXTURE_PNG)
        geometry.write_skin_texture(texture_path)

        obj_path = store.path_for(scan_id, MESH_OBJ)
        mtl_path = store.path_for(scan_id, MESH_MTL)
        vertex_count = geometry.write_cylinder_obj(
            obj_path, mtl_path, TEXTURE_PNG, radius=self.radius, height=self.height
        )

        progress(0.7, "Generating thumbnail")
        thumb_path = store.path_for(scan_id, THUMBNAIL_PNG)
        geometry.write_thumbnail(texture_path, thumb_path)

        progress(1.0, "Mesh ready")
        return MeshResult(
            obj_path=store.relative(obj_path),
            mtl_path=store.relative(mtl_path),
            texture_path=store.relative(texture_path),
            thumbnail_path=store.relative(thumb_path),
            vertex_count=vertex_count,
            radius=self.radius,
            height=self.height,
        )
