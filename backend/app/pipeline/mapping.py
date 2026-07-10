"""Map stage: UV lesion coordinates -> 3D points on the mesh surface.

This is the concrete mechanism behind "mark the lesion's coordinates on the 3D model".
For the DEMO cylinder mesh the mapping is exact (see geometry.uv_to_cylinder). The real
inverse-UV mapping against the reconstructed mesh lands alongside Milestone 7.
"""

from __future__ import annotations

from app.pipeline import geometry
from app.pipeline.types import DetectedLesion, MeshResult


def map_lesions_to_mesh(lesions: list[DetectedLesion], mesh: MeshResult) -> list[DetectedLesion]:
    """Fill each lesion's (x, y, z) from its UV coord on the given mesh."""
    for lesion in lesions:
        x, y, z = geometry.uv_to_cylinder(lesion.uv_x, lesion.uv_y, mesh.radius, mesh.height)
        lesion.x, lesion.y, lesion.z = x, y, z
    return lesions
