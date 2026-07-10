"""Geometry helpers for the DEMO mock reconstruction.

Generates a simple capped cylinder (a stand-in "limb") as an OBJ with UV coordinates,
plus a synthetic skin-tone texture. The UV parameterization is intentionally trivial so
that ``mapping.uv_to_cylinder`` can place lesion markers exactly on the surface:

    u in [0,1] -> angle theta = 2*pi*u   (around the cylinder)
    v in [0,1] -> y = (v - 0.5) * height (along the axis)
    x = radius*cos(theta),  z = radius*sin(theta)
"""

from __future__ import annotations

import math
from pathlib import Path


def uv_to_cylinder(uv_x: float, uv_y: float, radius: float, height: float) -> tuple[float, float, float]:
    """Map a UV coordinate to a 3D point on the cylinder surface."""
    theta = 2.0 * math.pi * uv_x
    x = radius * math.cos(theta)
    z = radius * math.sin(theta)
    y = (uv_y - 0.5) * height
    return (x, y, z)


def write_cylinder_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_filename: str,
    *,
    radius: float = 1.0,
    height: float = 3.0,
    radial_segments: int = 32,
    height_segments: int = 12,
) -> int:
    """Write a UV-mapped cylinder OBJ + MTL. Returns the vertex count."""
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    for j in range(height_segments + 1):
        v = j / height_segments
        y = (v - 0.5) * height
        for i in range(radial_segments + 1):
            u = i / radial_segments
            theta = 2.0 * math.pi * u
            vertices.append((radius * math.cos(theta), y, radius * math.sin(theta)))
            uvs.append((u, v))

    def idx(i: int, j: int) -> int:
        # OBJ is 1-indexed.
        return j * (radial_segments + 1) + i + 1

    faces: list[tuple[int, int, int, int]] = []
    for j in range(height_segments):
        for i in range(radial_segments):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            faces.append((a, b, c, d))

    mtl_name = mtl_path.name
    lines = [f"mtllib {mtl_name}", "usemtl skin"]
    for (x, y, z) in vertices:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    for (u, v) in uvs:
        lines.append(f"vt {u:.5f} {v:.5f}")
    for (a, b, c, d) in faces:
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c} {d}/{d}")

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mtl_path.write_text(
        "newmtl skin\n"
        "Ka 0.2 0.2 0.2\n"
        "Kd 0.8 0.65 0.55\n"
        "Ks 0.1 0.1 0.1\n"
        "Ns 10.0\n"
        f"map_Kd {texture_filename}\n",
        encoding="utf-8",
    )
    return len(vertices)


def write_skin_texture(path: Path, *, size: int = 512) -> None:
    """Write a synthetic skin-tone texture with a faint grid (clearly not real skin)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (208, 168, 143))
    draw = ImageDraw.Draw(img)
    step = size // 16
    for k in range(0, size, step):
        draw.line([(k, 0), (k, size)], fill=(198, 158, 133), width=1)
        draw.line([(0, k), (size, k)], fill=(198, 158, 133), width=1)
    img.save(path)


def write_thumbnail(src_texture: Path, dst: Path, *, size: int = 256) -> None:
    """Downscale a texture into a thumbnail PNG."""
    from PIL import Image

    img = Image.open(src_texture).convert("RGB")
    img.thumbnail((size, size))
    img.save(dst)
