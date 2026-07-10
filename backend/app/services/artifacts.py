"""ArtifactStore — filesystem abstraction over ``artifacts/<scan_id>/``.

Keeps binary blobs (captured images, mesh, texture, thumbnails) out of the database;
the DB stores only relative paths. Providing this behind a small interface means the
storage backend can change later without touching the pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.config import get_settings

# Canonical filenames within a scan's artifact directory.
RAW_DIR = "raw"
MESH_OBJ = "mesh.obj"
MESH_MTL = "mesh.mtl"
TEXTURE_PNG = "texture.png"
THUMBNAIL_PNG = "thumbnail.png"
LESIONS_JSON = "lesions.json"


class ArtifactStore:
    """Manages per-scan artifact directories under a configured root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_settings().artifact_root

    # --- directories ---
    def scan_dir(self, scan_id: str, *, create: bool = True) -> Path:
        path = self.root / scan_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def raw_dir(self, scan_id: str, *, create: bool = True) -> Path:
        path = self.scan_dir(scan_id, create=create) / RAW_DIR
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, scan_id: str, name: str, *, create_parent: bool = True) -> Path:
        """Absolute path for a named artifact within a scan's directory."""
        path = self.scan_dir(scan_id, create=create_parent) / name
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path) -> str:
        """Path relative to the artifact root, for storing in the DB (POSIX form)."""
        return path.relative_to(self.root).as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Inverse of :meth:`relative` — absolute path from a stored relative path."""
        return self.root / relative_path

    # --- io ---
    def write_bytes(self, scan_id: str, name: str, data: bytes) -> Path:
        path = self.path_for(scan_id, name)
        path.write_bytes(data)
        return path

    def exists(self, scan_id: str) -> bool:
        return (self.root / scan_id).is_dir()

    def delete_scan(self, scan_id: str) -> None:
        """Remove a scan's entire artifact directory (privacy: hard delete)."""
        path = self.root / scan_id
        if path.is_dir():
            shutil.rmtree(path)
