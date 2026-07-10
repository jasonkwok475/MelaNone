"""ArtifactStore tests."""

from __future__ import annotations

from pathlib import Path

from app.services.artifacts import ArtifactStore


def test_write_and_relative_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    path = store.write_bytes("scan-1", "texture.png", b"pixels")
    assert path.read_bytes() == b"pixels"

    rel = store.relative(path)
    assert rel == "scan-1/texture.png"
    assert store.resolve(rel) == path


def test_raw_dir_created(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    raw = store.raw_dir("scan-1")
    assert raw.is_dir()
    assert raw.name == "raw"


def test_delete_scan_removes_dir(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    store.write_bytes("scan-1", "mesh.obj", b"data")
    assert store.exists("scan-1")

    store.delete_scan("scan-1")
    assert not store.exists("scan-1")
    # Deleting a non-existent scan is a no-op, not an error.
    store.delete_scan("scan-1")
