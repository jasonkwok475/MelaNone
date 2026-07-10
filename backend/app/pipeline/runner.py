"""Pipeline orchestration: run stages [0]-[6], emit progress, persist results.

This function is transport-agnostic: it takes an ``emit`` callback and the stage
implementations (real or mock) and runs them in order. On any stage failure it lets the
``PipelineError`` propagate — the caller (JobRunner) records it as a ``failed`` scan.
NEVER substitutes fake output for a failed stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import Lesion, ModelRegistry, Scan, ScanStatus
from app.pipeline.capture import Capturer
from app.pipeline.detect import Detector
from app.pipeline.mapping import map_lesions_to_mesh
from app.pipeline.reconstruct import Reconstructor
from app.pipeline.types import Stage
from app.services.artifacts import LESIONS_JSON, ArtifactStore

# Progress is reported globally 0-100. Each stage owns a slice; a stage's local 0..1
# progress is scaled into its slice.
_STAGE_RANGES: dict[Stage, tuple[int, int]] = {
    Stage.acquire: (0, 5),
    Stage.capture: (5, 40),
    Stage.reconstruct: (40, 65),
    Stage.detect: (65, 82),
    Stage.map: (82, 90),
    Stage.persist: (90, 100),
}

# Classifications considered "concerning" for the denormalized count / flags.
CONCERNING_CLASSES = {"melanoma"}

# Emits (stage, progress_0_100, message).
EmitFn = Callable[[Stage, int, str], None]


@dataclass
class PipelineDeps:
    hardware: object  # HardwareService (only used for acquire in demo)
    capturer: Capturer
    reconstructor: Reconstructor
    detector: Detector


def _scaled(stage: Stage, local: float) -> int:
    lo, hi = _STAGE_RANGES[stage]
    return int(lo + (hi - lo) * max(0.0, min(1.0, local)))


def run_pipeline(
    scan: Scan,
    db: Session,
    store: ArtifactStore,
    deps: PipelineDeps,
    emit: EmitFn,
) -> None:
    """Run the full pipeline for ``scan``, updating the row and emitting progress.

    Raises PipelineError if a stage fails (the caller marks the scan failed).
    """
    scan.status = ScanStatus.running
    scan.started_at = utcnow()
    db.commit()

    # [0] acquire hardware
    emit(Stage.acquire, _scaled(Stage.acquire, 0.0), "Acquiring hardware")
    deps.hardware.connect()
    deps.hardware.home()
    emit(Stage.acquire, _scaled(Stage.acquire, 1.0), "Hardware ready")

    # [1] capture
    capture = deps.capturer.capture(
        scan.id, store, lambda p, msg: emit(Stage.capture, _scaled(Stage.capture, p), msg)
    )

    # [2] reconstruct
    mesh = deps.reconstructor.reconstruct(
        scan.id,
        capture,
        store,
        lambda p, msg: emit(Stage.reconstruct, _scaled(Stage.reconstruct, p), msg),
    )

    # [3] detect + localize
    lesions = deps.detector.detect(
        mesh.texture_path, lambda p, msg: emit(Stage.detect, _scaled(Stage.detect, p), msg)
    )

    # [4] map UV -> 3D
    emit(Stage.map, _scaled(Stage.map, 0.0), "Mapping lesions to surface")
    map_lesions_to_mesh(lesions, mesh)
    emit(Stage.map, _scaled(Stage.map, 1.0), f"Mapped {len(lesions)} lesions")

    # [5] persist
    emit(Stage.persist, _scaled(Stage.persist, 0.0), "Saving results")
    _register_model(db, deps.detector)

    concerning = 0
    for det in lesions:
        db.add(
            Lesion(
                scan_id=scan.id,
                uv_x=det.uv_x,
                uv_y=det.uv_y,
                x=det.x or 0.0,
                y=det.y or 0.0,
                z=det.z or 0.0,
                bbox_w=det.bbox_w,
                bbox_h=det.bbox_h,
                area=det.area,
                classification=det.classification,
                confidence=det.confidence,
            )
        )
        if det.classification in CONCERNING_CLASSES:
            concerning += 1

    # Cache the lesion list as an artifact too.
    store.write_bytes(
        scan.id,
        LESIONS_JSON,
        json.dumps([det.__dict__ for det in lesions], indent=2).encode("utf-8"),
    )

    scan.mesh_path = mesh.obj_path
    scan.texture_path = mesh.texture_path
    scan.thumbnail_path = mesh.thumbnail_path
    scan.vertex_count = mesh.vertex_count
    scan.total_lesions = len(lesions)
    scan.concerning_count = concerning
    scan.model_version = getattr(deps.detector, "model_version", None)
    scan.status = ScanStatus.complete
    scan.progress = 100
    scan.current_step = Stage.complete.value
    scan.completed_at = utcnow()
    db.commit()

    emit(Stage.complete, 100, f"Scan complete — {len(lesions)} lesions, {concerning} concerning")


def mark_scan_failed(db: Session, scan: Scan, stage: str, reason: str) -> None:
    """Record a failed scan (fail loud). Never leaves a half-scan looking complete."""
    scan.status = ScanStatus.failed
    scan.failure_stage = stage
    scan.failure_reason = reason
    scan.current_step = stage
    scan.completed_at = utcnow()
    db.commit()


def _register_model(db: Session, detector: Detector) -> None:
    """Upsert the detector version into the model registry."""
    version = getattr(detector, "model_version", None)
    if not version:
        return
    if db.get(ModelRegistry, version) is None:
        db.add(
            ModelRegistry(
                version=version,
                kind="detector",
                notes="Synthetic DEMO detector — not a real model.",
            )
        )
        db.flush()
