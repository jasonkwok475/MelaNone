"""Pipeline runner tests — happy path AND failure path.

The failure path is the important one: a stage that raises must yield a FAILED scan with
a recorded stage/reason and NO persisted lesions — never a fake success.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db.models import Lesion, ModelRegistry, Patient, Scan, ScanStatus
from app.pipeline.capture import FailingCapture, MockCapture
from app.pipeline.detect import MockDetector
from app.pipeline.reconstruct import MockReconstructor
from app.pipeline.runner import PipelineDeps, mark_scan_failed, run_pipeline
from app.pipeline.types import PipelineError, Stage
from app.services.artifacts import ArtifactStore
from app.services.hardware import MockHardwareService


def _make_scan(db: Session, notes: str = "") -> Scan:
    patient = Patient(display_id="P-1")
    db.add(patient)
    db.flush()
    scan = Scan(patient_id=patient.id, body_site="left_forearm", notes=notes)
    db.add(scan)
    db.commit()
    return scan


def _deps(capturer) -> PipelineDeps:
    return PipelineDeps(
        hardware=MockHardwareService(),
        capturer=capturer,
        reconstructor=MockReconstructor(),
        detector=MockDetector(),
    )


def test_run_pipeline_happy_path(db_session: Session, tmp_path: Path) -> None:
    scan = _make_scan(db_session)
    store = ArtifactStore(root=tmp_path)
    events: list[tuple[str, int]] = []

    run_pipeline(
        scan,
        db_session,
        store,
        _deps(MockCapture(rotation_steps=2, step_delay_s=0.0)),
        lambda stage, pct, msg: events.append((stage.value, pct)),
    )

    assert scan.status is ScanStatus.complete
    assert scan.progress == 100
    assert scan.total_lesions > 0
    assert scan.model_version == "demo-mock-0"

    lesions = db_session.query(Lesion).filter_by(scan_id=scan.id).all()
    assert len(lesions) == scan.total_lesions
    # Every lesion was mapped to a 3D coordinate on the surface.
    assert all(le.x is not None and le.y is not None and le.z is not None for le in lesions)

    # Artifacts exist on disk.
    assert store.resolve(scan.mesh_path).exists()
    assert store.resolve(scan.texture_path).exists()
    assert store.resolve(scan.thumbnail_path).exists()

    # Model registered; final event is the terminal complete.
    assert db_session.get(ModelRegistry, "demo-mock-0") is not None
    assert events[-1][0] == Stage.complete.value and events[-1][1] == 100


def test_run_pipeline_failure_marks_failed_not_fake_success(
    db_session: Session, tmp_path: Path
) -> None:
    scan = _make_scan(db_session)
    store = ArtifactStore(root=tmp_path)

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(
            scan, db_session, store, _deps(FailingCapture()), lambda *a: None
        )
    assert exc_info.value.stage == Stage.capture.value

    # Simulate what JobRunner does on PipelineError.
    mark_scan_failed(db_session, scan, exc_info.value.stage, exc_info.value.reason)

    assert scan.status is ScanStatus.failed
    assert scan.failure_stage == "capture"
    assert scan.failure_reason
    # Crucially: no fake results were persisted.
    assert scan.total_lesions == 0
    assert db_session.query(Lesion).filter_by(scan_id=scan.id).count() == 0
