"""Scan routes: create (starts a job), list, get, delete, SSE events, artifacts."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.db.models import Lesion, Patient, Scan, ScanStatus
from app.db.session import get_db
from app.schemas.scan import LesionRead, ScanAck, ScanCreate, ScanRead, ScanSummary
from app.services.artifacts import ArtifactStore
from app.services.jobs import event_bus, job_runner

router = APIRouter(prefix="/scans", tags=["scans"])


def _get_or_404(db: Session, scan_id: str) -> Scan:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
    return scan


def _thumb_url(scan: Scan) -> str | None:
    return f"/api/scans/{scan.id}/thumbnail" if scan.thumbnail_path else None


def _to_summary(scan: Scan) -> ScanSummary:
    return ScanSummary(
        id=scan.id,
        patient_id=scan.patient_id,
        body_site=scan.body_site,
        status=scan.status.value,
        progress=scan.progress,
        current_step=scan.current_step,
        concerning_count=scan.concerning_count,
        total_lesions=scan.total_lesions,
        model_version=scan.model_version,
        failure_stage=scan.failure_stage,
        failure_reason=scan.failure_reason,
        thumbnail_url=_thumb_url(scan),
        created_at=scan.created_at,
        completed_at=scan.completed_at,
    )


def _to_read(scan: Scan, lesions: list[Lesion]) -> ScanRead:
    return ScanRead(
        **_to_summary(scan).model_dump(),
        notes=scan.notes,
        vertex_count=scan.vertex_count,
        mesh_url=f"/api/scans/{scan.id}/mesh" if scan.mesh_path else None,
        texture_url=f"/api/scans/{scan.id}/texture" if scan.texture_path else None,
        started_at=scan.started_at,
        lesions=[
            LesionRead(
                id=le.id,
                uv_x=le.uv_x,
                uv_y=le.uv_y,
                x=le.x,
                y=le.y,
                z=le.z,
                bbox_w=le.bbox_w,
                bbox_h=le.bbox_h,
                area=le.area,
                classification=le.classification,
                confidence=le.confidence,
                track_id=le.track_id,
            )
            for le in lesions
        ],
    )


@router.post("", response_model=ScanAck, status_code=status.HTTP_202_ACCEPTED)
def create_scan(payload: ScanCreate, db: Session = Depends(get_db)) -> ScanAck:
    if db.get(Patient, payload.patient_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Patient {payload.patient_id} not found")

    scan = Scan(
        patient_id=payload.patient_id,
        body_site=payload.body_site,
        notes=payload.notes,
        status=ScanStatus.queued,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    job_runner.submit(scan.id)
    return ScanAck(scan_id=scan.id, status=scan.status.value)


@router.get("", response_model=list[ScanSummary])
def list_scans(
    patient_id: str | None = Query(default=None), db: Session = Depends(get_db)
) -> list[ScanSummary]:
    stmt = select(Scan).order_by(Scan.created_at.desc())
    if patient_id:
        stmt = stmt.where(Scan.patient_id == patient_id)
    return [_to_summary(s) for s in db.scalars(stmt)]


@router.get("/{scan_id}", response_model=ScanRead)
def get_scan(scan_id: str, db: Session = Depends(get_db)) -> ScanRead:
    scan = _get_or_404(db, scan_id)
    lesions = list(db.scalars(select(Lesion).where(Lesion.scan_id == scan_id)))
    return _to_read(scan, lesions)


@router.get("/{scan_id}/events")
async def scan_events(scan_id: str, db: Session = Depends(get_db)) -> EventSourceResponse:
    _get_or_404(db, scan_id)

    async def event_generator():
        async for event in event_bus.subscribe(scan_id):
            yield {"event": event["type"], "data": _dump(event)}

    return EventSourceResponse(event_generator())


def _dump(event: dict) -> str:
    import json

    return json.dumps(event)


def _serve(db: Session, scan_id: str, relpath_attr: str, media_type: str) -> FileResponse:
    scan = _get_or_404(db, scan_id)
    relpath = getattr(scan, relpath_attr)
    if not relpath:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not available for this scan")
    path = ArtifactStore().resolve(relpath)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact file missing on disk")
    return FileResponse(path, media_type=media_type)


@router.get("/{scan_id}/mesh")
def get_mesh(scan_id: str, db: Session = Depends(get_db)) -> FileResponse:
    return _serve(db, scan_id, "mesh_path", "model/obj")


@router.get("/{scan_id}/texture")
def get_texture(scan_id: str, db: Session = Depends(get_db)) -> FileResponse:
    return _serve(db, scan_id, "texture_path", "image/png")


@router.get("/{scan_id}/thumbnail")
def get_thumbnail(scan_id: str, db: Session = Depends(get_db)) -> FileResponse:
    return _serve(db, scan_id, "thumbnail_path", "image/png")


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(scan_id: str, db: Session = Depends(get_db)) -> None:
    scan = _get_or_404(db, scan_id)
    db.delete(scan)
    db.commit()
    ArtifactStore().delete_scan(scan_id)
