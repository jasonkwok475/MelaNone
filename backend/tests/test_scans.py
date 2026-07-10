"""Scan API contract tests (job execution itself is covered in test_pipeline)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _make_patient(client: TestClient) -> str:
    return client.post("/api/patients", json={"display_id": "P-1"}).json()["id"]


def test_create_scan_returns_ack(client: TestClient) -> None:
    pid = _make_patient(client)
    resp = client.post("/api/scans", json={"patient_id": pid, "body_site": "left_forearm"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["scan_id"]


def test_create_scan_unknown_patient_404(client: TestClient) -> None:
    resp = client.post("/api/scans", json={"patient_id": "nope", "body_site": "arm"})
    assert resp.status_code == 404


def test_list_and_get_scan(client: TestClient) -> None:
    pid = _make_patient(client)
    scan_id = client.post(
        "/api/scans", json={"patient_id": pid, "body_site": "left_forearm"}
    ).json()["scan_id"]

    listing = client.get(f"/api/scans?patient_id={pid}").json()
    assert any(s["id"] == scan_id for s in listing)

    detail = client.get(f"/api/scans/{scan_id}").json()
    assert detail["id"] == scan_id
    assert detail["body_site"] == "left_forearm"
    assert "lesions" in detail


def test_get_missing_scan_404(client: TestClient) -> None:
    assert client.get("/api/scans/nope").status_code == 404


def test_events_endpoint_requires_existing_scan(client: TestClient) -> None:
    assert client.get("/api/scans/nope/events").status_code == 404
