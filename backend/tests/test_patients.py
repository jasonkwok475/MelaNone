"""Patient CRUD API tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_create_and_get_patient(client: TestClient) -> None:
    resp = client.post("/api/patients", json={"display_id": "P-001", "consent_ack": True})
    assert resp.status_code == 201
    created = resp.json()
    assert created["display_id"] == "P-001"
    assert created["consent_ack"] is True
    assert created["name"] is None
    pid = created["id"]

    got = client.get(f"/api/patients/{pid}")
    assert got.status_code == 200
    assert got.json()["id"] == pid


def test_list_patients(client: TestClient) -> None:
    assert client.get("/api/patients").json() == []
    client.post("/api/patients", json={"display_id": "P-001"})
    client.post("/api/patients", json={"display_id": "P-002"})
    listing = client.get("/api/patients").json()
    assert {p["display_id"] for p in listing} == {"P-001", "P-002"}


def test_update_patient(client: TestClient) -> None:
    pid = client.post("/api/patients", json={"display_id": "P-001"}).json()["id"]
    resp = client.patch(f"/api/patients/{pid}", json={"name": "Alex", "notes": "left arm mole"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Alex"
    assert body["notes"] == "left arm mole"
    # Unset fields are unchanged.
    assert body["display_id"] == "P-001"


def test_delete_patient(client: TestClient) -> None:
    pid = client.post("/api/patients", json={"display_id": "P-001"}).json()["id"]
    assert client.delete(f"/api/patients/{pid}").status_code == 204
    assert client.get(f"/api/patients/{pid}").status_code == 404


def test_get_missing_patient_404(client: TestClient) -> None:
    assert client.get("/api/patients/does-not-exist").status_code == 404


def test_create_requires_display_id(client: TestClient) -> None:
    # Empty display_id violates min_length -> 422 validation error.
    assert client.post("/api/patients", json={"display_id": ""}).status_code == 422
