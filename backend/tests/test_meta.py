"""Contract tests for the meta/health routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str)
    assert isinstance(body["demo_mode"], bool)


def test_config_safe_subset(client: TestClient) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    # Only the safe subset is exposed — no serial port, db url, or paths.
    assert set(body) == {
        "app_name",
        "demo_mode",
        "rotation_steps",
        "angle_step_deg",
        "camera_indices",
        "serial_baud",
    }
