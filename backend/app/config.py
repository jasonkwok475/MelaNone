"""Environment-driven application settings.

Every value the old code hardcoded (serial port, camera indices, image paths, rotation
count) lives here instead, sourced from environment variables or a local ``.env`` file.
No machine-specific value should ever be baked into a module again.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (backend/app/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application configuration.

    Values are read from environment variables (case-insensitive) or a ``.env`` file at
    the repo root. Complex fields such as ``camera_indices`` accept either a JSON list
    (``[1,2]``) or a comma-separated string (``1,2``).
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App / API ---
    app_name: str = "MelaNone"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Frontend dev origins allowed for CORS.
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://127.0.0.1:5173"])

    # --- Demo mode ---
    # When true, hardware + ML are mocked with clearly-labeled synthetic data so the app
    # runs with no rig attached. MUST default to false (guardrail).
    demo_mode: bool = False

    # --- Hardware / serial ---
    # None => auto-detect via pyserial.tools.list_ports at connect time.
    serial_port: str | None = None
    serial_baud: int = 115200
    serial_timeout_s: float = 5.0

    # --- Capture ---
    # v1 used devices 1 and 3; default to 1,2 and make it configurable.
    camera_indices: list[int] = Field(default=[1, 2])
    rotation_steps: int = 4
    angle_step_deg: float = 90.0

    # --- Storage ---
    artifact_root: Path = REPO_ROOT / "artifacts"
    db_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'melanone.db').as_posix()}"

    @field_validator("cors_origins", "camera_indices", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated strings in addition to JSON lists for list fields."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("artifact_root", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide settings instance."""
    return Settings()
