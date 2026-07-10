"""Capture stage: acquire images across rotation steps.

The real threaded webcam capture lands in Milestone 6. For DEMO_MODE / tests we provide
a MockCapture that writes small synthetic images so the artifact tree is realistic, plus
a FailingCapture used to exercise the failure path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from app.pipeline.types import CaptureResult, PipelineError, Stage
from app.services.artifacts import ArtifactStore

ProgressFn = Callable[[float, str], None]


class Capturer(Protocol):
    def capture(self, scan_id: str, store: ArtifactStore, progress: ProgressFn) -> CaptureResult: ...


class MockCapture:
    """Synthetic capture: writes tiny placeholder frames per rotation step."""

    def __init__(self, rotation_steps: int, camera_count: int = 1, step_delay_s: float = 0.15) -> None:
        self.rotation_steps = rotation_steps
        self.camera_count = camera_count
        self.step_delay_s = step_delay_s

    def capture(self, scan_id: str, store: ArtifactStore, progress: ProgressFn) -> CaptureResult:
        from PIL import Image

        raw_dir = store.raw_dir(scan_id)
        paths: list[str] = []
        total = max(self.rotation_steps, 1)
        for step in range(total):
            for cam in range(self.camera_count):
                shade = 150 + (step * 7 + cam * 20) % 80
                img = Image.new("RGB", (64, 64), (shade, shade - 20, shade - 40))
                fname = raw_dir / f"step{step:02d}_cam{cam}.png"
                img.save(fname)
                paths.append(store.relative(fname))
            time.sleep(self.step_delay_s)
            progress((step + 1) / total, f"Captured rotation step {step + 1}/{total}")
        return CaptureResult(
            image_paths=paths, rotation_steps=self.rotation_steps, camera_count=self.camera_count
        )


class FailingCapture:
    """Capturer that always raises — used to demo/test the failure path."""

    def __init__(self, reason: str = "Synthetic capture failure (demo)") -> None:
        self.reason = reason

    def capture(self, scan_id: str, store: ArtifactStore, progress: ProgressFn) -> CaptureResult:
        raise PipelineError(Stage.capture, self.reason)
