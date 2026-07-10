"""Detect + localize stage: find and classify lesions on the surface.

This is the capability v1 lacked — *localized* lesions with coordinates, not a single
whole-image guess. The real detector + transfer-learning classifier lands in Milestone 7
behind this same interface. For DEMO_MODE / tests, MockDetector emits deterministic,
clearly-synthetic lesions (seeded by scan_id) with calibrated-looking confidences.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol

from app.pipeline.types import DetectedLesion

ProgressFn = Callable[[float, str], None]

# Demo class distribution — mostly benign, occasionally concerning. Purely synthetic.
_DEMO_CLASSES = [
    ("benign", 0.45),
    ("nevus", 0.30),
    ("keratosis", 0.12),
    ("melanoma", 0.13),
]

MODEL_VERSION = "demo-mock-0"


class Detector(Protocol):
    model_version: str

    def detect(self, texture_relpath: str, progress: ProgressFn) -> list[DetectedLesion]: ...


class MockDetector:
    """Deterministic synthetic detector. NOT a real model — DEMO only."""

    model_version = MODEL_VERSION

    def __init__(self, seed_salt: str = "") -> None:
        self.seed_salt = seed_salt

    def detect(self, texture_relpath: str, progress: ProgressFn) -> list[DetectedLesion]:
        rng = random.Random(f"{self.seed_salt}:{texture_relpath}")
        count = rng.randint(3, 6)
        classes = [c for c, _ in _DEMO_CLASSES]
        weights = [w for _, w in _DEMO_CLASSES]

        lesions: list[DetectedLesion] = []
        for i in range(count):
            cls = rng.choices(classes, weights=weights, k=1)[0]
            # Melanoma-ish spots get higher (still uncertain) confidence in the demo.
            base = 0.72 if cls == "melanoma" else 0.55
            conf = round(min(0.97, base + rng.uniform(0.0, 0.25)), 3)
            w = round(rng.uniform(0.02, 0.06), 4)
            h = round(rng.uniform(0.02, 0.06), 4)
            lesions.append(
                DetectedLesion(
                    uv_x=round(rng.uniform(0.05, 0.95), 4),
                    uv_y=round(rng.uniform(0.1, 0.9), 4),
                    classification=cls,
                    confidence=conf,
                    bbox_w=w,
                    bbox_h=h,
                    area=round(w * h, 6),
                )
            )
            progress((i + 1) / count, f"Analyzed spot {i + 1}/{count}")
        return lesions
