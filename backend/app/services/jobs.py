"""Background job runner + per-scan SSE event bus.

- ``ScanEventBus`` bridges the worker *thread* to the async SSE endpoints: the worker
  publishes plain-dict events; async subscribers receive them via the event loop. It keeps
  a per-scan history so a subscriber that connects mid-scan (or after it finished) still
  sees every event.
- ``JobRunner`` owns a single worker thread that runs scans serially. On failure it records
  a ``failed`` scan with stage + reason and emits a terminal ``failed`` event — never a
  fake success (guardrail).
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections import defaultdict
from collections.abc import AsyncIterator

from app.config import get_settings
from app.db.models import Scan
from app.db.session import SessionLocal
from app.pipeline.capture import FailingCapture, MockCapture
from app.pipeline.detect import MockDetector
from app.pipeline.reconstruct import MockReconstructor
from app.pipeline.runner import PipelineDeps, mark_scan_failed, run_pipeline
from app.pipeline.types import PipelineError, Stage
from app.services.artifacts import ArtifactStore
from app.services.hardware import MockHardwareService

TERMINAL_TYPES = {"complete", "failed"}

# Sentinel an operator can put in scan notes to exercise the failure path in DEMO_MODE.
DEMO_FAIL_TOKEN = "[demo:fail]"


class ScanEventBus:
    """Thread-safe pub/sub of scan progress events, bridged to the asyncio loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, scan_id: str, event: dict) -> None:
        """Called from the worker thread."""
        with self._lock:
            self._history[scan_id].append(event)
            subscribers = list(self._subscribers.get(scan_id, ()))
        if self._loop is not None:
            for q in subscribers:
                self._loop.call_soon_threadsafe(q.put_nowait, event)

    def is_terminated(self, scan_id: str) -> bool:
        with self._lock:
            hist = self._history.get(scan_id, [])
            return bool(hist) and hist[-1].get("type") in TERMINAL_TYPES

    async def subscribe(self, scan_id: str) -> AsyncIterator[dict]:
        """Yield the backlog then live events until a terminal event."""
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            backlog = list(self._history.get(scan_id, []))
            self._subscribers[scan_id].add(q)
        try:
            terminated = False
            for event in backlog:
                yield event
                if event.get("type") in TERMINAL_TYPES:
                    terminated = True
            if terminated:
                return
            while True:
                event = await q.get()
                yield event
                if event.get("type") in TERMINAL_TYPES:
                    return
        finally:
            with self._lock:
                self._subscribers[scan_id].discard(q)


class JobRunner:
    """Single-worker background runner. Scans execute serially and safely."""

    def __init__(self, bus: ScanEventBus) -> None:
        self._bus = bus
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="scan-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("__stop__")

    def submit(self, scan_id: str) -> None:
        self._queue.put(scan_id)

    # --- worker ---
    def _worker(self) -> None:
        while not self._stop.is_set():
            scan_id = self._queue.get()
            if scan_id == "__stop__":
                break
            try:
                self._run_scan(scan_id)
            except Exception as exc:  # last-resort guard: never kill the worker
                self._bus.publish(
                    scan_id,
                    {
                        "type": "failed",
                        "scan_id": scan_id,
                        "stage": "unknown",
                        "reason": f"Unexpected worker error: {exc}",
                    },
                )

    def _run_scan(self, scan_id: str) -> None:
        db = SessionLocal()
        store = ArtifactStore()
        scan = db.get(Scan, scan_id)
        if scan is None:
            db.close()
            return

        def emit(stage: Stage, progress: int, message: str) -> None:
            scan.progress = progress
            scan.current_step = stage.value
            db.commit()
            self._bus.publish(
                scan_id,
                {
                    "type": "complete" if stage is Stage.complete else "progress",
                    "scan_id": scan_id,
                    "stage": stage.value,
                    "progress": progress,
                    "message": message,
                },
            )

        try:
            deps = self._build_deps(scan)
            run_pipeline(scan, db, store, deps, emit)
        except PipelineError as exc:
            self._fail(db, scan, exc.stage, exc.reason)
        except Exception as exc:  # noqa: BLE001 — record any error as a failed scan
            self._fail(db, scan, "unknown", str(exc))
        finally:
            db.close()

    def _build_deps(self, scan: Scan) -> PipelineDeps:
        settings = get_settings()
        if not settings.demo_mode:
            # Real hardware/reconstruction/ML arrive in Milestones 5-7. Fail loud.
            raise PipelineError(
                Stage.acquire,
                "Real scanning pipeline is not implemented yet (Milestones 5-7). "
                "Enable DEMO_MODE to run a synthetic scan.",
            )
        capturer = (
            FailingCapture()
            if DEMO_FAIL_TOKEN in (scan.notes or "")
            else MockCapture(rotation_steps=settings.rotation_steps)
        )
        return PipelineDeps(
            hardware=MockHardwareService(),
            capturer=capturer,
            reconstructor=MockReconstructor(),
            detector=MockDetector(),
        )

    def _fail(self, db, scan: Scan, stage: str, reason: str) -> None:
        mark_scan_failed(db, scan, stage, reason)
        self._bus.publish(
            scan.id,
            {"type": "failed", "scan_id": scan.id, "stage": stage, "reason": reason},
        )


# Process-wide singletons, started in the app lifespan.
event_bus = ScanEventBus()
job_runner = JobRunner(event_bus)


@contextlib.asynccontextmanager
async def lifespan_jobs():
    """Wire the event loop into the bus and start the worker; stop on shutdown."""
    event_bus.set_loop(asyncio.get_running_loop())
    job_runner.start()
    try:
        yield
    finally:
        job_runner.stop()
