"""Hardware service: wraps the ESP32 serial protocol behind a mockable interface.

Serial protocol (from v1 firmware — DO NOT redesign, only wrap/harden):
  Host -> device  3 : home/reset against limit switch. Device replies 2 when done.
  Host -> device  1 : advance one rotation step. Device replies 2 on complete,
                      -1 when it has reached end-of-travel.
  Baud 115200.

The real ``SerialHardwareService`` (with timeouts + port auto-detect) lands in Milestone 6.
Here we define the interface + ``MockHardwareService`` for DEMO_MODE / tests.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class StepResult(str, enum.Enum):
    complete = "complete"
    end_of_travel = "end_of_travel"


@dataclass
class HardwareStatus:
    connected: bool
    position_index: int
    last_message: str | None = None


class HardwareError(Exception):
    """Raised on hardware/serial failure (timeouts, disconnects, protocol errors)."""


@runtime_checkable
class HardwareService(Protocol):
    def connect(self) -> None: ...
    def home(self) -> None: ...
    def advance_step(self) -> StepResult: ...
    def status(self) -> HardwareStatus: ...
    def emergency_stop(self) -> None: ...
    def disconnect(self) -> None: ...


class MockHardwareService:
    """In-memory hardware stand-in. Clearly labeled synthetic — DEMO / tests only."""

    def __init__(self, travel_limit: int = 1_000) -> None:
        self._connected = False
        self._position = 0
        self._travel_limit = travel_limit
        self._last_message: str | None = None

    def connect(self) -> None:
        self._connected = True
        self._last_message = "mock connected"

    def home(self) -> None:
        if not self._connected:
            raise HardwareError("home() called before connect()")
        self._position = 0
        self._last_message = "homed"

    def advance_step(self) -> StepResult:
        if not self._connected:
            raise HardwareError("advance_step() called before connect()")
        if self._position >= self._travel_limit:
            self._last_message = "end of travel"
            return StepResult.end_of_travel
        self._position += 1
        self._last_message = f"advanced to {self._position}"
        return StepResult.complete

    def status(self) -> HardwareStatus:
        return HardwareStatus(
            connected=self._connected,
            position_index=self._position,
            last_message=self._last_message,
        )

    def emergency_stop(self) -> None:
        self._last_message = "E-STOP"

    def disconnect(self) -> None:
        self._connected = False
        self._last_message = "disconnected"
