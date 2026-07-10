"""ScanEventBus tests."""

from __future__ import annotations

import asyncio

from app.services.jobs import ScanEventBus


def test_event_bus_replays_backlog_and_stops_on_terminal() -> None:
    async def run() -> list[dict]:
        bus = ScanEventBus()
        bus.set_loop(asyncio.get_running_loop())
        bus.publish("s1", {"type": "progress", "progress": 10})
        bus.publish("s1", {"type": "progress", "progress": 50})
        bus.publish("s1", {"type": "complete", "progress": 100})
        collected: list[dict] = []
        async for event in bus.subscribe("s1"):
            collected.append(event)
        return collected

    events = asyncio.run(run())
    assert [e["type"] for e in events] == ["progress", "progress", "complete"]


def test_event_bus_delivers_live_events() -> None:
    async def run() -> list[str]:
        bus = ScanEventBus()
        bus.set_loop(asyncio.get_running_loop())
        collected: list[str] = []

        async def consume() -> None:
            async for event in bus.subscribe("s2"):
                collected.append(event["type"])

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let the subscriber register
        bus.publish("s2", {"type": "progress"})
        bus.publish("s2", {"type": "failed", "reason": "boom"})
        await asyncio.wait_for(task, timeout=1.0)
        return collected

    events = asyncio.run(run())
    assert events == ["progress", "failed"]
