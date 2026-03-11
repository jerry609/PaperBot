from __future__ import annotations

import asyncio
import json

import pytest

from paperbot.api.streaming import StreamEvent, sse_response, wrap_generator


async def _simple_stream():
    yield StreamEvent(type="progress", data={"phase": "judge", "message": "running"})
    yield StreamEvent(type="search_done", data={"ok": True})
    yield StreamEvent(type="result", data={"ok": True})


@pytest.mark.asyncio
async def test_wrap_generator_injects_envelope():
    payloads = []
    async for raw in wrap_generator(
        _simple_stream(),
        workflow="paperscool_analyze",
        run_id="run_x",
        trace_id="trace_x",
    ):
        if not raw.startswith("data: "):
            continue
        data = raw.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))

    assert len(payloads) == 3
    for idx, payload in enumerate(payloads, start=1):
        env = payload["envelope"]
        assert env["workflow"] == "paperscool_analyze"
        assert env["run_id"] == "run_x"
        assert env["trace_id"] == "trace_x"
        assert env["seq"] == idx
        assert isinstance(env["ts"], str)
        assert env["event"] in {"progress", "result"}

    assert payloads[0]["envelope"]["phase"] == "judge"
    assert payloads[0]["event"] == "progress"
    assert payloads[1]["event"] == "progress"
    assert payloads[2]["event"] == "result"


@pytest.mark.asyncio
async def test_wrap_generator_emits_heartbeat_for_slow_stream():
    async def _slow_stream():
        await asyncio.sleep(0.02)
        yield StreamEvent(type="result", data={"ok": True})

    frames = []
    async for raw in wrap_generator(
        _slow_stream(),
        workflow="slow",
        heartbeat_seconds=0.005,
        timeout_seconds=1.0,
    ):
        frames.append(raw)

    assert any(frame.startswith(": keepalive") for frame in frames)
    assert any('"type": "result"' in frame for frame in frames if frame.startswith("data: "))


@pytest.mark.asyncio
async def test_wrap_generator_times_out_and_closes_generator():
    state = {"closed": False}

    async def _never_finishes():
        try:
            await asyncio.sleep(1.0)
            yield StreamEvent(type="progress", data={"ok": True})
        finally:
            state["closed"] = True

    frames = []
    async for raw in wrap_generator(
        _never_finishes(),
        workflow="timeout",
        heartbeat_seconds=0.005,
        timeout_seconds=0.02,
    ):
        frames.append(raw)

    assert any("stream timed out" in frame for frame in frames)
    assert frames[-1] == "data: [DONE]\n\n"
    assert state["closed"] is True


def test_sse_response_includes_nginx_buffering_header():
    async def _stream():
        yield StreamEvent(type="done", data={"ok": True})

    response = sse_response(_stream(), workflow="headers")

    assert response.headers["X-Accel-Buffering"] == "no"
    assert response.media_type == "text/event-stream"
