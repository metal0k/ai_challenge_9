"""Wire tap: mirrors frames, changes nothing."""

from __future__ import annotations

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCRequest, JSONRPCResponse

from advent_core.mcp_wire import TapTransport, frame_line

PING = SessionMessage(JSONRPCRequest(jsonrpc="2.0", id=1, method="ping"))
PONG = SessionMessage(JSONRPCResponse(jsonrpc="2.0", id=1, result={}))


class FakeTransport:
    """Inner transport: a memory pipe in each direction, plus enter/exit bookkeeping."""

    def __init__(self) -> None:
        self.to_client_tx, self.to_client_rx = anyio.create_memory_object_stream(10)
        self.to_server_tx, self.to_server_rx = anyio.create_memory_object_stream(10)
        self.exits: list[type[BaseException] | None] = []

    async def __aenter__(self):
        return self.to_client_rx, self.to_server_tx

    async def __aexit__(self, exc_type, exc, tb):
        self.exits.append(exc_type)
        return None


def test_frame_line_uses_literal_arrow_and_compact_json():
    assert frame_line("→", PING) == '→ {"jsonrpc":"2.0","id":1,"method":"ping"}'
    assert frame_line("←", PONG) == '← {"jsonrpc":"2.0","id":1,"result":{}}'


def test_frame_line_marks_a_non_frame_item():
    line = frame_line("←", ValueError("bad json"))
    assert line == "← [не кадр: ValueError] bad json"


def test_both_directions_are_mirrored_and_delivered_unchanged():
    lines: list[str] = []
    inner = FakeTransport()
    received: list = []

    async def scenario() -> None:
        async with TapTransport(inner, lines.append) as (read, write):
            await write.send(PING)
            await inner.to_client_tx.send(PONG)
            received.append(await read.receive())

    anyio.run(scenario)
    assert lines == [
        '→ {"jsonrpc":"2.0","id":1,"method":"ping"}',
        '← {"jsonrpc":"2.0","id":1,"result":{}}',
    ]
    assert received == [PONG]
    assert inner.to_server_rx.receive_nowait() is PING


def test_async_iteration_is_mirrored_and_ends_with_the_inner_stream():
    lines: list[str] = []
    inner = FakeTransport()
    seen: list = []

    async def scenario() -> None:
        await inner.to_client_tx.send(PONG)
        await inner.to_client_tx.aclose()
        async with TapTransport(inner, lines.append) as (read, _write):
            async for item in read:
                seen.append(item)

    anyio.run(scenario)
    assert seen == [PONG]
    assert lines == ['← {"jsonrpc":"2.0","id":1,"result":{}}']


def test_exception_inside_the_block_propagates_and_reaches_inner_exit():
    inner = FakeTransport()

    async def scenario() -> None:
        async with TapTransport(inner, lambda line: None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        anyio.run(scenario)
    assert inner.exits == [RuntimeError]


def test_inner_stream_errors_are_not_swallowed():
    inner = FakeTransport()

    async def scenario() -> None:
        async with TapTransport(inner, lambda line: None) as (read, _write):
            await inner.to_client_tx.aclose()
            await read.receive()

    with pytest.raises(anyio.EndOfStream):
        anyio.run(scenario)


@pytest.mark.parametrize("error", [OSError("stderr closed"), RuntimeError("sink bug")])
def test_a_failing_sink_does_not_break_the_transport(error):
    inner = FakeTransport()

    def broken(_line: str) -> None:
        raise error

    async def scenario() -> None:
        async with TapTransport(inner, broken) as (_read, write):
            await write.send(PING)

    anyio.run(scenario)
    assert inner.to_server_rx.receive_nowait() is PING


def test_an_unserialisable_item_does_not_break_the_transport():
    inner = FakeTransport()
    lines: list[str] = []
    bad = SessionMessage(message=object())

    async def scenario() -> None:
        async with TapTransport(inner, lines.append) as (_read, write):
            await write.send(bad)

    anyio.run(scenario)
    assert lines == []
    assert inner.to_server_rx.receive_nowait() is bad


def test_on_error_sees_non_frame_items_only():
    inner = FakeTransport()
    errors: list[Exception] = []
    problem = ValueError("bad json")

    async def scenario() -> None:
        await inner.to_client_tx.send(PONG)
        await inner.to_client_tx.send(problem)
        async with TapTransport(inner, lambda line: None, errors.append) as (read, _w):
            await read.receive()
            await read.receive()

    anyio.run(scenario)
    assert errors == [problem]
