"""Wire tap: mirrors every JSON-RPC frame of a transport into a text sink.

Wraps any MCP transport (an async context manager yielding a read/write stream
pair) and changes nothing else: frames pass through untouched, exceptions
propagate.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from types import TracebackType
from typing import Any

from mcp.shared.message import SessionMessage

SENT = "→"
RECEIVED = "←"

Sink = Callable[[str], None]


def frame_line(marker: str, item: SessionMessage | Exception) -> str:
    """One line per frame; a non-message item (parse failure) is marked as such."""
    if isinstance(item, SessionMessage):
        body = item.message.model_dump_json(by_alias=True, exclude_unset=True)
        return f"{marker} {body}"
    return f"{marker} [не кадр: {type(item).__name__}] {item}"


def _emit(sink: Sink, marker: str, item: Any) -> None:
    # Formatting and the sink both live inside the guard: a tap failure of any
    # kind (closed stderr, unserialisable item) must not touch the transport.
    with suppress(Exception):
        sink(frame_line(marker, item))


def _notify(on_error: Callable[[Exception], None] | None, item: Any) -> None:
    if on_error is not None and isinstance(item, Exception):
        with suppress(Exception):
            on_error(item)


class _TappedRead:
    def __init__(
        self, inner: Any, sink: Sink, on_error: Callable[[Exception], None] | None
    ) -> None:
        self._inner = inner
        self._sink = sink
        self._on_error = on_error

    async def receive(self) -> Any:
        item = await self._inner.receive()
        _emit(self._sink, RECEIVED, item)
        _notify(self._on_error, item)
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self) -> _TappedRead:
        return self

    async def __anext__(self) -> Any:
        # StopAsyncIteration from the inner stream propagates untouched.
        item = await self._inner.__anext__()
        _emit(self._sink, RECEIVED, item)
        _notify(self._on_error, item)
        return item

    async def __aenter__(self) -> _TappedRead:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> bool | None:
        return await self._inner.__aexit__(*exc)


class _TappedWrite:
    def __init__(self, inner: Any, sink: Sink) -> None:
        self._inner = inner
        self._sink = sink

    async def send(self, item: SessionMessage, /) -> None:
        _emit(self._sink, SENT, item)
        await self._inner.send(item)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _TappedWrite:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> bool | None:
        return await self._inner.__aexit__(*exc)


class TapTransport:
    """Delegates to `inner`, mirrors both directions to `sink`.

    `on_error` sees every non-frame item (a parse failure) arriving from the server.
    """

    def __init__(
        self, inner: Any, sink: Sink, on_error: Callable[[Exception], None] | None = None
    ) -> None:
        self._inner = inner
        self._sink = sink
        self._on_error = on_error

    async def __aenter__(self) -> tuple[_TappedRead, _TappedWrite]:
        read, write = await self._inner.__aenter__()
        return _TappedRead(read, self._sink, self._on_error), _TappedWrite(write, self._sink)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self._inner.__aexit__(exc_type, exc, tb)
