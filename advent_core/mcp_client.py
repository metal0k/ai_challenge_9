"""Minimal MCP client: spawn a stdio server, handshake, list/call its tools.

Every failure is translated into `MCPError` (exit code 8) with a distinct
cause; the child process is always reaped because the whole session lives in
one `async with`.
"""

from __future__ import annotations

import logging
import os
import shlex
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import anyio
from mcp import Client
from mcp import MCPError as SdkMCPError
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from advent_core.errors import MCPError
from advent_core.mcp_wire import TapTransport

MAX_PAGES = 100
CONNECTION_CLOSED_CODE = -32000  # SDK: the transport closed under a pending request

STAGE_INITIALIZE = "handshake"
STAGE_LIST = "tools/list"
STAGE_CALL = "tools/call"


@dataclass(slots=True, frozen=True)
class ToolArg:
    name: str
    type: str
    required: bool
    enum: tuple[str, ...] = ()
    default: str | None = None


@dataclass(slots=True, frozen=True)
class ToolInfo:
    name: str
    description: str
    args: tuple[ToolArg, ...] = ()
    # The RAW JSON Schema, as the server sent it. `args` is the summarized,
    # printable form for `adventmcp tools`; this one goes straight into a
    # Mistral FunctionTool's `parameters`, where the summary would not fit.
    # Trailing and defaulted so day-16 positional constructions still work.
    input_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class ListResult:
    server_name: str
    server_version: str
    protocol_version: str
    instructions: str | None
    tools: list[ToolInfo] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ToolCallOutcome:
    text: str
    is_error: bool


def split_command(text: str, *, windows: bool | None = None) -> list[str]:
    """Split a command line; Windows paths keep their backslashes."""
    windows = os.name == "nt" if windows is None else windows
    try:
        if windows:
            # posix=True would eat every backslash of C:\Program Files\...
            parts = [
                part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'" else part
                for part in shlex.split(text, posix=False)
            ]
        else:
            parts = shlex.split(text)
    except ValueError as exc:
        raise MCPError(
            f"не удалось разобрать команду сервера: {exc}",
            hint="Проверьте кавычки в значении --server.",
        ) from None
    return parts


def _type_of(schema: dict[str, Any]) -> str:
    kind = schema.get("type")
    if isinstance(kind, list):
        return "|".join(str(k) for k in kind)
    if kind:
        return str(kind)
    options = schema.get("anyOf") or schema.get("oneOf") or []
    names = [_type_of(o) for o in options if isinstance(o, dict)]
    return "|".join(dict.fromkeys(names)) or "any"


def summarize_args(input_schema: dict[str, Any] | None) -> tuple[ToolArg, ...]:
    schema = input_schema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    args = []
    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        args.append(
            ToolArg(
                name=name,
                type=_type_of(prop),
                required=name in required,
                enum=tuple(str(v) for v in prop.get("enum") or ()),
                default=str(prop["default"]) if "default" in prop else None,
            )
        )
    return tuple(args)


STDERR_TAIL_CHARS = 300


def _stdio(params: StdioServerParameters, quiet: bool) -> Any:
    """`stdio_client`, with the child's stderr captured to a temp file when `quiet`.

    Off by default: days 16-19 keep the SDK's own behaviour (child stderr on screen).
    Captured rather than discarded so a failing server's last words can join the error.
    """
    if not quiet:
        return stdio_client(params)
    errlog: TextIO = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")  # noqa: SIM115
    try:
        return _ClosingStdio(stdio_client(params, errlog=errlog), errlog)
    except BaseException:
        errlog.close()
        raise


class _ClosingStdio:
    """Async-CM wrapper that keeps the stderr tail and closes the capture file on exit."""

    def __init__(self, inner: Any, errlog: TextIO) -> None:
        self._inner = inner
        self._errlog = errlog
        self.tail = ""

    async def __aenter__(self) -> Any:
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            self.close()

    def close(self) -> None:
        """Idempotent; also the cleanup for a transport that never got entered."""
        if self._errlog.closed:
            return
        try:
            self._errlog.flush()
            self._errlog.seek(0)
            text = self._errlog.read()
            self.tail = " ".join(text.split())[-STDERR_TAIL_CHARS:]
        except (OSError, ValueError):
            pass
        finally:
            self._errlog.close()


def _with_stderr(error: MCPError, stdio: Any) -> MCPError:
    """Append the child's stderr tail (quiet mode only) to the error text."""
    if isinstance(stdio, _ClosingStdio):
        stdio.close()
        if stdio.tail:
            return MCPError(f"{error.message} (stderr сервера: …{stdio.tail})", hint=error.hint)
    return error


def _flatten(exc: BaseException) -> Iterator[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _flatten(sub)
    else:
        yield exc


def _translate(
    exc: BaseException,
    stage: str,
    timeout: float,
    command: list[str],
    *,
    non_frames: list[Exception] | None = None,
) -> MCPError:
    """Map an SDK/OS failure to one of the distinct MCPError causes."""
    leaves = list(_flatten(exc))
    # Our own errors raised inside the SDK's task group arrive wrapped in a group.
    for leaf in leaves:
        if isinstance(leaf, MCPError):
            return leaf
    if non_frames:
        return MCPError(
            "сервер пишет в stdout не JSON-RPC",
            hint="В stdout MCP-сервера по stdio допустим только протокол: "
            "логи и print() нужно слать в stderr.",
        )
    # TimeoutError is an OSError subclass: it has to be tested first.
    if any(isinstance(leaf, TimeoutError) for leaf in leaves):
        return MCPError(
            f"сервер не ответил за {timeout:g} с на этапе {stage}",
            hint="Убедитесь, что команда запускает MCP-сервер по stdio; "
            "увеличьте --timeout, если сервер стартует долго.",
        )
    for leaf in leaves:
        if isinstance(leaf, SdkMCPError) and leaf.code == CONNECTION_CLOSED_CODE:
            return MCPError(
                f"процесс сервера завершился до конца handshake (этап {stage})",
                hint="Запустите команду вручную и посмотрите её stderr.",
            )
    for leaf in leaves:
        if isinstance(leaf, FileNotFoundError):
            return MCPError(
                f"исполняемый файл не найден: {command[0]}",
                hint="Проверьте значение --server и что команда есть в PATH.",
            )
        if isinstance(leaf, OSError | ValueError):
            return MCPError(
                f"не удалось запустить сервер {command[0]}: {leaf}",
                hint="Проверьте путь, права доступа и значение --server.",
            )
    for leaf in leaves:
        if isinstance(leaf, SdkMCPError):
            return MCPError(
                f"протокольная ошибка MCP на этапе {stage}: {leaf.message} (код {leaf.code})",
                hint="Сервер отвечает не по спецификации MCP.",
            )
    leaf = leaves[0]
    return MCPError(
        f"протокольная ошибка на этапе {stage}: {type(leaf).__name__}: {leaf}",
        hint="Сервер отвечает не по спецификации MCP; его stdout должен нести только JSON-RPC.",
    )


async def aconnect_and_list(
    server_cmd: list[str],
    *,
    timeout: float,
    raw: bool,
    sink: Callable[[str], None],
    cwd: Path | None = None,
    quiet: bool = False,
) -> ListResult:
    if not server_cmd:
        raise MCPError("команда сервера пустая", hint='Передайте --server "<команда>".')
    params = StdioServerParameters(
        command=server_cmd[0], args=server_cmd[1:], cwd=cwd, encoding="utf-8"
    )
    # The tap is always on so non-JSON output is detected; frames are mirrored
    # to the sink only with --raw.
    non_frames: list[Exception] = []
    stdio = _stdio(params, quiet)
    transport: Any = TapTransport(stdio, sink if raw else (lambda line: None), non_frames.append)
    stage = STAGE_INITIALIZE
    try:
        # One scope for the whole session: its deadline is re-armed per stage,
        # and the SDK's own task group stays properly nested inside it.
        with anyio.fail_after(timeout) as scope:
            async with Client(transport) as client:
                scope.deadline = anyio.current_time() + timeout
                stage = STAGE_LIST
                info = client.server_info
                result = ListResult(
                    server_name=info.name,
                    server_version=info.version or "",
                    protocol_version=str(client.protocol_version),
                    instructions=client.instructions,
                )
                cursor: str | None = None
                seen: set[str] = set()
                for _ in range(MAX_PAGES):
                    page = await client.list_tools(cursor=cursor)
                    for tool in page.tools:
                        result.tools.append(
                            ToolInfo(
                                name=tool.name,
                                description=tool.description or "",
                                args=summarize_args(tool.input_schema),
                                input_schema=tool.input_schema,
                            )
                        )
                    cursor = page.next_cursor
                    if not cursor:
                        return result
                    if cursor in seen:
                        raise MCPError(
                            f"сервер зациклил пагинацию tools/list (курсор {cursor!r} повторился)",
                            hint="Ошибка в сервере: next_cursor должен меняться.",
                        )
                    seen.add(cursor)
                    scope.deadline = anyio.current_time() + timeout
                raise MCPError(
                    f"tools/list не закончился за {MAX_PAGES} страниц",
                    hint="Ошибка в сервере: список не должен быть бесконечным.",
                )
    except MCPError as error:
        raise _with_stderr(error, stdio) from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if not isinstance(exc, Exception) and not isinstance(exc, BaseExceptionGroup):
            raise  # cancellation and other BaseExceptions are not ours to translate
        raise _with_stderr(
            _translate(exc, stage, timeout, server_cmd, non_frames=non_frames), stdio
        ) from None


_LOG_LOCK = threading.Lock()
_LOG_HOLDERS = 0
_LOG_SAVED = logging.NOTSET


class _quiet_sdk_logger:  # noqa: N801 - context manager used like a function
    """Silence `mcp.client.stdio` for the duration; refcounted so threads cannot leak CRITICAL.

    A per-call save/restore races when calls overlap (Router.connect runs a pool):
    the second saver would read CRITICAL as the "previous" level.
    """

    def __enter__(self) -> None:
        global _LOG_HOLDERS, _LOG_SAVED
        logger = logging.getLogger("mcp.client.stdio")
        with _LOG_LOCK:
            if _LOG_HOLDERS == 0:
                _LOG_SAVED = logger.level
                logger.setLevel(logging.CRITICAL)
            _LOG_HOLDERS += 1

    def __exit__(self, *exc: Any) -> None:
        global _LOG_HOLDERS
        with _LOG_LOCK:
            _LOG_HOLDERS -= 1
            if _LOG_HOLDERS == 0:
                logging.getLogger("mcp.client.stdio").setLevel(_LOG_SAVED)


def connect_and_list(
    server_cmd: list[str],
    *,
    timeout: float,
    raw: bool,
    sink: Callable[[str], None],
    cwd: Path | None = None,
    quiet: bool = False,
) -> ListResult:
    # The SDK logs a rich traceback for every unparsable stdout line; we report
    # the cause ourselves (as advent_cli/obs.py does for obsws_python).
    with _quiet_sdk_logger():
        return anyio.run(
            lambda: aconnect_and_list(
                server_cmd, timeout=timeout, raw=raw, sink=sink, cwd=cwd, quiet=quiet
            )
        )


def _outcome_of(result: Any) -> ToolCallOutcome:
    # Non-text blocks are skipped: this project's server only ever returns text.
    text = "".join(block.text for block in result.content if isinstance(block, TextContent))
    return ToolCallOutcome(text=text, is_error=bool(result.is_error))


async def acall_tool(
    server_cmd: list[str],
    name: str,
    arguments: dict[str, Any],
    *,
    timeout: float,
    cwd: Path | None = None,
    quiet: bool = False,
) -> ToolCallOutcome:
    """One short-lived subprocess: spawn, handshake, call, close. No long-lived connection."""
    if not server_cmd:
        raise MCPError("команда сервера пустая", hint='Передайте --server "<команда>".')
    params = StdioServerParameters(
        command=server_cmd[0], args=server_cmd[1:], cwd=cwd, encoding="utf-8"
    )
    non_frames: list[Exception] = []
    stdio = _stdio(params, quiet)
    transport: Any = TapTransport(stdio, lambda line: None, non_frames.append)
    stage = STAGE_INITIALIZE
    try:
        with anyio.fail_after(timeout) as scope:
            async with Client(transport) as client:
                scope.deadline = anyio.current_time() + timeout
                stage = STAGE_CALL
                result = await client.call_tool(name, arguments)
                return _outcome_of(result)
    except MCPError as error:
        raise _with_stderr(error, stdio) from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if not isinstance(exc, Exception) and not isinstance(exc, BaseExceptionGroup):
            raise  # cancellation and other BaseExceptions are not ours to translate
        raise _with_stderr(
            _translate(exc, stage, timeout, server_cmd, non_frames=non_frames), stdio
        ) from None


def call_tool_once(
    server_cmd: list[str],
    name: str,
    arguments: dict[str, Any],
    *,
    timeout: float,
    cwd: Path | None = None,
    quiet: bool = False,
) -> ToolCallOutcome:
    with _quiet_sdk_logger():
        return anyio.run(
            lambda: acall_tool(server_cmd, name, arguments, timeout=timeout, cwd=cwd, quiet=quiet)
        )
