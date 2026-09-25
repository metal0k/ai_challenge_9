"""Multi-server MCP router (week 04, day 20).

Reads a registry of stdio servers, merges their tools into one `tools=` list under
`server__tool` names and dispatches each model call to the owning server. Spawning
stays per-call (`mcp_client.call_tool_once`), so nothing here holds a connection.

`Router.call` matches `Agent.call_tool`'s contract: it never raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from advent_core import mcp_client
from advent_core.errors import MCPError
from advent_core.mcp_client import ToolCallOutcome, ToolInfo

SEPARATOR = "__"
# Mistral follows the OpenAI shape for function names; longer/other names are 400s.
NAME_MAX = 64
NAME_BAD = re.compile(r"[^A-Za-z0-9_-]")
DEFAULT_TIMEOUT = 15.0
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "week_04" / "servers.json"


@dataclass(slots=True, frozen=True)
class ServerSpec:
    name: str
    command: list[str]
    cwd: Path | None = None
    allow: tuple[str, ...] | None = None  # None: every tool the server offers
    timeout: float = DEFAULT_TIMEOUT
    ensure_dirs: tuple[Path, ...] = ()


@dataclass(slots=True, frozen=True)
class Route:
    exposed: str
    server: str
    tool: str


@dataclass(slots=True, frozen=True)
class RouteRecord:
    """One dispatched call, for the trace line, the end-of-turn table and the journal."""

    round: int
    server: str
    tool: str
    exposed: str
    is_error: bool
    result_bytes: int
    seconds: float
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConnectReport:
    tools: list[dict[str, Any]] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    server_tools: dict[str, list[str]] = field(default_factory=dict)  # after `allow`
    server_offered: dict[str, list[str]] = field(default_factory=dict)  # before `allow`
    warnings: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _expand(text: str, repo: Path) -> str:
    return text.replace("{repo}", repo.as_posix()).replace("{python}", sys.executable)


def _resolve_command(command: str) -> str:
    # npx is a .cmd shim on Windows; CreateProcess does not resolve it from a bare name.
    if os.name == "nt" and command == "npx":
        return "npx.cmd"
    return command


def _string_list(entry: dict, key: str) -> list[str] | None:
    """A registry list field: absent is None, anything but a list of strings is an error."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{key} у сервера {entry.get('name')!r} должен быть списком строк")
    return value


def load_registry(path: Path | None = None, *, repo_root: Path | None = None) -> list[ServerSpec]:
    """Parse the registry; a malformed one is an MCPError with a hint, not a traceback."""
    path = path or DEFAULT_REGISTRY
    repo = repo_root or REPO_ROOT
    specs: list[ServerSpec] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for entry in raw["servers"]:
            name = str(entry["name"])
            if not name or SEPARATOR in name or NAME_BAD.search(name):
                raise ValueError(f"недопустимое имя сервера {name!r}")
            if name in seen:
                raise ValueError(f"имя сервера {name!r} повторяется")
            seen.add(name)
            allow = _string_list(entry, "allow")
            args = _string_list(entry, "args") or []
            ensure_dirs = _string_list(entry, "ensure_dirs") or []
            cwd = entry.get("cwd")
            specs.append(
                ServerSpec(
                    name=name,
                    command=[
                        _resolve_command(_expand(str(entry["command"]), repo)),
                        *(_expand(str(a), repo) for a in args),
                    ],
                    cwd=Path(_expand(cwd, repo)) if cwd else None,
                    allow=tuple(allow) if allow is not None else None,
                    timeout=float(entry.get("timeout", DEFAULT_TIMEOUT)),
                    ensure_dirs=tuple(Path(_expand(d, repo)) for d in ensure_dirs),
                )
            )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise MCPError(
            f"реестр серверов {path.name} не прочитан: {error}",
            hint='Ожидается {"servers": [{"name", "command", "args", ...}]}.',
        ) from None
    if not specs:
        raise MCPError(f"реестр серверов {path.name} пуст", hint="Добавьте хотя бы один сервер.")
    return specs


def exposed_name(server: str, tool: str, taken: set[str]) -> str:
    """`server__tool`, kept inside NAME_MAX and the allowed alphabet, unique in `taken`.

    Deterministic: a shortened or de-collided name carries a hash of the FULL
    original pair, so the same registry always yields the same names.
    """
    full = f"{server}{SEPARATOR}{tool}"
    name = NAME_BAD.sub("_", full)
    if len(name) > NAME_MAX or name in taken:
        digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:6]
        name = f"{name[: NAME_MAX - 7]}_{digest}"
    return name


def _function_tool(name: str, tool: ToolInfo, server: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"[{server}] {tool.description}".strip(),
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


class Router:
    def __init__(
        self,
        specs: list[ServerSpec],
        *,
        trace: Callable[[str], None] | None = None,
        journal: Callable[[RouteRecord], None] | None = None,
        current_round: Callable[[], int] | None = None,
    ) -> None:
        self.specs = {spec.name: spec for spec in specs}
        self._trace = trace
        self._journal = journal
        self._current_round = current_round
        self._routes: dict[str, Route] = {}
        self._records: list[RouteRecord] = []
        self._calls = 0
        self.failed: list[str] = []  # servers unreachable at the last connect()

    @property
    def routes(self) -> list[Route]:
        return list(self._routes.values())

    @property
    def valid_names(self) -> list[str]:
        return list(self._routes)

    # --- connect ------------------------------------------------------------

    def _list_one(self, spec: ServerSpec) -> tuple[ServerSpec, Any]:
        try:
            for directory in spec.ensure_dirs:
                directory.mkdir(parents=True, exist_ok=True)
            listing = mcp_client.connect_and_list(
                spec.command,
                timeout=spec.timeout,
                raw=False,
                sink=lambda _line: None,
                cwd=spec.cwd,
                quiet=True,
            )
            return spec, listing
        except MCPError as error:
            return spec, error
        except Exception as error:  # one broken server must not take the others down
            return spec, MCPError(str(error) or repr(error))

    def connect(self) -> ConnectReport:
        """tools/list from every server; unreachable ones become warnings, not failures."""
        report = ConnectReport()
        self._routes = {}
        self.failed = []
        taken: set[str] = set()
        with ThreadPoolExecutor(max_workers=len(self.specs)) as pool:
            outcomes = list(pool.map(self._list_one, self.specs.values()))
        for spec, outcome in outcomes:
            if isinstance(outcome, MCPError):
                report.failed.append(spec.name)
                self.failed.append(spec.name)
                report.warnings.append(f"сервер {spec.name} недоступен: {outcome.message}")
                continue
            offered = {tool.name: tool for tool in outcome.tools}
            report.server_offered[spec.name] = list(offered)
            if spec.allow is not None:
                for missing in (n for n in spec.allow if n not in offered):
                    report.warnings.append(
                        f"сервер {spec.name}: в allow есть {missing}, но сервер такого не отдал"
                    )
                chosen = [offered[n] for n in spec.allow if n in offered]
            else:
                chosen = list(offered.values())
            report.server_tools[spec.name] = [tool.name for tool in chosen]
            for tool in chosen:
                name = exposed_name(spec.name, tool.name, taken)
                taken.add(name)
                route = Route(name, spec.name, tool.name)
                self._routes[name] = route
                report.routes.append(route)
                report.tools.append(_function_tool(name, tool, spec.name))
        return report

    # --- dispatch -----------------------------------------------------------

    def _unknown(self, name: str) -> ToolCallOutcome:
        valid = ", ".join(self._routes) or "ни одного"
        return ToolCallOutcome(
            text=f"неизвестный инструмент {name!r}; доступные: {valid}", is_error=True
        )

    def call(self, name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
        """Never raises (Agent.call_tool contract); failures come back as error text."""
        route = self._routes.get(name)
        if route is None:
            outcome = self._unknown(name)
            self._record(name, "?", name, outcome, 0.0, arguments)
            return outcome
        spec = self.specs[route.server]
        started = time.perf_counter()
        try:
            outcome = mcp_client.call_tool_once(
                spec.command,
                route.tool,
                arguments,
                timeout=spec.timeout,
                cwd=spec.cwd,
                quiet=True,
            )
        except MCPError as error:
            outcome = ToolCallOutcome(text=error.message, is_error=True)
        except Exception as error:
            outcome = ToolCallOutcome(text=str(error) or repr(error), is_error=True)
        elapsed = time.perf_counter() - started
        self._record(route.exposed, route.server, route.tool, outcome, elapsed, arguments)
        return outcome

    def _record(
        self,
        exposed: str,
        server: str,
        tool: str,
        outcome: ToolCallOutcome,
        seconds: float,
        arguments: dict[str, Any],
    ) -> None:
        self._calls += 1
        try:
            round_no = self._current_round() if self._current_round else self._calls
        except Exception:
            round_no = self._calls
        record = RouteRecord(
            round=round_no,
            server=server,
            tool=tool,
            exposed=exposed,
            is_error=outcome.is_error,
            result_bytes=len(outcome.text.encode("utf-8")),
            seconds=seconds,
            arguments=dict(arguments),
        )
        self._records.append(record)
        # Trace and journal are side effects: neither may break the never-raises contract.
        try:
            if self._trace:
                mark = " ОШИБКА" if record.is_error else ""
                self._trace(
                    f"раунд {record.round} · {server} → {tool} "
                    f"({record.result_bytes} Б, {seconds:.1f} с){mark}"
                )
            if self._journal:
                self._journal(record)
        except Exception:
            pass

    def take_records(self) -> list[RouteRecord]:
        """Calls made since the last take (one turn's route). Hands them over once."""
        records, self._records = self._records, []
        return records
