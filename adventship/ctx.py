"""Shared contract: errors, stage result, per-run context, default subprocess runner."""

from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from advent_core.errors import AdventError

if TYPE_CHECKING:
    from adventship.sheet_driver import SheetDriver
    from adventship.state import ShipState

EXIT_CONFIG = 2
EXIT_AUTH = 3
EXIT_TRUNCATED = 7
EXIT_NO_TTY = 8
EXIT_GATE_DENIED = 9
EXIT_STAGE_FAILED = 10
EXIT_REVIEW_FAIL = 11
EXIT_CHECKS_FAIL = 12


class ShipError(AdventError):
    """A pipeline failure with an exit code from the spec table."""

    exit_code = EXIT_STAGE_FAILED

    def __init__(
        self, message: str, *, exit_code: int = EXIT_STAGE_FAILED, hint: str | None = None
    ) -> None:
        super().__init__(message, hint=hint)
        self.exit_code = exit_code


@dataclass
class StageResult:
    """`data` holds public values only: it is persisted and feeds downstream input hashes."""

    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class TaskSpec:
    week: int
    day: int
    source: str
    text: str

    @property
    def title(self) -> str:
        for line in self.text.splitlines():
            if line.strip():
                return line.strip()
        return ""

    @property
    def task_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


RunFn = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass
class ShipContext:
    week: int
    day: int
    tag: str
    root: Path
    workdir: Path
    env: dict[str, str]
    state: ShipState
    run: RunFn
    complete: Callable[..., Any]
    gate: Callable[[str, str], bool]
    driver_factory: Callable[[], SheetDriver]
    now: Callable[[], str]
    task: TaskSpec | None = None
    task_file: Path | None = None
    dry_run: bool = False
    no_record: bool = False
    journal_path: Path | None = None


def kill_tree(pid: int) -> None:
    """Kill a process and its children; `taskkill /T` on Windows, process group elsewhere."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(OSError):
            os.killpg(pid, 9)


def run_command(
    argv: list[str],
    *,
    timeout: float,
    input: str | None = None,
    env: dict[str, str] | None = None,
    inherit_env: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """argv list, utf-8, stdin closed unless `input` is given, whole tree killed on timeout.

    `inherit_env=False` passes `env` as the complete environment (no os.environ merge).
    """
    full_env = {**(os.environ if inherit_env else {}), **(env or {})}
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=full_env,
        cwd=cwd,
        **popen_kwargs,
    )
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process.pid)
        process.communicate()
        raise ShipError(
            f"команда превысила таймаут {timeout:g} с: {argv[0]}",
            hint="Процесс и его потомки остановлены.",
        ) from None
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
