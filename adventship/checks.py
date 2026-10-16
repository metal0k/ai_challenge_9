"""Stage `checks`: ruff, format check, pytest, personal-data scan of the staged index."""

from __future__ import annotations

import sys

from advent_core.config import redact
from adventship.ctx import EXIT_CHECKS_FAIL, ShipContext, ShipError, StageResult

CHECKS: list[tuple[str, list[str], float]] = [
    ("ruff check", ["uv", "run", "ruff", "check", "."], 300),
    ("ruff format --check", ["uv", "run", "ruff", "format", "--check", "."], 300),
    ("pytest -q", ["uv", "run", "pytest", "-q"], 1800),
    ("check_staged", [sys.executable, "tools/check_staged.py"], 120),
]
_TAIL_LINES = 15


def _tail(*streams: str) -> str:
    lines = "\n".join(s for s in streams if s).strip().splitlines()
    return redact("\n".join(lines[-_TAIL_LINES:]))


def run_checks(ctx: ShipContext) -> StageResult:
    names: list[str] = []
    for name, argv, timeout in CHECKS:
        result = ctx.run(argv, timeout=timeout)
        if result.returncode != 0:
            raise ShipError(
                f"проверка не прошла: {name}",
                exit_code=EXIT_CHECKS_FAIL,
                hint=_tail(result.stdout or "", result.stderr or ""),
            )
        names.append(name)
    return StageResult(data={"checks": names}, summary=f"прошли: {', '.join(names)}")
