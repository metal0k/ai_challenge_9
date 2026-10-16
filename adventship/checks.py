"""Stage `checks`: ruff, format check, pytest, personal-data scan of the staged index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from advent_core.config import SECRET_KEYS, redact
from adventship.ctx import EXIT_CHECKS_FAIL, ShipContext, ShipError, StageResult

CHECKS: list[tuple[str, list[str], float]] = [
    ("ruff check", ["uv", "run", "ruff", "check", "."], 300),
    ("ruff format --check", ["uv", "run", "ruff", "format", "--check", "."], 300),
    ("pytest -q", ["uv", "run", "pytest", "-q"], 1800),
    ("check_staged", [sys.executable, "tools/check_staged.py"], 120),
]
_TAIL_LINES = 15
# Flags the tests set themselves; inheriting them from the shipping shell would skew the run.
_TEST_CONTROLLED = ("ADVENT_OFFLINE", "ADVENT_BASE_URL")


def sanitized_env(root: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Parent env minus every key of `root/.env`, SECRET_KEYS and test-controlled flags.

    `adventship run` loads .env into os.environ; a child pytest that inherits the real
    secrets fails tests that expect a clean shell. Only key names are read, never printed.
    """
    source = os.environ if base is None else base
    env_file = root / ".env"
    dropped = set(SECRET_KEYS) | set(_TEST_CONTROLLED)
    if env_file.is_file():
        dropped |= set(dotenv_values(env_file))
    upper = {key.upper() for key in dropped}
    return {k: v for k, v in source.items() if k.upper() not in upper}


def _tail(*streams: str) -> str:
    lines = "\n".join(s for s in streams if s).strip().splitlines()
    return redact("\n".join(lines[-_TAIL_LINES:]))


def run_checks(ctx: ShipContext) -> StageResult:
    names: list[str] = []
    env = sanitized_env(ctx.root)
    for name, argv, timeout in CHECKS:
        result = ctx.run(argv, timeout=timeout, env=env, inherit_env=False)
        if result.returncode != 0:
            raise ShipError(
                f"проверка не прошла: {name}",
                exit_code=EXIT_CHECKS_FAIL,
                hint=_tail(result.stdout or "", result.stderr or ""),
            )
        names.append(name)
    return StageResult(data={"checks": names}, summary=f"прошли: {', '.join(names)}")
