"""Small git helpers shared by `gitops` and `llm`; every call goes through `ctx.run`."""

from __future__ import annotations

import subprocess

from adventship.ctx import ShipContext, ShipError

GIT_TIMEOUT = 60.0


def git(
    ctx: ShipContext,
    *args: str,
    timeout: float = GIT_TIMEOUT,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `git <args>` in the repo; a non-zero exit raises unless `check=False`."""
    try:
        result = ctx.run(["git", *args], timeout=timeout, env=env)
    except OSError as error:
        raise ShipError(f"git не запустился: {error}") from error
    if check and result.returncode != 0:
        raise ShipError(
            f"git {args[0]} завершился с кодом {result.returncode}",
            hint=(result.stderr or result.stdout or "").strip()[-400:],
        )
    return result


def changed_paths(ctx: ShipContext) -> list[str]:
    """Paths from `git status --porcelain -z` (untracked files expanded), sorted, unique."""
    out = git(ctx, "status", "--porcelain", "-z", "--untracked-files=all").stdout
    tokens = out.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        entry = tokens[index]
        index += 1
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if entry[0] in "RC":  # -z lists the rename source as a separate token
            index += 1
    return sorted(set(paths))
