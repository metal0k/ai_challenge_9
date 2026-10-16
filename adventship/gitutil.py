"""Small git helpers shared by `gitops` and `llm`; every call goes through `ctx.run`."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

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


def _status_paths(ctx: ShipContext, untracked: str) -> list[str]:
    out = git(ctx, "status", "--porcelain", "-z", f"--untracked-files={untracked}").stdout
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


def changed_paths(ctx: ShipContext) -> list[str]:
    """Paths from `git status --porcelain -z` (untracked files expanded), sorted, unique."""
    return _status_paths(ctx, "all")


def dirty_tracked_paths(ctx: ShipContext) -> list[str]:
    """Tracked paths with uncommitted changes; untracked files are never listed."""
    return _status_paths(ctx, "no")


_DAY_TAG = re.compile(r"^w(\d{2})d(\d{2})$")


@dataclass
class DayChanges:
    base: str
    range: str
    commits: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    stat: str = ""
    dirty_tracked: list[str] = field(default_factory=list)


def day_range(ctx: ShipContext, tag: str | None = None) -> tuple[str, str]:
    """`(base_ref, "base..HEAD")`: base is the highest local day tag below `tag`, else the root."""
    tag = tag or ctx.tag
    current = _DAY_TAG.match(tag)
    best: tuple[tuple[int, int], str] | None = None
    if current:
        limit = (int(current[1]), int(current[2]))
        for name in git(ctx, "tag", "--list").stdout.split():
            match = _DAY_TAG.match(name)
            if not match:
                continue
            key = (int(match[1]), int(match[2]))
            if key < limit and (best is None or key > best[0]):
                best = (key, name)
    if best:
        base = best[1]
    else:
        roots = git(ctx, "rev-list", "--max-parents=0", "HEAD").stdout.split()
        if not roots:
            raise ShipError("в репозитории нет коммитов")
        base = roots[-1]
    return base, f"{base}..HEAD"


def day_changes(ctx: ShipContext, tag: str | None = None) -> DayChanges:
    """The day's change set: commits in `base..HEAD`. Untracked files are never part of it."""
    base, rng = day_range(ctx, tag)
    return DayChanges(
        base=base,
        range=rng,
        commits=git(ctx, "log", "--format=%s", rng).stdout.splitlines(),
        files=git(ctx, "diff", "--name-only", rng).stdout.splitlines(),
        stat=git(ctx, "diff", "--stat", rng).stdout.strip(),
        dirty_tracked=dirty_tracked_paths(ctx),
    )
