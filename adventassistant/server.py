"""Read-only git MCP server for the day-31 assistant (stdio).

stdout is the protocol and nothing else. The repo root is this file's
grandparent, not the client's cwd. Run: `python -m adventassistant.server`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_NAME = "advent-assistant"
TOOL_NAMES = ("git_branch", "git_diff_stat")
GIT_TIMEOUT = 10.0
DIFF_LINE_CAP = 80

REPO_ROOT = Path(__file__).resolve().parent.parent

mcp = MCPServer(
    SERVER_NAME,
    instructions="Read-only git branch and diff for the AI Advent repository.",
    version="0.1.0",
    log_level="WARNING",
)


def _git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"git не выполнился: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"код {proc.returncode}"
        raise ToolError(f"git завершился с ошибкой: {detail}")
    return proc.stdout


def format_branch(name: str, sha12: str | None = None) -> str:
    if name == "HEAD":
        return f"ветка: detached {sha12}"
    return f"ветка: {name}"


def clip_diff(text: str) -> str:
    """One half of a diff stat: `нет` when empty, at most 80 lines otherwise."""
    if not text.strip():
        return "нет"
    lines = text.splitlines()
    if len(lines) <= DIFF_LINE_CAP:
        return "\n".join(lines)
    keep = DIFF_LINE_CAP - 1
    omitted = len(lines) - keep
    return "\n".join([*lines[:keep], f"… ещё {omitted} строк"])


def format_diff(work: str, staged: str) -> str:
    if not work.strip() and not staged.strip():
        return "изменений нет"
    return f"незакоммичено:\n{clip_diff(work)}\nв индексе:\n{clip_diff(staged)}"


@mcp.tool(description="Текущая ветка репозитория: `ветка: <имя>` или detached sha.")
def git_branch() -> str:
    name = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if name == "HEAD":
        sha = _git(["rev-parse", "--short=12", "HEAD"]).strip()
        return format_branch(name, sha)
    return format_branch(name)


@mcp.tool(description="Незакоммиченные и проиндексированные изменения (`git diff --stat`).")
def git_diff_stat() -> str:
    work = _git(["diff", "--stat"])
    staged = _git(["diff", "--cached", "--stat"])
    return format_diff(work, staged)


if __name__ == "__main__":
    mcp.run()
