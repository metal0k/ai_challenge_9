"""Minimal read-only MCP server about this repository (stdio).

stdout carries the protocol and nothing else: a stray print() breaks framing,
so diagnostics go to stderr only. Run: `python -m week_04.server`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from advent_core import tokens
from week_04 import scheduler

SERVER_NAME = "advent-repo"
# The client's default `--expect` list; tests assert the literal names.
TOOL_NAMES = (
    "list_days",
    "get_task",
    "count_tokens",
    "git_log",
    "schedule_job",
    "repo_activity_summary",
)
SCHEDULE_MIN = scheduler.MIN_INTERVAL_SECONDS
SCHEDULE_MAX = scheduler.MAX_INTERVAL_SECONDS

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"^w\d{2}d\d{2}$")
DEFAULT_MODEL = "ministral-14b-latest"
GIT_TIMEOUT = 10.0
GIT_LOG_MAX = 20

mcp = MCPServer(
    SERVER_NAME,
    instructions="Read-only tools about the AI Advent coursework repository.",
    version="0.1.0",
    log_level="WARNING",
)


def submitted_days(root: Path = REPO_ROOT) -> list[str]:
    """Tags `wNNdDD`, oldest first; empty when git is absent or not a repo."""
    try:
        proc = subprocess.run(
            ["git", "tag", "--list", "--sort=version:refname"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [tag for tag in proc.stdout.split() if TAG_RE.match(tag)]


def task_file(week: int, root: Path = REPO_ROOT) -> Path | None:
    """`week_0W/tasks_0W.md` or, for week 4, the singular `task_04.md`."""
    folder = root / f"week_{week:02d}"
    for name in (f"tasks_{week:02d}.md", f"task_{week:02d}.md"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def extract_day(text: str, day: int) -> str | None:
    """Body of the `## Day N` section, up to the next `## ` heading."""
    lines = text.splitlines()
    head = re.compile(rf"^##\s+Day\s+0*{day}\s*$", re.IGNORECASE)
    start = next((i for i, line in enumerate(lines) if head.match(line)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).strip()


@mcp.tool(description="Список сданных дней: git-теги вида w01d01, по одному в строке.")
def list_days() -> str:
    days = submitted_days()
    return "\n".join(days) if days else "тегов wNNdDD не найдено"


@mcp.tool(description="Текст задания за день: раздел `## Day N` из файла задач недели.")
def get_task(week: int, day: int) -> str:
    path = task_file(week)
    if path is None:
        raise ToolError(f"нет файла задач для недели {week}")
    section = extract_day(path.read_text(encoding="utf-8"), day)
    if section is None:
        raise ToolError(f"в {path.name} нет раздела Day {day}")
    return section


@mcp.tool(description="Число токенов в тексте по токенизатору модели (или оценка).")
def count_tokens(
    text: str,
    model: Literal[
        "ministral-3b-latest", "ministral-8b-latest", "ministral-14b-latest"
    ] = DEFAULT_MODEL,
) -> str:
    # download=False: the server must never touch the network.
    counter, _warning = tokens.counter_for(model, download=False)
    full = counter.count([{"role": "user", "content": text}])
    if full is None:
        raise ToolError("не удалось посчитать токены")
    # Exact counter: subtract the chat-template overhead so only the text's own
    # cost is left. An estimate has no overhead model, so it is left as is.
    overhead = (counter.count([{"role": "user", "content": ""}]) or 0) if counter.exact else 0
    total = full - overhead
    kind = "точно" if counter.exact else "оценка"
    return f"{total} токенов ({kind}, {model})"


@mcp.tool(description="Последние N коммитов этого репозитория: hash, дата, автор, тема.")
def git_log(n: int = 5) -> str:
    if not 1 <= n <= GIT_LOG_MAX:
        raise ToolError(f"n должен быть от 1 до {GIT_LOG_MAX}, получено {n}")
    try:
        proc = subprocess.run(
            ["git", "log", f"-{n}", "--pretty=format:%h  %ad  %an  %s", "--date=short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"git log не выполнился: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"git log завершился с ошибкой: {proc.stderr.strip()}")
    return proc.stdout


@mcp.tool(
    description="Запланировать периодический сбор активности репозитория (git log) "
    "с заданным интервалом в секундах; повторный вызов меняет интервал "
    "уже существующего job'а."
)
def schedule_job(interval_seconds: int) -> str:
    if not SCHEDULE_MIN <= interval_seconds <= SCHEDULE_MAX:
        raise ToolError(
            f"interval_seconds должен быть от {SCHEDULE_MIN} до {SCHEDULE_MAX}, "
            f"получено {interval_seconds}"
        )
    try:
        previous = scheduler.load_state().job
        scheduler.upsert_job(interval_seconds)
    except OSError as exc:
        raise ToolError(f"не удалось обновить состояние планировщика: {exc}") from exc
    if previous is None:
        return (
            f"job repo_activity создан: интервал {interval_seconds} с. "
            "Демон подхватит его на следующем опросе."
        )
    if previous.interval_seconds == interval_seconds:
        return f"job repo_activity: интервал уже {interval_seconds} с, без изменений."
    return (
        f"job repo_activity: интервал изменён {previous.interval_seconds} → {interval_seconds} с."
    )


@mcp.tool(
    description="Агрегированная сводка по периодически собираемой активности "
    "репозитория: сколько новых коммитов и за сколько тиков "
    "(опционально — только за последние N минут)."
)
def repo_activity_summary(minutes: int | None = None) -> str:
    if minutes is not None and minutes <= 0:
        raise ToolError(f"minutes должен быть положительным, получено {minutes}")
    try:
        state = scheduler.load_state()
    except OSError as exc:
        raise ToolError(f"не удалось прочитать состояние планировщика: {exc}") from exc
    if state.job is None:
        raise ToolError("job repo_activity ещё не создан — сначала вызови schedule_job")
    return scheduler.summarize(state, minutes=minutes)


if __name__ == "__main__":
    mcp.run()
