"""Minimal read-only MCP server about this repository (stdio).

stdout carries the protocol and nothing else: a stray print() breaks framing,
so diagnostics go to stderr only. Run: `python -m week_04.server`.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from advent_core import chat as chat_core
from advent_core import journal, tokens
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError
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
    "summarize_text",
    "save_to_file",
    "commit_digest",
)
SCHEDULE_MIN = scheduler.MIN_INTERVAL_SECONDS
SCHEDULE_MAX = scheduler.MAX_INTERVAL_SECONDS

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"^w\d{2}d\d{2}$")
DEFAULT_MODEL = "ministral-14b-latest"
GIT_TIMEOUT = 10.0
GIT_LOG_MAX = 20
DIGEST_DEFAULT_N = GIT_LOG_MAX

SUMMARIZE_MODEL = DEFAULT_MODEL
SUMMARIZE_MAX_TOKENS = 300
SUMMARIZE_SYSTEM_PROMPT = (
    "Сделай краткую сводку текста ниже на русском языке: 3-5 предложений, "
    "по существу, без вступлений и оценок от себя. Отвечай только текстом "
    "сводки, без заголовков и пояснений."
)

PIPELINE_DIR = REPO_ROOT / "logs" / "pipeline"  # logs/ is fully gitignored
# Derived once from the real PIPELINE_DIR, not restated as a literal: the
# confirmation message must stay "logs/pipeline/..." even when tests
# monkeypatch PIPELINE_DIR to a tmp_path outside REPO_ROOT for isolation —
# `target.relative_to(REPO_ROOT)` would raise ValueError there (not OSError,
# so save_to_file's except would not catch it).
PIPELINE_REL = PIPELINE_DIR.relative_to(REPO_ROOT).as_posix()

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


def _git_log_process(n: int, query: str | None = None) -> subprocess.CompletedProcess[str]:
    """Shared git call for git_log and commit_digest — raw CompletedProcess,
    no interpretation of returncode/empty output (the caller decides that)."""
    cmd = ["git", "log", f"-{n}"]
    if query:
        # -F: literal substring, not regex — the user types "MCP", not a
        # pattern; -i: case-insensitive, as expected of commit-message search.
        cmd += [f"--grep={query}", "-i", "-F"]
    cmd += ["--pretty=format:%h  %ad  %an  %s", "--date=short"]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT,
        check=False,
    )


@mcp.tool(
    description="Последние N коммитов этого репозитория; с query — только те, "
    "чьё сообщение содержит подстроку (без учёта регистра)."
)
def git_log(n: int = 5, query: str | None = None) -> str:
    if not 1 <= n <= GIT_LOG_MAX:
        raise ToolError(f"n должен быть от 1 до {GIT_LOG_MAX}, получено {n}")
    try:
        proc = _git_log_process(n, query)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"git log не выполнился: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"git log завершился с ошибкой: {proc.stderr.strip()}")
    if query and not proc.stdout.strip():
        return f"коммитов по запросу «{query}» не найдено (среди последних {n})"
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


@mcp.tool(
    description="Сжать произвольный текст в краткую сводку на русском "
    "(реальный вызов Mistral, ~300 токенов ответа)."
)
def summarize_text(text: str) -> str:
    if not text.strip():
        raise ToolError("text пустой — нечего сжимать")
    try:
        config = Config.resolve(
            model=SUMMARIZE_MODEL, max_tokens=SUMMARIZE_MAX_TOKENS, stream=False
        )
    except ConfigError as exc:
        raise ToolError(str(exc)) from exc
    messages = chat_core.build_messages(text, system=SUMMARIZE_SYSTEM_PROMPT)
    try:
        result = chat_core.complete(config, messages)
    except AdventError as exc:
        raise ToolError(f"не удалось получить сводку: {exc}") from exc
    journal.log_call(result, messages, week=4, day=19, extra={"kind": "mcp_summarize"})
    return result.text.strip()


def _safe_filename(filename: str) -> str:
    name = filename.strip()
    if not name or Path(name).name != name or ".." in name:
        raise ToolError(f"недопустимое имя файла: {filename!r}")
    return name


@mcp.tool(
    description="Сохранить текст в файл под logs/pipeline/. filename — только "
    "имя файла (без подкаталогов и без '..')."
)
def save_to_file(content: str, filename: str) -> str:
    if not content.strip():
        raise ToolError("content пустой — нечего сохранять")
    name = _safe_filename(filename)
    target = PIPELINE_DIR / name
    try:
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"не удалось сохранить файл: {exc}") from exc
    return f"Сохранено: {PIPELINE_REL}/{name} ({len(content)} симв.)"


def _slugify(text: str) -> str:
    # \w is Unicode-aware for str patterns — keeps Cyrillic letters, not just
    # ASCII, so a Russian-language query still yields a query-derived name
    # instead of always collapsing to the "digest" fallback.
    slug = re.sub(r"[^\w]+", "-", text.lower()).strip("-_")
    return slug[:40] or "digest"


@mcp.tool(
    description="Пайплайн в один вызов: находит коммиты по запросу (git log --grep), "
    "сжимает найденное сводкой (Mistral) и сохраняет результат в logs/pipeline/. "
    "Если совпадений нет — Mistral и файл не задействуются."
)
def commit_digest(query: str, n: int = DIGEST_DEFAULT_N) -> str:
    if not query.strip():
        raise ToolError("query пустой")
    if not 1 <= n <= GIT_LOG_MAX:
        raise ToolError(f"n должен быть от 1 до {GIT_LOG_MAX}, получено {n}")

    # Step 1 — search: same git call as git_log, WITHOUT going through the
    # decorated wrapper (needs the raw CompletedProcess, not a ready-made
    # string/"not found" text — control over the empty result stays here).
    try:
        proc = _git_log_process(n, query)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"git log не выполнился: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"git log завершился с ошибкой: {proc.stderr.strip()}")
    commits = [line for line in proc.stdout.splitlines() if line]
    if not commits:
        return f"Коммитов по запросу «{query}» не найдено — Mistral и файл не задействованы."

    # Step 2 — summarize: a plain Python function call (still a tool — the
    # @mcp.tool decorator returns the function unchanged).
    commit_block = "\n".join(commits)
    summary = summarize_text(f"Список коммитов по запросу «{query}»:\n\n{commit_block}")

    # Step 3 — saveToFile.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{_slugify(query)}-{stamp}.md"
    commit_list = "\n".join(f"- {line}" for line in commits)
    content = (
        f"# Сводка по запросу «{query}»\n\n{summary}\n\n"
        f"## Исходные коммиты ({len(commits)})\n\n{commit_list}\n"
    )
    saved = save_to_file(content, filename)

    return f"{saved}\n\nСводка:\n{summary}"


if __name__ == "__main__":
    mcp.run()
