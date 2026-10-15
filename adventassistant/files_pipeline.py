"""Day 34 pipeline: an agent loop over the files MCP server, ending in a reviewable diff.

The model only stages changes (changeset on disk); this module prints the diff built
by code and never touches the working tree. `files-apply` writes the saved changeset.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from advent_core import console
from advent_core.agent import Agent, ToolCallLog
from advent_core.config import PROJECT_ROOT, redact
from advent_core.errors import AdventError, StreamTruncated, translate
from advent_core.journal import log_call
from advent_core.mcp_client import ToolCallOutcome, call_tool_once, connect_and_list
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult
from adventassistant import files_store
from adventassistant.corpus import cloud_config
from week_01.models_bench import call_cost, load_prices

WEEK = 7
DAY = 34
ANSWER_MODEL = "ministral-14b-latest"
_QUIET_WARNINGS = ("max_context_length", "счётчика токенов нет")
TOOL_TIMEOUT = 15.0
MAX_GOAL_CHARS = 2000
MAX_ROUNDS_RANGE = (1, 15)
MAX_CALLS_RANGE = (1, 60)
DEFAULT_MAX_ROUNDS = 12
DEFAULT_MAX_CALLS = 24
READ_CALLS_BUDGET = 30
READ_CHARS_BUDGET = 250_000
LIMIT_EXIT_CODE = 11
READ_BUDGET_TEXT = "бюджет чтения исчерпан: больше не читай, закончи ответ по собранному"
PARTIAL_MARK = "частичный результат"
_FILES_PROMPT = (
    (Path(__file__).resolve().parent / "prompts" / "files.md").read_text(encoding="utf-8").strip()
)
CompleteFn = Callable[..., CallResult]
CallToolFn = Callable[[str, dict[str, Any]], ToolCallOutcome]


class LimitReachedError(AdventError):
    """Rounds, calls or volume ran out before the model gave a final answer."""

    exit_code = LIMIT_EXIT_CODE


@dataclass(slots=True)
class FilesResult:
    run_id: str
    answer: str
    diff: str
    rounds: int
    calls: int


@dataclass(slots=True)
class _Stats:
    calls: int = 0
    read_calls: int = 0
    read_chars: int = 0
    read_paths: set[str] = field(default_factory=set)


def _check_int(name: str, value: object, bounds: tuple[int, int]) -> int:
    low, high = bounds
    if type(value) is not int or not low <= value <= high:
        raise AdventError(f"{name} должен быть целым числом от {low} до {high}.")
    return value


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdventError(f"Не удалось запустить git: {exc}") from exc


def _require_repo(root: Path) -> None:
    proc = _git(root, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise AdventError(f"{root} не является git-репозиторием: ассистенту нечего читать.")


def _dirty_tracked(root: Path) -> list[str]:
    proc = _git(root, "status", "--porcelain", "--untracked-files=no")
    if proc.returncode != 0:
        return []
    return [line[3:] for line in proc.stdout.splitlines() if len(line) > 3]


def _warn(text: str) -> None:
    # Window-fill notices are noise here: the tool loop sends no history to trim.
    if not any(marker in text for marker in _QUIET_WARNINGS):
        console.warn(text)


def _stderr(line: str) -> None:
    console.err.print(line, markup=False, highlight=False)


# --- trace ----------------------------------------------------------------------


def _short(value: object, limit: int = 40) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def short_args(name: str, arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        if name == "write_file" and key == "content" and isinstance(value, str):
            parts.append(f"content=<{len(value)} симв.>")
        elif name == "edit_file" and key in ("old", "new") and isinstance(value, str):
            parts.append(f"{key}=<{len(value)} симв.>")
        else:
            parts.append(f"{key}={_short(value)}")
    return ", ".join(parts)


_READ_HEAD = re.compile(r"строки (\d+)-(\d+) из (\d+)")
_SEARCH_TAIL = re.compile(r"совпадений: (\d+) в файлах: (\d+)")
_LIST_TAIL = re.compile(r"всего: (\d+)")


def summarize_result(name: str, text: str, is_error: bool) -> str:
    if is_error:
        first = text.strip().splitlines()[0] if text.strip() else "без текста"
        return f"ОШИБКА: {_short(first, 120)}"
    lines = text.strip().splitlines()
    if name == "search_text" and lines and (m := _SEARCH_TAIL.search(lines[-1])):
        extra = " (скан обрезан)" if "скан обрезан" in lines[-1] else ""
        return f"{m.group(1)} совпадений в {m.group(2)} файлах{extra}"
    if name == "list_files" and lines and (m := _LIST_TAIL.search(lines[-1])):
        return f"{m.group(1)} файлов"
    if name == "read_file" and lines and (m := _READ_HEAD.search(lines[0])):
        return f"прочитано {m.group(1)}–{m.group(2)} из {m.group(3)}"
    return _short(lines[0] if lines else "", 120)


# --- server wiring -----------------------------------------------------------------


def _server_command(run_id: str, root: Path) -> tuple[list[str], Path]:
    """Command and cwd. `--root` only for a non-default root (tests), then cwd keeps imports."""
    command = [sys.executable, "-m", "adventassistant.files_server", "--run-id", run_id]
    if root.resolve() == PROJECT_ROOT.resolve():
        return command, root
    return [*command, "--root", str(root)], PROJECT_ROOT


def _function_tools(command: list[str], cwd: Path) -> list[dict[str, Any]]:
    try:
        listed = connect_and_list(
            command, timeout=TOOL_TIMEOUT, raw=False, sink=lambda _line: None, cwd=cwd, quiet=True
        )
    except AdventError:
        raise
    except Exception as exc:
        raise translate(exc) from exc
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in listed.tools
    ]


def _complete_guard(complete: CompleteFn) -> CompleteFn:
    def guarded(*args: Any, **kwargs: Any) -> CallResult:
        try:
            return complete(*args, **kwargs)
        except AdventError:
            raise
        except Exception as exc:
            raise translate(exc) from exc

    return guarded


# --- journal / footer -----------------------------------------------------------


def _journal(results: list[CallResult], run_id: str, first_round: int = 1) -> None:
    for number, result in enumerate(results, start=first_round):
        log_call(
            result,
            list(result.sent_messages or []),
            week=WEEK,
            day=DAY,
            extra={
                "command": "assistant_files",
                "run_id": run_id,
                "round": number,
                "tool_calls": [call.name for call in result.tool_calls],
            },
        )


def _footer(results: list[CallResult], stats: _Stats, model: str) -> None:
    prompt = sum(r.usage.prompt_tokens or 0 for r in results)
    completion = sum(r.usage.completion_tokens or 0 for r in results)
    cost = "—"
    try:
        price = load_prices().price_of(model)
    except Exception:
        price = None
    if price is not None and results:
        costs = [
            call_cost(
                r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens, price
            )
            for r in results
        ]
        known = [c for c in costs if c is not None]
        if known:
            cost = f"${sum(known):.4f}" + ("+" if len(known) < len(costs) else "")
    if results:
        console.footer(results[-1])
    console.note(
        f"· раундов {len(results)}  ·  вызовов {stats.calls}  ·  "
        f"прочитано файлов {len(stats.read_paths)} / символов {stats.read_chars}"
    )
    console.note(f"· токены {prompt}/{completion}/{prompt + completion}  ·  стоимость {cost}")


# --- output ---------------------------------------------------------------------


def _stdout(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def render_output(answer: str, diff: str, run_id: str, *, partial: bool) -> str:
    parts: list[str] = []
    if answer.strip():
        parts.append(answer.strip())
    mark = f" ({PARTIAL_MARK})" if partial else ""
    if diff:
        parts.append(f"Изменения{mark}:\n{diff.rstrip(chr(10))}")
        parts.append(f"run: {run_id}\nприменить: adventassistant files-apply {run_id}")
    else:
        parts.append(f"Изменений нет.{mark}")
    return "\n\n".join(parts)


def _safe_diff(run_id: str, root: Path) -> str:
    try:
        raw = files_store.read_diff(run_id, root)
        shown = redact(raw)
        if shown != raw:  # the server child may not know a secret the parent does
            files_store.write_saved_diff(run_id, shown, root)
        return shown
    except AdventError:
        return ""


# --- main entry -----------------------------------------------------------------


def run_files(
    goal: str,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_calls: int = DEFAULT_MAX_CALLS,
    complete: CompleteFn,
    call_tool: CallToolFn | None = None,
    root: Path | None = None,
    config: Any = None,
) -> FilesResult:
    """Run the agent and print answer + diff to stdout; trace and footer go to stderr."""
    if not isinstance(goal, str):
        raise AdventError("Цель должна быть текстом.")
    text = goal.strip()
    if not text:
        raise AdventError("Пустая цель.")
    if len(text) > MAX_GOAL_CHARS:
        raise AdventError(f"Цель длиннее {MAX_GOAL_CHARS} символов.")
    _check_int("--max-rounds", max_rounds, MAX_ROUNDS_RANGE)
    _check_int("--max-calls", max_calls, MAX_CALLS_RANGE)
    root = Path(root) if root is not None else PROJECT_ROOT
    _require_repo(root)

    cfg = config if config is not None else cloud_config(ANSWER_MODEL)
    cfg = replace(cfg, system_prompt_path=None, params=GenerationParams())

    dirty = _dirty_tracked(root)
    if dirty:
        console.warn(
            "незакоммиченные правки tracked-файлов могут уйти в облако вместе с прочитанным: "
            + ", ".join(dirty)
        )

    run_id, run_dir = files_store.create_run(text, cfg.model, root)
    command, cwd = _server_command(run_id, root)
    tools = _function_tools(command, cwd)

    stats = _Stats()

    def default_call(name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
        try:
            return call_tool_once(
                command, name, arguments, timeout=TOOL_TIMEOUT, cwd=cwd, quiet=True
            )
        except AdventError:
            raise
        except Exception as exc:
            raise translate(exc) from exc

    def budgeted_call(name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
        if name == "read_file" and (
            stats.read_calls >= READ_CALLS_BUDGET or stats.read_chars >= READ_CHARS_BUDGET
        ):
            return ToolCallOutcome(text=READ_BUDGET_TEXT, is_error=True)
        try:
            outcome = (call_tool or default_call)(name, arguments)
        except AdventError as error:
            stats.calls += 1
            shown = _short(error.message, 120)
            _stderr(f"· {name}({short_args(name, arguments)}) → ОШИБКА MCP: {shown}")
            raise
        if name == "read_file" and not outcome.is_error:
            stats.read_calls += 1
            stats.read_chars += len(outcome.text)
            path = arguments.get("path")
            if isinstance(path, str):
                stats.read_paths.add(path.replace("\\", "/").casefold())
        return outcome

    def trace(log: ToolCallLog) -> None:
        stats.calls += 1
        _stderr(
            f"· {log.name}({short_args(log.name, log.arguments)}) → "
            f"{summarize_result(log.name, log.result_text, log.is_error)}"
        )

    guarded = _complete_guard(complete)
    agent = Agent(
        cfg,
        complete=guarded,
        stream=guarded,
        persona=_FILES_PROMPT,
        on_warning=_warn,
        tools=tools,
        call_tool=budgeted_call,
        mcp_enabled=True,
        max_tool_rounds=max_rounds,
        on_tool_call=trace,
        max_tool_calls=max_calls,
    )

    try:
        reply = agent.ask(text, [])
    except Exception as exc:
        error = exc if isinstance(exc, AdventError) else translate(exc)
        paid = agent.take_pending_tool_rounds()
        _journal(paid, run_id)
        diff = _safe_diff(run_id, root)
        _stdout(render_output("", diff, run_id, partial=True))
        _footer(paid, stats, cfg.model)
        raise error from None

    results = list(reply.tool_round_results)
    # On limit exhaustion the last round is both parked and the reply's result.
    if not results or results[-1] is not reply.result:
        results.append(reply.result)
    _journal(results, run_id)

    diff = _safe_diff(run_id, root)
    limit = reply.tool_limit_hit
    reason = reply.result.finish_reason
    # Anything but a clean stop (error, length, ...) is a cut-off, not a final answer.
    cut = limit is None and reason not in (None, "stop")
    _stdout(
        render_output("" if cut else reply.text, diff, run_id, partial=limit is not None or cut)
    )
    if diff:
        _stderr(f"diff: {(run_dir / 'changes.diff').relative_to(root).as_posix()}")
    _footer(results, stats, cfg.model)
    if limit is not None:
        what = "раундов" if limit == "rounds" else "вызовов инструментов"
        raise LimitReachedError(
            f"Лимит {what} исчерпан, финального ответа нет. Прогон {run_id} сохранён.",
            hint="Увеличьте --max-rounds/--max-calls или уточните цель.",
        )
    if cut:
        if reply.text.strip():
            _stderr(f"обрезанный текст модели: {_short(reply.text.strip(), 200)}")
        raise StreamTruncated(
            f"Модель оборвала ответ (finish_reason={reason}), финального ответа нет. "
            f"Прогон {run_id} сохранён.",
            hint="Повторите запуск или уточните цель.",
        )
    return FilesResult(run_id, reply.text, diff, len(results), stats.calls)
