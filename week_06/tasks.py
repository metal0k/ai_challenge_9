"""The three comparison tasks and their code-based checks (no judge)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from advent_core.config import ConfigError
from week_01.strategies import (
    VERDICT_LABELS,
    check_answer,
    load_problem,
    marker_instruction,
)
from week_06 import codecheck

TASKS_FILE = Path(__file__).with_name("tasks.json")

V_OK = "ok"
V_WRONG = "wrong"
V_NO_MARKER = "no_marker"
V_PARTIAL = codecheck.VERDICT_PARTIAL
V_NO_CODE = codecheck.VERDICT_NO_CODE
V_UNSAFE = codecheck.VERDICT_UNSAFE
V_ERROR = codecheck.VERDICT_ERROR

FACT_ACCEPT = frozenset({"канберра", "canberra"})
_WRAPPERS = re.compile(r"\*+|`+|[«»\"'“”„‘’]")


@dataclass(slots=True, frozen=True)
class Task:
    id: str
    level: str
    prompt: str
    check: str


@dataclass(slots=True)
class TaskVerdict:
    verdict: str
    label: str
    ok: bool
    detail: str | None = None
    code: codecheck.CodeCheck | None = None


def load_tasks(path: Path | None = None) -> dict[str, Task]:
    """Tasks keyed by id in file order; `alice` comes from the week-01 bank."""
    raw = json.loads((path or TASKS_FILE).read_text(encoding="utf-8"))
    tasks: dict[str, Task] = {}
    for item in raw:
        if "problem" in item:
            # Marker line only: no hint at the trap, no persona (see CLAUDE.md).
            prompt = load_problem(item["problem"]).statement + "\n\n" + marker_instruction()
        else:
            prompt = item["prompt"]
        tasks[item["id"]] = Task(item["id"], item["level"], prompt, item["check"])
    return tasks


def parse_task_ids(value: str, tasks: dict[str, Task]) -> list[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [i for i in ids if i not in tasks]
    if unknown or not ids:
        raise ConfigError(
            f"Неизвестные задачи: {', '.join(unknown) or value!r}. Есть: {', '.join(tasks)}"
        )
    return ids


def normalize_fact(text: str) -> str:
    cleaned = _WRAPPERS.sub("", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t.,!;:…?-—")
    return cleaned.casefold().replace("ё", "е")


def check_fact(text: str) -> TaskVerdict:
    """Exact match after stripping formatting: 'Не Канберра, а Сидней' fails."""
    if normalize_fact(text) in FACT_ACCEPT:
        return TaskVerdict(V_OK, "✓ верно", True)
    return TaskVerdict(V_WRONG, "✗ неверно", False)


def check_task(task_id: str, text: str) -> TaskVerdict:
    """Dispatch the per-task check; returns a verdict plus a printable label."""
    if task_id == "fact":
        return check_fact(text)
    if task_id == "alice":
        result = check_answer(text, load_problem("alice"))
        label = VERDICT_LABELS[result.verdict]
        verdict = {"ok": V_OK, "no_marker": V_NO_MARKER}.get(result.verdict, V_WRONG)
        return TaskVerdict(verdict, label, result.ok, result.answer)
    if task_id == "palindrome":
        code = codecheck.check_code_answer(text)
        parts = []
        if code.failed:
            parts.append("упали: " + "; ".join(code.failed))
        elif code.error:
            parts.append(code.error)
        if code.executed:
            shown = {"ok": "ок", "failed": "упали", "error": "ошибка"}.get(code.own_asserts, "нет")
            parts.append(f"свои тесты: {shown}")
        detail = " · ".join(parts) or None
        return TaskVerdict(code.verdict, code.label, code.verdict == V_OK, detail, code)
    raise ConfigError(f"Нет проверки для задачи {task_id!r}.")
