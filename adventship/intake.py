"""Stage `intake`: cut the `## Day NN` section out of `week_NN/task_NN.md`."""

from __future__ import annotations

import re
from pathlib import Path

from adventship.ctx import EXIT_CONFIG, ShipContext, ShipError, StageResult, TaskSpec

_HEADING = re.compile(r"^## Day (\d+)[ \t]*$", re.MULTILINE)


def parse_day_section(text: str, day: int) -> str | None:
    """Body of the `## Day <day>` section, heading included; None if absent."""
    text = text.replace("\r\n", "\n")
    headings = list(_HEADING.finditer(text))
    for index, match in enumerate(headings):
        if int(match.group(1)) == day:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            return text[match.start() : end].strip()
    return None


def default_task_file(root: Path, week: int) -> Path:
    return root / f"week_{week:02d}" / f"task_{week:02d}.md"


def _display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_task(root: Path, week: int, day: int, task_file: Path | None = None) -> TaskSpec:
    path = (task_file or default_task_file(root, week)).resolve()
    source = _display(root, path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ShipError(
            f"файл задания не найден: {source}",
            exit_code=EXIT_CONFIG,
            hint="Положи задание в week_NN/task_NN.md или укажи --task-file.",
        ) from None
    section = parse_day_section(raw, day)
    if not section:
        raise ShipError(
            f"в {source} нет раздела «## Day {day:02d}» с текстом",
            exit_code=EXIT_CONFIG,
        )
    return TaskSpec(week=week, day=day, source=source, text=section)


def run_intake(ctx: ShipContext) -> StageResult:
    task = load_task(ctx.root, ctx.week, ctx.day, ctx.task_file)
    ctx.task = task
    return StageResult(
        data={
            "task_hash": task.task_hash,
            "week": task.week,
            "day": task.day,
            "source": task.source,
        },  # no text/title: stages re-read the task file via ctx.task
        summary=f"{task.source}: {task.title} ({len(task.text)} симв.)",
    )
