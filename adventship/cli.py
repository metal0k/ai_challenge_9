"""Entry point `adventship`: run the day-submission pipeline, inspect it, drive the sheet.

The status table is the product (stdout); stage progress, gates and errors go to stderr.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.config import PROJECT_ROOT, ConfigError, load_env
from advent_core.errors import AdventError
from adventship import gitutil
from adventship.ctx import ShipContext, ShipError, run_command
from adventship.gates import tty_gate
from adventship.stages import load_fingerprint, run_pipeline, status_rows
from adventship.state import STAGES, ShipState

DEFAULT_WEEK = 7
DEFAULT_DAY = 35

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Сдача дня курса: от готового кода до комментария в таблице.",
)


def day_tag(week: int, day: int) -> str:
    return f"w{week:02d}d{day:02d}"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _driver_factory(env: dict[str, str], root: Path, day: int):
    """Built lazily: Playwright is an optional dependency and tests never import it."""

    def make():
        from adventship.sheet_driver import PlaywrightDriver

        return PlaywrightDriver.from_env(env, root, day)

    return make


def build_context(
    week: int,
    day: int,
    *,
    task_file: Path | None = None,
    dry_run: bool = False,
    no_record: bool = False,
    root: Path | None = None,
) -> ShipContext:
    root = (root or PROJECT_ROOT).resolve()
    tag = day_tag(week, day)
    workdir = root / "logs" / "ship" / tag
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    def run(argv, *, timeout, input=None, env=None, inherit_env=True):
        return run_command(
            argv, timeout=timeout, input=input, env=env, inherit_env=inherit_env, cwd=root
        )

    return ShipContext(
        week=week,
        day=day,
        tag=tag,
        root=root,
        workdir=workdir,
        env=env,
        state=ShipState.load(workdir / "state.json"),
        run=run,
        complete=chat_core.complete,
        gate=tty_gate,
        driver_factory=_driver_factory(env, root, day),
        now=_now,
        task_file=task_file,
        dry_run=dry_run,
        no_record=no_record,
        fingerprint=gitutil.code_fingerprint,
    )


def _print_status(ctx: ShipContext) -> None:
    load_fingerprint(ctx, tolerant=True)
    table = Table(title=f"adventship: {ctx.tag}", show_header=True)
    for column in ("стадия", "статус", "когда"):
        table.add_column(column)
    for stage, status, at in status_rows(ctx.state):
        table.add_row(stage, status, at)
    console.out.print(table)


@app.command("run")
def run_cmd(
    day: int = typer.Option(..., "--day", "-d", help="Номер дня внутри недели."),
    week: int = typer.Option(DEFAULT_WEEK, "--week", "-w", help="Номер недели."),
    task_file: Path | None = typer.Option(None, "--task-file", help="Файл с заданием."),
    from_stage: str | None = typer.Option(None, "--from", help="Начать с этой стадии."),
    only: str | None = typer.Option(None, "--only", help="Выполнить только эту стадию."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Только план, без внешних действий."),
    no_record: bool = typer.Option(False, "--no-record", help="Пропустить запись видео."),
) -> None:
    """Проводит день через стадии; продолжает с первой незавершённой."""
    load_env()
    ctx = build_context(week, day, task_file=task_file, dry_run=dry_run, no_record=no_record)
    skip = {"record"} if no_record else set()
    try:
        run_pipeline(ctx, from_stage=from_stage, only=only, skip=skip)
    finally:
        _print_status(ctx)


@app.command("status")
def status_cmd(
    day: int = typer.Option(DEFAULT_DAY, "--day", "-d"),
    week: int = typer.Option(DEFAULT_WEEK, "--week", "-w"),
) -> None:
    """Таблица стадий и их статус."""
    _print_status(build_context(week, day))


@app.command("sheet-login")
def sheet_login(
    day: int = typer.Option(DEFAULT_DAY, "--day", "-d"),
    week: int = typer.Option(DEFAULT_WEEK, "--week", "-w"),
) -> None:
    """Открыть Chrome с постоянным профилем для ручного входа в Google."""
    load_env()
    from adventship import sheet

    sheet.run_login(build_context(week, day))


@app.command("sheet-probe")
def sheet_probe(
    day: int = typer.Option(DEFAULT_DAY, "--day", "-d"),
    week: int = typer.Option(DEFAULT_WEEK, "--week", "-w"),
) -> None:
    """Прочитать раскладку таблицы и слот; ничего не менять."""
    load_env()
    from adventship import sheet

    result = sheet.run_probe(build_context(week, day))
    console.out.print(result.summary, markup=False)


@app.command("sheet-comment")
def sheet_comment(
    day: int = typer.Option(DEFAULT_DAY, "--day", "-d"),
    week: int = typer.Option(DEFAULT_WEEK, "--week", "-w"),
    text_file: Path | None = typer.Option(None, "--text-file", help="Готовый текст комментария."),
) -> None:
    """Только последняя стадия: комментарий в таблицу (нужны выполненные предыдущие)."""
    load_env()
    ctx = build_context(week, day)
    if text_file is not None:
        ctx.env["SHIP_COMMENT_TEXT_FILE"] = str(text_file.resolve())
    try:
        run_pipeline(ctx, only="sheet")
    finally:
        _print_status(ctx)


def main() -> None:
    """Точка входа. Переводит ошибки в человеческий текст и exit code."""
    console.force_utf8()
    try:
        app()
    except (AdventError, ConfigError) as error:
        if isinstance(error, ConfigError):
            error = AdventError(str(error))
            error.exit_code = ConfigError.exit_code
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


__all__ = ["STAGES", "ShipError", "app", "build_context", "main"]

if __name__ == "__main__":
    sys.exit(main() or 0)
