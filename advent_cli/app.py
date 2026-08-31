"""Корневой CLI курса: `advent`.

Неделя — явная подкоманда (`advent w01 ...`), поэтому команды прошлых недель
продолжают работать после того, как приложение уехало вперёд.
"""

from __future__ import annotations

import sys

import typer

from advent_cli.record import record
from advent_cli.submit import submit
from advent_core import console
from advent_core.config import ConfigError
from advent_core.errors import AdventError
from week_01.cli import app as week_01_app

app = typer.Typer(
    help="Задания курса AI Advent 9. Каждая неделя — своя подкоманда.",
    no_args_is_help=True,
)
app.add_typer(week_01_app, name="w01", help="Неделя 01 — первый запрос к LLM.")
app.command("record")(record)
app.command("submit")(submit)


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


if __name__ == "__main__":
    sys.exit(main() or 0)
