"""Entry point `adventassistant`: index the docs corpus, review a diff, answer one question, REPL.

`python -m adventassistant.cli` is how `advent record` launches a step.
The product (answer or tool text, index summary) is stdout. Branch, MCP,
sources, threshold, progress and errors are stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import typer

from advent_core import chat as chat_core
from advent_core import console
from advent_core.config import PROJECT_ROOT, ConfigError
from advent_core.errors import AdventError, translate
from adventassistant.corpus import INDEX_PATH, build_index
from adventassistant.pipeline import run_help
from adventassistant.review import run_review
from adventassistant.review_index import REVIEW_INDEX_PATH, build_review_index
from adventassistant.watcher import run_watch

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help="Ассистент по документации репозитория и текущей ветке.",
)

_EXAMPLES = (
    "На какой ветке сейчас репозиторий?",
    "Из каких трёх слоёв состоит репозиторий?",
)


def help_usage() -> str:
    return (
        'Как звать: /help <вопрос> или adventassistant help "вопрос"\n'
        f"Например: /help {_EXAMPLES[0]}\n"
        f"Например: /help {_EXAMPLES[1]}"
    )


def write_stdout(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _help_body(stripped: str) -> str | None:
    """Question after `/help`, or "" when there is none. None when the line is not that command."""
    if stripped == "/help":
        return ""
    # A whitespace boundary: `/helpful` is an unknown command, not a question "ful".
    if len(stripped) > 5 and stripped.startswith("/help") and stripped[5].isspace():
        return stripped[5:].strip()
    return None


def handle_line(line: str) -> str:
    """`exit` stops the REPL. Anything else stays in it. `/help` alone does not run the pipeline."""
    stripped = line.strip()
    if stripped in {"/exit", "/quit"}:
        return "exit"
    body = _help_body(stripped)
    if body is None:
        console.warn("Неизвестная команда. Напиши /help <вопрос> или /exit.")
        return "continue"
    if not body:
        console.err.print(help_usage(), markup=False, highlight=False)
        return "continue"
    write_stdout(run_help(body, db_path=INDEX_PATH, complete=chat_core.complete))
    return "continue"


def repl() -> None:
    """Read lines until `/exit`. No pickers, no dialog history."""
    while True:
        if sys.stdin.isatty():
            console.err.print("› ", end="", markup=False, highlight=False)
        raw = sys.stdin.readline()
        if raw == "":
            return
        line = raw.rstrip("\r\n")
        if line and not sys.stdin.isatty():
            console.echo_input(line)
        if handle_line(line) == "exit":
            return


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    """Ассистент по документации репозитория и текущей ветке."""
    if ctx.invoked_subcommand is None:
        repl()


@app.command("index")
def index_command(
    rev: str = typer.Option("HEAD", "--rev", help="Ревизия git, с которой читается корпус."),
) -> None:
    """Собрать корпус документации и записать индекс дня 31."""
    write_stdout(build_index(PROJECT_ROOT, rev=rev, db_path=INDEX_PATH))


@app.command("help")
def help_command(
    question: str = typer.Argument(..., help="Вопрос о проекте."),
    db: str | None = typer.Option(None, "--db", help="Путь к индексу (для тестов)."),
) -> None:
    """Ответить на один вопрос: документация из индекса, ветка из git."""
    if not question.strip():
        raise AdventError("Пустой вопрос.")
    path = Path(db) if db else INDEX_PATH
    write_stdout(run_help(question, db_path=path, complete=chat_core.complete))


@app.command("index-review")
def index_review_command(
    rev: str = typer.Option("HEAD", "--rev", help="Ревизия git, с которой читается корпус."),
    db: str | None = typer.Option(None, "--db", help="Путь к индексу ревью (для тестов)."),
) -> None:
    """Собрать корпус ревью и записать индекс дня 32."""
    path = Path(db) if db else REVIEW_INDEX_PATH
    write_stdout(build_review_index(PROJECT_ROOT, rev=rev, db_path=path))


@app.command("review")
def review_command(
    base: str = typer.Option(..., "--base", help="База трёхточечного диапазона."),
    head: str = typer.Option(..., "--head", help="Голова трёхточечного диапазона."),
    db: str | None = typer.Option(None, "--db", help="Путь к индексу ревью (для тестов)."),
    pr: int | None = typer.Option(None, "--pr", help="Номер PR: записать комментарий после ревью."),
    diff_only: bool = typer.Option(False, "--diff-only", help="Напечатать diff и выйти."),
) -> None:
    """Ревью диапазона base...head: баги, архитектура, рекомендации."""
    path = Path(db) if db else REVIEW_INDEX_PATH
    run_review(
        PROJECT_ROOT,
        base=base,
        head=head,
        db_path=path,
        pr=pr,
        diff_only=diff_only,
        complete=chat_core.complete,
    )


@app.command("watch-pr")
def watch_pr_command(
    interval: int = typer.Option(60, "--interval", help="Пауза между проходами, секунды (от 5)."),
    once: bool = typer.Option(False, "--once", help="Один проход и выход."),
    pr: int | None = typer.Option(None, "--pr", help="Смотреть только этот PR."),
    post: bool = typer.Option(True, "--post/--no-post", help="Записывать комментарий после ревью."),
    state: str | None = typer.Option(None, "--state", help="Файл state watcher."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Каталог кэша индексов."),
) -> None:
    """Следить за открытыми PR репозитория: новое ревью на каждую пару base/head."""
    code = run_watch(
        PROJECT_ROOT,
        complete=chat_core.complete,
        interval=interval,
        once=once,
        pr=pr,
        post=post,
        state_path=Path(state) if state else None,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    if code:
        raise typer.Exit(code)


def main() -> None:
    """Entry point `adventassistant`: errors as text, not a traceback."""
    console.force_utf8()
    try:
        app()
    except AdventError as error:
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except ConfigError as error:
        console.fail(AdventError(str(error)))
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None
    except (click.exceptions.Exit, click.exceptions.Abort, click.exceptions.ClickException):
        # Typer/click own --help and usage errors. Translating them would hide the real exit.
        raise
    except Exception as exc:
        error = translate(exc)
        console.fail(error)
        raise SystemExit(error.exit_code) from None


if __name__ == "__main__":
    main()
