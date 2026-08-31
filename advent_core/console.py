"""Единый вывод на все недели курса.

Ответ модели идёт в stdout, всё служебное — в stderr. Это делает
`advent w01 chat "..." > answer.txt` корректным.
"""

from __future__ import annotations

import sys
from contextlib import suppress

from rich.console import Console
from rich.table import Table

from advent_core.errors import AdventError
from advent_core.telemetry import CallResult


def force_utf8() -> None:
    """Принудительный UTF-8 на потоках.

    Консоль Windows по умолчанию отдаёт cp1251, и кириллица в ответе модели
    превращается в мусор прямо в кадре видео. Вызывается первым делом в CLI.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


out = Console(soft_wrap=True)
err = Console(stderr=True)


def write_chunk(text: str) -> None:
    """Печатает кусок стрима как есть.

    Сырой текст, без markdown-рендера: rich.Live с перерисовкой мигает на
    длинных ответах и ломается при редиректе в файл.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


def finish_answer() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def footer(result: CallResult) -> None:
    """Телеметрия после ответа: модель, latency, токены."""
    model = result.model_actual or result.model_requested
    if result.model_actual and result.model_actual != result.model_requested:
        model = f"{result.model_requested} → {result.model_actual}"

    parts = [f"model {model}", f"{result.latency_ms} ms"]
    usage = result.usage
    if not usage.is_empty():
        parts.append(
            f"tokens {usage.prompt_tokens}/{usage.completion_tokens}/{usage.total_tokens}"
        )
    if result.truncated:
        parts.append("[yellow]ответ оборван[/yellow]")

    err.print(f"[dim]· {'  ·  '.join(parts)}[/dim]")


def echo_input(text: str) -> None:
    """Эхо строки, поданной в stdin из скрипта записи.

    В stderr, а не в stdout: контракт «ответ модели — в stdout, служебное — в
    stderr» держит редирект в файл чистым.
    """
    err.print(text, markup=False, highlight=False)


def note(message: str) -> None:
    err.print(f"[dim]{message}[/dim]")


def warn(message: str) -> None:
    err.print(f"[yellow]{message}[/yellow]")


def fail(error: AdventError) -> None:
    """Ошибка человеческим текстом, без traceback."""
    err.print(f"[bold red]Ошибка:[/bold red] {error.message}")
    if error.hint:
        err.print(f"[dim]{error.hint}[/dim]")


def models_table(models: list[dict], highlight: str | None = None) -> None:
    """Таблица моделей: id, контекст, возможности, рекомендованная температура."""
    table = Table(title="Модели Mistral, доступные аккаунту")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("ctx", justify="right")
    table.add_column("t°", justify="right")
    table.add_column("возможности", style="dim")
    table.add_column("алиасы", style="dim")

    for model in sorted(models, key=lambda m: m.get("id", "")):
        model_id = model.get("id", "")
        aliases = model.get("aliases") or []
        is_current = highlight and (model_id == highlight or highlight in aliases)

        row_style = "bold green" if is_current else None
        if model.get("deprecation"):
            row_style = row_style or "yellow"

        table.add_row(
            model_id,
            _short_ctx(model.get("max_context_length")),
            _short_temp(model.get("default_model_temperature")),
            _short_caps(model.get("capabilities") or {}),
            ", ".join(aliases) or "—",
            style=row_style,
        )

    out.print(table)


# Порядок задаёт приоритет в узкой колонке: то, что важнее для выбора модели.
CAPABILITY_LABELS = (
    ("reasoning", "reasoning"),
    ("vision", "vision"),
    ("function_calling", "tools"),
    ("completion_fim", "fim"),
    ("audio", "audio"),
    ("fine_tuning", "ft"),
)


def _short_caps(capabilities: dict) -> str:
    names = [label for key, label in CAPABILITY_LABELS if capabilities.get(key)]
    return ", ".join(names) or "—"


def _short_ctx(value: object) -> str:
    if not isinstance(value, int):
        return "—"
    return f"{value // 1024}k" if value >= 1024 else str(value)


def _short_temp(value: object) -> str:
    return "—" if value is None else str(value)


def model_card(model: dict, requested: str) -> None:
    """Карточка одной модели для `/model info`."""
    capabilities = model.get("capabilities") or {}
    enabled = [label for key, label in CAPABILITY_LABELS if capabilities.get(key)]

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("запрошена", requested)
    table.add_row("реальная версия", model.get("id", "—"))
    if description := model.get("description"):
        table.add_row("описание", description)
    table.add_row("контекст", _short_ctx(model.get("max_context_length")))
    table.add_row("рекоменд. t°", _short_temp(model.get("default_model_temperature")))
    table.add_row("возможности", ", ".join(enabled) or "—")
    table.add_row("алиасы", ", ".join(model.get("aliases") or []) or "—")

    out.print(table)

    # Про снятие с поддержки нужно узнавать заранее, а не по внезапной 404.
    if deprecation := model.get("deprecation"):
        replacement = model.get("deprecation_replacement_model") or "замена не указана"
        warn(f"модель снимается с поддержки {deprecation}; замена: {replacement}")


def params_table(rows: list[tuple[str, str, str]]) -> None:
    """Текущие параметры генерации для `/params`."""
    table = Table(title="Параметры генерации")
    table.add_column("параметр", style="cyan", no_wrap=True)
    table.add_column("значение", justify="right")
    table.add_column("что делает", style="dim")

    for name, value, help_text in rows:
        table.add_row(name, value, help_text)

    out.print(table)
    note("— означает «не передаётся, сервер применит свой дефолт»")


def commands_help(commands: list[tuple[str, str]]) -> None:
    """Список команд REPL для `/help`."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim")
    for name, description in commands:
        table.add_row(name, description)
    out.print(table)
