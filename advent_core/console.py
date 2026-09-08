"""Единый вывод на все недели курса.

Ответ модели идёт в stdout, всё служебное — в stderr. Это делает
`advent w01 chat "..." > answer.txt` корректным.
"""

from __future__ import annotations

import json
import sys
from contextlib import suppress

from rich.console import Console
from rich.table import Table

from advent_core.errors import AdventError
from advent_core.telemetry import CallResult


def force_utf8() -> None:
    """Принудительный UTF-8 на потоках, включая stdin.

    Консоль Windows по умолчанию отдаёт cp1251, и кириллица в ответе модели
    превращается в мусор прямо в кадре видео. Вызывается первым делом в CLI.

    stdin здесь не ради симметрии. Когда ввод приходит из ПАЙПА, а не из
    консоли, Python берёт кодировку из локали и ставит обработчик
    surrogateescape: замерено 2026-09-07, `sys.stdin.encoding == "cp1252"`,
    `sys.stdin.errors == "surrogateescape"`. Кириллица в UTF-8 почти вся
    как-то отображается в cp1252, но байт 0x81 в cp1252 НЕ ОПРЕДЕЛЁН, и
    surrogateescape превращает его в одинокий суррогат. А 0x81 — это второй
    байт буквы «с» (U+0441). Такая строка уходит в модель испорченной и
    роняет запись в журнал: json.dumps её собирает молча, а запись в файл
    падает с "surrogates not allowed".

    Отсюда две неочевидные вещи. Баг ЗАВИСИТ ОТ ДАННЫХ: "привет" проходит,
    "число" падает — поэтому он и дожил незамеченным. И он есть в неделе 01
    тоже: `advent w01 chat` из пайпа падает ровно так же. Там его маскировало
    то, что advent_cli/record.py выставляет дочернему процессу
    PYTHONIOENCODING=utf-8, а интерактивная консоль Windows и так отдаёт
    stdin в UTF-8 — ломается только пайп.

    Интерактивному вводу это не вредит: там Python работает через консольный
    API уже в UTF-8, и reconfigure оказывается пустой операцией.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


out = Console(soft_wrap=True)
err = Console(stderr=True)


def clear_screen() -> None:
    """Чистит видимый экран И буфер прокрутки терминала.

    Нужно ровно одному потребителю — `advent record` перед стартом записи.
    OBS снимает окно целиком, а не поток вывода, поэтому в первые кадры
    попадает всё, что осталось на экране от подготовки: репетиция пайпа с её
    `/params`, заведомо неверной командой и `/exit`. На видео это читается
    как «ролик начался с середины чужой сессии».

    Именно `3J`, а не только `2J`: без него Windows Terminal оставляет
    прежние строки в буфере прокрутки, и они уезжают вверх, а не исчезают —
    в кадре это выглядит так же грязно. `H` возвращает курсор в начало,
    иначе вывод пойдёт с той строки, где стоял курсор.

    В stderr, а не в stdout: контракт проекта — в stdout только ответ модели,
    а управляющая последовательность им не является. Терминал у обоих потоков
    один и тот же, так что на картинку это не влияет.
    """
    sys.stderr.write("\033[2J\033[3J\033[H")
    sys.stderr.flush()


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


def print_answer(result: CallResult, format_name: str | None) -> None:
    """Печатает нестримленный ответ в stdout (контракт: ответ — в stdout).

    format=json/schema печатается разобранным и с отступами — это тоже
    часть демонстрации дня: три `/again` подряд читаются глазами за секунду,
    а не построчным сравнением сырых строк (SPEC-w01d02.md §6.6). Условие —
    result.format_ok is True: formats.verify() уже разобрал JSON один раз,
    повторный json.loads() здесь — не вторая проверка, а просто способ
    получить объект для pretty-print без второго источника истины насчёт
    того, валиден ли ответ. Битый JSON (format_ok is False/None) печатается
    как есть — не терять текст ответа на несовпадении формата.
    """
    text = result.text
    if format_name in ("json", "schema") and result.format_ok:
        # format_ok уже сказал "валиден" — suppress здесь на случай, если это
        # когда-нибудь разойдётся, а не как ожидаемый путь выполнения.
        with suppress(json.JSONDecodeError):
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    write_chunk(text)
    finish_answer()


def footer(result: CallResult) -> None:
    """Телеметрия после ответа: модель, latency, токены, finish_reason, формат."""
    model = result.model_actual or result.model_requested
    if result.model_actual and result.model_actual != result.model_requested:
        model = f"{result.model_requested} → {result.model_actual}"

    parts = [f"model {model}", f"{result.latency_ms} ms"]
    usage = result.usage
    if not usage.is_empty():
        parts.append(f"tokens {usage.prompt_tokens}/{usage.completion_tokens}/{usage.total_tokens}")
    if result.truncated:
        parts.append("[yellow]ответ оборван[/yellow]")

    err.print(f"[dim]· {'  ·  '.join(parts)}[/dim]")

    # Вторая строка: finish_reason отличает "модель закончила сама" от
    # "упёрлась в max_tokens" (главный сигнал дня, SPEC-w01d02.md §6.3), и
    # вердикт по формату. Обе части опциональны по отдельности — CallResult,
    # собранный вручную для лога ошибки (week_01/cli.py), может не нести ни
    # одной из них.
    detail_bits: list[str] = []
    if result.finish_reason:
        length_ish = result.finish_reason in ("length", "model_length")
        detail_bits.append(
            f"[yellow]finish={result.finish_reason} ⚠[/yellow]"
            if length_ish
            else f"finish={result.finish_reason}"
        )
    if result.format_detail is not None:
        detail = result.format_detail
        if result.format_ok is False and result.finish_reason in ("length", "model_length"):
            # Обрыв по длине — самая частая причина невалидного JSON; явное
            # слово рядом с "✗" избавляет от догадок, глядя на footer.
            detail += " обрыв"
        style = {True: "green", False: "red"}.get(result.format_ok)
        detail_bits.append(f"[{style}]{detail}[/{style}]" if style else detail)
    if detail_bits:
        err.print(f"[dim] {'  ·  '.join(detail_bits)}[/dim]")


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


def models_table(
    models: list[dict], highlight: str | None = None, *, target: Console | None = None
) -> None:
    """Таблица моделей: id, контекст, возможности, рекомендованная температура.

    `target` по умолчанию `out`, потому что для `advent w01 models` таблица —
    это ПРОДУКТ команды, а продукт по контракту проекта идёт в stdout. Агент
    недели 02 зовёт ту же функцию с `target=err`: там та же таблица показана
    по слэш-команде посреди разговора, то есть это хром, а продукт — ответ
    модели (SPEC-w02d06.md §14).
    """
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

    (target or out).print(table)


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


def model_card(model: dict, requested: str, *, target: Console | None = None) -> None:
    """Карточка одной модели для `/model info`. См. про `target` в models_table."""
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

    (target or out).print(table)

    # Про снятие с поддержки нужно узнавать заранее, а не по внезапной 404.
    if deprecation := model.get("deprecation"):
        replacement = model.get("deprecation_replacement_model") or "замена не указана"
        warn(f"модель снимается с поддержки {deprecation}; замена: {replacement}")


def params_table(rows: list[tuple[str, str, str]], *, target: Console | None = None) -> None:
    """Текущие параметры генерации для `/params`. См. про `target` в models_table."""
    table = Table(title="Параметры генерации")
    table.add_column("параметр", style="cyan", no_wrap=True)
    table.add_column("значение", justify="right")
    table.add_column("что делает", style="dim")

    for name, value, help_text in rows:
        table.add_row(name, value, help_text)

    (target or out).print(table)
    note("— означает «не передаётся, сервер применит свой дефолт»")


def commands_help(commands: list[tuple[str, str]], *, target: Console | None = None) -> None:
    """Список команд REPL для `/help`. См. про `target` в models_table."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim")
    for name, description in commands:
        table.add_row(name, description)
    (target or out).print(table)
