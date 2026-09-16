"""Единый вывод на все недели курса.

Ответ модели идёт в stdout, всё служебное — в stderr. Это делает
`advent w01 chat "..." > answer.txt` корректным.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import suppress

import typer
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


def _record_console_kwargs() -> dict[str, object]:
    """Return the explicit Rich settings needed by the Windows video take.

    Windows Terminal can expose an inherited pipe to a child process, which
    makes Rich decide that colour is unavailable even though OBS captures the
    rendered terminal window.  Keep the normal auto-detection untouched and
    opt in only for the recording subprocess.
    """
    if os.environ.get("ADVENT_RECORD_COLOR") != "1":
        return {}
    return {
        "force_terminal": True,
        "color_system": "standard",
        "legacy_windows": False,
        "no_color": False,
    }


_record_color_kwargs = _record_console_kwargs()
out = Console(soft_wrap=True, **_record_color_kwargs)
err = Console(stderr=True, **_record_color_kwargs)


def enable_record_color() -> None:
    """Enable Rich colour after the recording command has already imported us.

    ``advent record`` needs coloured preparation output as well as coloured
    child demos, but this module's consoles are normally created at import
    time. Rebuild the two shared consoles at the recording boundary so the
    OBS window gets ANSI/Rich colour even when the child inherits a pipe.
    """
    global out, err
    os.environ["ADVENT_RECORD_COLOR"] = "1"
    kwargs = _record_console_kwargs()
    out = Console(soft_wrap=True, **kwargs)
    err = Console(stderr=True, **kwargs)


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


def _token_cell(value: int | None, estimate: int | None = None) -> str:
    """Одно число в тройке tokens p/c/t: значение, оценка с тильдой или прочерк.

    is_empty() пускает в эту ветку и usage вида {"total_tokens": N} без
    prompt/completion (так отвечают локальные серверы) — тогда value is None
    без оценки, и раньше сюда попадал буквальный None из f-строки: «tokens
    None/None/51». Прочерк — то же правило проекта, что и у «tokens ?» строкой
    ниже: неизвестно — не значит «ноль» и не значит слово None.
    """
    if value is not None:
        return str(value)
    if estimate is not None:
        return f"~{estimate}"
    return "—"


def footer(result: CallResult, completion_estimate: int | None = None) -> None:
    """Телеметрия после ответа: модель, latency, токены, finish_reason, формат.

    `completion_estimate` — локальная оценка ответа на случай, когда сервер
    не прислал completion_tokens (LM Studio в стриме не присылает usage
    вовсе, SPEC-w02d08.md §6). Оценка печатается с тильдой; нет и её —
    честный «?» вместо молчания про токены: usage не пришёл — не значит
    «нечего считать», ответ есть всегда.
    """
    model = result.model_actual or result.model_requested
    if result.model_actual and result.model_actual != result.model_requested:
        model = f"{result.model_requested} → {result.model_actual}"

    parts = [f"model {model}", f"{result.latency_ms} ms"]
    usage = result.usage
    if not usage.is_empty():
        prompt = _token_cell(usage.prompt_tokens)
        completion = _token_cell(usage.completion_tokens, completion_estimate)
        total = _token_cell(usage.total_tokens)
        parts.append(f"tokens {prompt}/{completion}/{total}")
    elif completion_estimate is not None:
        parts.append(f"tokens ~{completion_estimate} (оценка)")
    else:
        parts.append("tokens ?")
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


# Потолок эха одной строки: гигантский ввод дня 08 (~700 КБ одной строкой)
# при выводе целиком прокручивал бы терминал на тысячи строк шума — в кадре
# важно, что ввод огромен, а не его текст. Реальная длина называется вслух.
ECHO_LIMIT = 200


def echo_input(text: str) -> None:
    """Эхо строки, поданной в stdin из скрипта записи.

    В stderr, а не в stdout: контракт «ответ модели — в stdout, служебное — в
    stderr» держит редирект в файл чистым.
    """
    if len(text) > ECHO_LIMIT:
        # Полная длина ввода, а не длина отрезанного хвоста: в кадре важно
        # знать, насколько огромен ввод целиком (CLAUDE.md: echo_input
        # «называет вслух реальную длину»), а не сколько символов не влезло.
        text = f"{text[:ECHO_LIMIT]}… (всего {len(text)} символов)"
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


# --------------------------------------------------------------------------
# Выбор модели при старте
# --------------------------------------------------------------------------


def _unavailable_head(model: str, *, base_url: str | None) -> str:
    """Первая строка сообщения: причина зависит от того, облако это или свой сервер.

    Без base_url модель «недоступна аккаунту». С base_url — «не найдена на
    сервере <url>»: LM Studio отдаёт свой список моделей, и совет «посмотри
    список» без --base-url показал бы ОБЛАЧНЫЙ список, то есть увёл бы от
    разгадки — живой пример: запуск против http://127.0.0.1:1234 падал с
    текстом про аккаунт и про `advent w01 models`.
    """
    if base_url is None:
        return f"Модель {model!r} недоступна аккаунту."
    return f"Модель {model!r} не найдена на сервере {base_url}."


def model_unavailable_text(model: str, names: list[str], *, base_url: str | None) -> str:
    """Полный текст ошибки для неинтерактивного случая (не-tty: пайп, CI, демо).

    Одна функция на все точки отказа — старт агента недели 02, старт REPL
    недели 01, проверка модели судьи: три копии одного текста — источник
    рассинхронизации, тут это названо прямо (CLAUDE.md).
    """
    if base_url is None:
        hint = "Посмотри список: advent w01 models"
    else:
        hint = "Укажи модель флагом --model."
    available = ", ".join(names) if names else "—"
    return f"{_unavailable_head(model, base_url=base_url)}\nДоступно: {available}\n{hint}"


def choose_model(models: list[dict], *, current: str, base_url: str | None) -> str | None:
    """Интерактивный выбор модели из списка сервера. None — отказ (Enter/EOF/Ctrl+C).

    Список передаётся уже отфильтрованным (chat_models): этот модуль не
    импортирует client.py — console владеет только контрактом вывода и вводом.
    Спрашивать, а не молча брать первую из списка, обязательно при любом числе
    кандидатов: ответ придёт от модели, которую пользователь не заказывал.

    Весь вывод и приглашение — в stderr (контракт stdout/stderr, CLAUDE.md):
    выбор модели это служебный эпизод старта, а не продукт команды. Прочитанный
    при не-tty stdin ввод эхом в stderr — как у _read_line() в week_02/cli.py.
    """
    if not models:
        return None
    choices = [model.get("id", "") for model in models]
    # Точное имя принимает и id, и алиас: конфиг живёт алиасами (-latest), и
    # заставлять пользователя угадывать каноническое имя было бы лишним.
    by_name: dict[str, str] = {}
    for model, model_id in zip(models, choices, strict=True):
        if model_id:
            by_name.setdefault(model_id, model_id)
        for alias in model.get("aliases") or []:
            by_name.setdefault(alias, model_id)

    err.print(_unavailable_head(current, base_url=base_url))
    for number, name in enumerate(choices, 1):
        err.print(f"  {number}. {name}")

    while True:
        try:
            raw = typer.prompt(
                "выбери номер или имя модели (Enter — выход)",
                default="",
                show_default=False,
                err=True,
            )
        except (EOFError, typer.Abort, KeyboardInterrupt):
            return None
        line = raw.strip()
        if line and not sys.stdin.isatty():
            echo_input(line)
        if not line:
            return None
        if line.isdigit() and 1 <= int(line) <= len(choices):
            chosen = choices[int(line) - 1]
            note(f"модель выбрана: {chosen}")
            return chosen
        if line in by_name:
            chosen = by_name[line]
            note(f"модель выбрана: {chosen}")
            return chosen
        warn(f"нет варианта {line!r} — введи номер из списка или точное имя")


def choose(prompt: str, options: list[str]) -> int | None:
    """Интерактивный выбор из нумерованного списка. None — отказ (Enter/EOF/Ctrl+C).

    Образец — choose_model() чуть выше: тот же вывод в stderr и тот же контракт
    «Enter = оставить как есть». Отдельная функция, а не обобщение choose_model:
    та знает про alias'ы и карточки моделей, и свести обе к одному коду можно
    было бы только опциональными параметрами ради одного вызова — преждевременное
    обобщение. Если появится третий пикер — тогда и смотреть.

    Возвращает индекс выбранного варианта: строка опции здесь — только подпись,
    а что делать с выбором (имя параметра, значение) решает вызывающий код.
    """
    if not options:
        return None
    err.print(prompt)
    for number, option in enumerate(options, 1):
        # markup=False: подписи приходят извне (значения параметров могут
        # содержать квадратные скобки — Rich съел бы их как незакрытый тег).
        err.print(f"  {number}. {option}", markup=False)
    # Точное имя принимаем без учёта регистра: пункты выбора — литеральные
    # значения параметров, и «Dialog» для mode=dialog — это не другой вариант.
    by_name = {option.lower(): index for index, option in enumerate(options)}

    while True:
        try:
            raw = typer.prompt(
                f"выбери номер или имя ({prompt}; Enter — выход)",
                default="",
                show_default=False,
                err=True,
            )
        except (EOFError, typer.Abort, KeyboardInterrupt):
            return None
        line = raw.strip()
        if line and not sys.stdin.isatty():
            echo_input(line)
        if not line:
            return None
        if line.isdigit() and 1 <= int(line) <= len(options):
            index = int(line) - 1
            note(f"выбрано: {options[index]}")
            return index
        if line.lower() in by_name:
            index = by_name[line.lower()]
            note(f"выбрано: {options[index]}")
            return index
        warn(f"нет варианта {line!r} — введи номер из списка или точное имя")


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
