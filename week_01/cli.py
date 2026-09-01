"""Неделя 01 — команды `advent w01 ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.markup import escape as rich_escape

from advent_core import chat as chat_core
from advent_core import console, formats
from advent_core.client import (
    capabilities_of,
    chat_models,
    find_model,
    list_models,
    model_names,
    resolve_alias,
)
from advent_core.config import LOG_DIR, Config, ConfigError
from advent_core.errors import AdventError, ConfigurationError
from advent_core.journal import log_call
from advent_core.params import BY_NAME, FORMAT_CHOICES, MODE_CHOICES, REASONING_EFFORTS, ParamError

WEEK = 1
# Поднимать вместе с номером текущего дня — единственное место, которое это
# знает. Раньше log_call() звался с зашитым day=1 буквально везде, включая
# код дня 02 (_handle_again, _run_dialog): `advent record --day 2` писал в
# logs/calls.jsonl день 1 (НАХОДКА 4 code review). Протаскивать day через всю
# цепочку вызовов ради одной константы — избыточно; альтернатива проще и
# заведомо не разъедется, пока не забыть её поднять.
DAY = 2
HISTORY_PATH = LOG_DIR / "repl_history_w01.json"
DIALOG_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog.md"

REPL_COMMANDS = [
    ("/help", "показать этот список"),
    ("/model", "текущая модель"),
    ("/model <имя>", "переключить модель"),
    ("/model list", "таблица chat-моделей (list all — вообще все)"),
    ("/model info", "карточка текущей модели: контекст, возможности, t°"),
    ("/params", "текущие параметры генерации (и итоговый system prompt)"),
    (
        "/set <параметр> <значение>",
        "изменить параметр, включая format/schema_file/done/mode/max_turns (default — сбросить)",
    ),
    ("/again", "повторить последний вопрос с текущими настройками, без истории"),
    ("/reset", "очистить историю диалога"),
    ("/exit", "выход"),
]

app = typer.Typer(help="Неделя 01: первый запрос к LLM через API.", no_args_is_help=True)


@app.command("chat")
def chat_command(
    question: str | None = typer.Argument(None, help="Вопрос. Без аргумента открывается REPL."),
    model: str | None = typer.Option(None, "--model", "-m", help="Имя модели Mistral."),
    system: Path | None = typer.Option(None, "--system", help="Файл с system prompt."),
    temperature: float | None = typer.Option(None, "--temperature", "-t", help="0..2."),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling, 0..1."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Потолок длины ответа."),
    seed: int | None = typer.Option(None, "--seed", help="Seed для воспроизводимости."),
    stop: str | None = typer.Option(None, "--stop", help="Стоп-строки через запятую."),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=f"Глубина рассуждения: {', '.join(REASONING_EFFORTS)}. Нужна capability reasoning.",
    ),
    format_: str | None = typer.Option(
        None,
        "--format",
        help=f"Пресет формата ответа: {', '.join(FORMAT_CHOICES)}.",
    ),
    schema_file: str | None = typer.Option(
        None, "--schema-file", help="Путь к .json со схемой ответа — для --format schema."
    ),
    done: str | None = typer.Option(
        None,
        "--done",
        help='Условие завершения диалога: "text:<строка>" или "json:<поле>".',
    ),
    mode: str | None = typer.Option(None, "--mode", help=f"Режим REPL: {', '.join(MODE_CHOICES)}."),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help="Потолок ходов в --mode dialog (по умолчанию 10)."
    ),
    no_stream: bool = typer.Option(False, "--no-stream", help="Получить ответ одним куском."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запроса в stderr."),
) -> None:
    """Задать вопрос модели: один ответ или интерактивный диалог."""
    config = Config.resolve(
        model=model,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        random_seed=seed,
        stop=stop,
        reasoning_effort=reasoning_effort,
        stream=not no_stream,
        verbose=verbose,
    )
    # Config.resolve() не знает про пять локальных параметров дня 02 (вне моей
    # зоны — не трогаю config.py), поэтому кладём их той же машинерией, что и
    # /set: GenerationParams.set() с той же валидацией и теми же ParamError.
    _apply_local_flags(
        config,
        format=format_,
        schema_file=schema_file,
        done=done,
        mode=mode,
        max_turns=max_turns,
    )
    _guard_one_shot_dialog(question, config)

    if question:
        _ask_once(config, question)
    else:
        _repl(config)


def _guard_one_shot_dialog(question: str | None, config: Config) -> None:
    """--mode dialog вместе с позиционным вопросом раньше молча одношотился.

    `_repl()`/`_run_dialog()` — единственное место, которое читает mode/done,
    а одношотовый `_ask_once()` про них ничего не знает. В результате
    `advent w01 chat "..." --mode dialog` тихо отвечал обычным одиночным
    ответом, как будто флага не было (НАХОДКА 2 code review). Диалог по
    природе многоходовый — молча игнорировать флаг нельзя, поэтому падаем
    понятной ошибкой конфигурации вместо этого. ConfigurationError — та же
    машинерия и тот же exit_code=2, что и у остальных ошибок конфигурации
    (advent_core/errors.py).
    """
    if question is not None and config.params.mode == "dialog":
        raise ConfigurationError(
            "--mode dialog работает только в интерактивном режиме — "
            "убери позиционный вопрос и открой REPL: advent w01 chat"
        )


def _apply_local_flags(
    config: Config,
    *,
    format: str | None,
    schema_file: str | None,
    done: str | None,
    mode: str | None,
    max_turns: int | None,
) -> None:
    """Кладёт флаги --format/--schema-file/--done/--mode/--max-turns в config.params.

    Одна и та же валидация, что у `/set`: GenerationParams.set() проверяет
    форму значения (choices, непустая строка, …), а `_check_local_param()`
    следом проверяет смысл (файл схемы существует, префикс `done` понятен) —
    на старте CLI ошибка должна остановить программу, а не всплыть только на
    первом запросе к модели.
    """
    given = (
        ("format", format),
        ("schema_file", schema_file),
        ("done", done),
        ("mode", mode),
        ("max_turns", max_turns),
    )
    try:
        for name, raw in given:
            if raw is None:
                continue
            config.params.set(name, raw)
            _check_local_param(name, raw)
    except ParamError as exc:
        raise ConfigError(str(exc)) from exc


def _check_local_param(name: str, value: str) -> None:
    """Ранняя проверка смысла (не только формы) для schema_file/done.

    Spec в params.py валидирует только форму значения (см. её комментарий);
    смысл — существование и валидность файла схемы, распознаваемый префикс
    done — знают formats.load_schema()/formats.parse_done(). Здесь их зовут
    ради самой дешёвой точки сообщить об ошибке: сразу, а не на первом
    запросе к модели или посреди демо.
    """
    if name == "schema_file":
        formats.load_schema(value)
    elif name == "done":
        formats.parse_done(value)


@app.command("models")
def models_command(
    model: str | None = typer.Option(None, "--model", "-m", help="Подсветить эту модель."),
    all_models: bool = typer.Option(False, "--all", help="Показать и не-chat модели."),
) -> None:
    """Список моделей аккаунта прямо из API."""
    config = Config.resolve(model=model)
    models = list_models(config)
    shown = models if all_models else chat_models(models)

    console.models_table(shown, highlight=config.model)

    if actual := resolve_alias(models, config.model):
        console.note(f"{config.model} сейчас разрешается в {actual}")
    console.note(f"показано {len(shown)} из {len(models)}; t° — рекомендованная моделью")


class Session:
    """Состояние одного запуска чата: модель, её метаданные и параметры.

    Держим карточку модели рядом с конфигом: она нужна и для отсева
    параметров по capabilities, и для `/model info`, а тянуть список моделей
    на каждый запрос — лишний round trip.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.models: list[dict] = []
        self.card: dict | None = None
        # Последний вопрос обычного (не dialog) режима — только для /again.
        # None, пока в этой сессии не было ни одного вопроса.
        self.last_question: str | None = None
        self.refresh()

    def refresh(self) -> None:
        """Перечитывает список моделей и проверяет, что текущая существует."""
        try:
            self.models = list_models(self.config)
        except AdventError:
            # Список — удобство, а не обязательное условие: ошибка всплывёт
            # на самом запросе и будет переведена в человеческий текст.
            self.models = []
            self.card = None
            return

        names = model_names(self.models)
        if names and self.config.model not in names:
            raise ConfigError(
                f"Модель {self.config.model!r} недоступна аккаунту.\n"
                f"Посмотри список: advent w01 models"
            )
        self.card = find_model(self.models, self.config.model)

    @property
    def capabilities(self) -> dict | None:
        return capabilities_of(self.models, self.config.model) if self.models else None

    @property
    def resolved(self) -> str | None:
        return resolve_alias(self.models, self.config.model) if self.models else None


def _ask_once(config: Config, question: str) -> None:
    """Один вопрос — один ответ, без истории."""
    session = Session(config)
    messages = chat_core.build_messages(question, system=config.system_prompt())

    if config.verbose:
        # should_stream() — единая точка решения (см. _run), не сырой
        # config.stream: format=json/schema всегда без стрима, и verbose
        # печатал stream=True в этом случае, хотя реально ушла ветка без
        # стрима (НАХОДКА 6 code review).
        console.note(
            f"model={config.model} stream={chat_core.should_stream(config)} "
            f"messages={len(messages)}"
        )

    result = _run(session, messages)
    console.footer(result)
    # result.sent_messages — то, что реально дописала chat._payload() поверх
    # messages (инструкция пресета формата); fallback на messages нужен
    # только теоретически (sent_messages=None означало бы, что _run() упал до
    # _payload(), а тогда мы бы сюда не дошли) — журнал не должен разъезжаться
    # с тем, что реально ушло в API (НАХОДКА 5 code review).
    log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)


def _repl(config: Config) -> None:
    """Интерактивный диалог с историей, командами и обрезкой контекста."""
    session = Session(config)
    system = config.system_prompt()
    history = _load_history()

    console.note("REPL. /help — список команд. Ctrl+C прерывает генерацию.")
    if history:
        console.note(f"подхвачена история прошлого запуска: {len(history)} сообщений")

    while True:
        try:
            line = typer.prompt("\nты", prompt_suffix=" › ", err=True).strip()
        except (EOFError, typer.Abort):
            console.note("пока")
            break

        # При запуске из скрипта записи stdin — это pipe, и терминал не
        # отображает то, что в него подали. Без эха в кадре видно приглашение
        # без вопроса.
        if line and not sys.stdin.isatty():
            console.echo_input(line)

        if not line:
            continue
        if line.startswith("/"):
            if _handle_command(line, session, history):
                break
            continue

        # mode=dialog уводит обычную реплику в отдельный цикл «вопрос за
        # вопросом» (см. _run_dialog) — история обычного REPL здесь не
        # трогается: это независимый разговор со своим system-пресетом.
        #
        # Проверка done — здесь, а не только внутри _run_dialog: раньше
        # _run_dialog() при отсутствии done печатала warning и делала return
        # ДО обращения к API, без записи в session.last_question и без
        # журнала — введённая строка просто исчезала (НАХОДКА 3 code
        # review). Если done не настроен, диалог технически не может
        # работать (не с чем сравнивать условие завершения) — реплика падает
        # в обычную одноходовую обработку ниже, а не теряется молча.
        if session.config.params.mode == "dialog":
            if config.params.done:
                pending_command = _run_dialog(session, line)
                # _run_dialog() возвращает отложенную команду (строку с "/"),
                # только если диалог прервался именно на слэш-команде — не на
                # завершении по done, не на max_turns, не на EOF (НАХОДКА 7
                # code review). Она ещё не была обработана нигде, поэтому
                # исполняем её здесь же, как будто пользователь только что её
                # ввёл — иначе "/exit" в конце demo-шага 5 молча терялся бы.
                if pending_command is not None and _handle_command(
                    pending_command, session, history
                ):
                    break
                continue
            console.warn(
                "mode=dialog требует done — /set done text:<строка> или "
                "json:<поле>; реплика обработана как обычный вопрос"
            )

        session.last_question = line
        history, dropped = chat_core.trim_history(history)
        if dropped:
            console.warn(f"история обрезана: выброшено {dropped} старых сообщений")

        messages = chat_core.build_messages(line, system=system, history=history)

        try:
            result = _run(session, messages)
        except AdventError as error:
            # В REPL ошибка не должна убивать сессию — печатаем и живём дальше.
            console.fail(error)
            # Ошибка случилась до/внутри _payload() — result.sent_messages
            # тут нет (ошибка сконструирована вручную), messages и так то,
            # что реально пытались отправить.
            log_call(
                chat_core.CallResult(model_requested=session.config.model),
                messages,
                week=WEEK,
                day=DAY,
                error=error.message,
            )
            continue

        console.footer(result)
        log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)

        history.append({"role": "user", "content": line})
        if result.text:
            history.append({"role": "assistant", "content": result.text})
        _save_history(history)


def _handle_command(line: str, session: Session, history: list[chat_core.Message]) -> bool:
    """Обрабатывает команду REPL. True означает «выходим»."""
    parts = line.split()
    command, args = parts[0], parts[1:]

    if command in ("/exit", "/quit"):
        console.note("пока")
        return True

    if command == "/help":
        console.commands_help(REPL_COMMANDS)
        return False

    if command == "/reset":
        history.clear()
        _save_history(history)
        console.note("история очищена")
        return False

    if command == "/params":
        _show_params(session)
        return False

    if command == "/set":
        _handle_set(args, session)
        return False

    if command == "/model":
        _handle_model(args, session)
        return False

    if command == "/again":
        _handle_again(session)
        return False

    console.warn(f"неизвестная команда {command}. /help — список")
    return False


def _show_params(session: Session) -> None:
    """`/params`: таблица параметров (локальные помечены) + итоговый system prompt."""
    rows = []
    for name, value, help_text in session.config.params.describe():
        if BY_NAME[name].local:
            # Скобки — круглые, не квадратные: console.params_table() кладёт
            # эту строку в ячейку Rich Table, а Rich-markup молча съедает
            # квадратные скобки как незакрытый тег (проверено вручную).
            help_text = f"{help_text} (local — не уходит на сервер)"
        rows.append((name, value, help_text))
    console.params_table(rows)

    system = _final_system_prompt(session)
    if system:
        # rich_escape: system prompt содержит инструкцию формата и, при
        # format=schema, дамп JSON Schema — квадратные скобки массивов иначе
        # console.note() (Rich-markup) молча съедает как незакрытый тег, и
        # часть текста пропадает с экрана без единой ошибки.
        console.note(f"итоговый system prompt:\n{rich_escape(system)}")
    else:
        console.note("system prompt не задан")


def _final_system_prompt(session: Session) -> str | None:
    """system пользователя + дописанная инструкция пресета формата/диалога.

    Раньше ветка mode=dialog делала ранний return сразу после
    _dialog_system_prompt() и НИКОГДА не доходила до formats.build_system()
    — а реально уходящий запрос (chat._payload()) безусловно дописывает слой
    формата поверх ЛЮБОГО system, dialog-пресет не исключение. /params врал:
    показывал dialog-промпт без инструкции формата, хотя она реально уйдёт
    (НАХОДКА 1 code review). Теперь обе ветки собирают base_system и в конце
    ОБЕ проходят через formats.build_system() — ту же функцию, что и
    chat._payload()/_with_format_instruction() зовут перед отправкой, так что
    правило дописывания формата живёт в одном месте и не может разъехаться
    между /params и реальным запросом.
    """
    config = session.config
    base_system = config.system_prompt()

    if config.params.mode == "dialog" and config.params.done:
        try:
            kind, needle = formats.parse_done(config.params.done)
        except ConfigError:
            pass  # done ещё не поправлен — покажем обычный (не dialog) system
        else:
            # _dialog_system_prompt() уже сама берёт config.system_prompt() и
            # дописывает пресет диалога — дальше форматный слой ложится
            # поверх результата, как и на запросе.
            base_system = _dialog_system_prompt(config, kind, needle)

    format_name = config.params.format or "text"
    schema = None
    if format_name == "schema" and config.params.schema_file:
        try:
            schema = formats.load_schema(config.params.schema_file)
        except ConfigError:
            schema = None  # покажем то, что есть — ошибка всплывёт на запросе
    return formats.build_system(format_name, base_system, schema)


def _handle_set(args: list[str], session: Session) -> None:
    if len(args) < 2:
        console.warn("нужно: /set <параметр> <значение>. /params — что есть")
        return

    name, raw = args[0], " ".join(args[1:])
    try:
        value = session.config.params.set(name, raw)
    except ParamError as error:
        console.warn(str(error))
        return

    # rich_escape: значения вроде done=text:[ГОТОВО] (пример из спеки дня)
    # содержат квадратные скобки — без экранирования console.note() (Rich
    # markup) тихо съедает их как незакрытый тег, и маркер пропадает с экрана.
    if value is None:
        shown = "сброшен"
    elif isinstance(value, str):
        shown = rich_escape(value)
    else:
        shown = value
    console.note(f"{name} = {shown}")

    if value is not None:
        # Смысловая проверка (файл схемы существует, префикс done понятен) —
        # сразу, а не молча на первом запросе к модели. Параметр остаётся
        # выставленным даже при ошибке: тот же принцип, что и с capabilities
        # ниже — пользователь видит предупреждение и правит следующей командой.
        try:
            _check_local_param(name, value)
        except ConfigError as error:
            console.warn(str(error))

    # Предупреждаем сразу, а не молча на запросе: иначе непонятно, почему
    # выставленный параметр ни на что не влияет.
    capabilities = session.capabilities
    _, skipped = session.config.params.as_payload(capabilities)
    if name in skipped:
        console.warn(f"{session.config.model} не поддерживает {name} — параметр не отправляется")


def _handle_model(args: list[str], session: Session) -> None:
    if not args:
        console.note(f"текущая модель: {session.config.model}")
        return

    sub = args[0]

    if sub == "list":
        if not session.models:
            console.warn("список моделей недоступен — не отвечает API")
            return
        show_all = len(args) > 1 and args[1] == "all"
        shown = session.models if show_all else chat_models(session.models)
        console.models_table(shown, highlight=session.config.model)
        console.note(f"показано {len(shown)} из {len(session.models)}")
        if not show_all:
            console.note("/model list all — включая embedding, OCR и аудио")
        return

    if sub == "info":
        if session.card is None:
            console.warn("карточка модели недоступна — не отвечает API")
            return
        console.model_card(session.card, session.config.model)
        return

    previous = session.config.model
    session.config.model = sub
    try:
        session.refresh()
    except (ConfigError, AdventError) as error:
        # Опечатка в имени модели не должна ронять сессию: чинится следующей
        # же командой.
        session.config.model = previous
        session.refresh()
        console.warn(str(getattr(error, "message", error)))
        return

    resolved = session.resolved
    console.note(f"модель переключена на {sub}" + (f" → {resolved}" if resolved else ""))

    _, skipped = session.config.params.as_payload(session.capabilities)
    if skipped:
        console.warn(f"{sub} не поддерживает: {', '.join(skipped)} — параметры не отправляются")


def _handle_again(session: Session) -> None:
    """`/again`: тот же последний вопрос, текущие настройки, БЕЗ истории.

    Это и есть суть задания дня («один и тот же запрос») и защита от подмены:
    при обычном повторе через историю модель видит собственный прошлый ответ
    и отвечает «как я говорил выше» вместо нового независимого прогона —
    тогда сравнение стабильности format=schema теряет смысл.
    """
    if session.config.params.mode == "dialog":
        console.warn("/again недоступна в mode=dialog — там история и есть суть диалога")
        return
    if session.last_question is None:
        console.warn("нечего повторять — сначала задай вопрос")
        return

    messages = chat_core.build_messages(
        session.last_question, system=session.config.system_prompt()
    )
    try:
        result = _run(session, messages)
    except AdventError as error:
        console.fail(error)
        log_call(
            chat_core.CallResult(model_requested=session.config.model),
            messages,
            week=WEEK,
            day=DAY,
            error=error.message,
        )
        return

    console.footer(result)
    log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)
    # Сознательно не трогаем историю REPL: /again — независимый прогон того
    # же запроса, а не ещё один ход обычного диалога.


def _dialog_marker_instruction(kind: str, needle: str) -> str:
    """Инструкция про признак завершения — вставляется в пресет dialog.md.

    Формат признака задаётся опцией `done`, а не зашит: text — подстрока в
    свободном тексте, json — булево поле в структурированном ответе.
    """
    if kind == "text":
        return (
            f'Когда результат готов, включи в ответ строку ровно так: "{needle}" — '
            "это и есть признак завершения. Пока диалог не закончен, эту строку "
            "не используй."
        )
    # parse_done() уже проверил kind — тут он либо "text", либо "json".
    return (
        "Отвечай только в формате JSON, без пояснений и без обёртки в ```. Пока "
        f'идёт уточнение — верни объект вида {{"{needle}": false, "question": '
        f'"..."}}. Когда результат готов — верни {{"{needle}": true, "result": '
        '{...}} с итоговыми данными вместо "...".'
    )


def _dialog_system_prompt(config: Config, kind: str, needle: str) -> str:
    """system для mode=dialog: пользовательский + пресет из prompts/dialog.md.

    Пресет — шаблон с плейсхолдером %%MARKER%%, наполняемым конкретным
    условием завершения (см. _dialog_marker_instruction). Дальше слой
    формата (format=json/schema/…) дописывается уже внутри chat_core поверх
    этого system — ровно тот же порядок наслоения, что и в обычном чате
    (SPEC-w01d02.md §4).
    """
    template = DIALOG_PROMPT_PATH.read_text(encoding="utf-8").strip()
    preset = template.replace("%%MARKER%%", _dialog_marker_instruction(kind, needle))
    parts = [part for part in (config.system_prompt(), preset) if part]
    return "\n\n".join(parts)


def _run_dialog(session: Session, task: str) -> str | None:
    """Цикл mode=dialog: по одному уточняющему вопросу за ход, до готовности.

    Отдельная история от обычного REPL — независимый разговор со своим
    system-пресетом (SPEC-w01d02.md §5). Маркер завершения из вывода не
    вырезается: он виден на экране, это и есть предмет демонстрации.

    Возвращает отложенную команду REPL (строку с ведущим "/"), если диалог
    закончился из-за того, что пользователь ввёл команду вместо реплики —
    иначе None. Раньше ЛЮБАЯ строка, прочитанная внутри цикла (включая
    "/exit"), безусловно уходила в модель как очередная реплика: слэш-команды
    не распознавались вообще (НАХОДКА 7 code review, воспроизведена живым
    прогоном demo-шага 5). Если диалог не сходится за отведённые ходы модели
    — а он по природе не гарантированно сходится, ходов сколько потребуется
    — "/exit", которым шаг 5 в advent_cli/record.py завершает демо, был бы
    отправлен в API как бессмысленный ответ про салат, а сама команда выхода
    потеряна: сессия осталась бы висеть в диалоге, дубль сорван перед
    камерой. Возврат строки наверх — единственный вариант, не теряющий
    команду молча и не глотающий её как реплику: _run_dialog не знает, как
    исполняются команды REPL (history, /model, /again — состояние в _repl),
    поэтому решение отдаётся вызывающему коду.
    """
    config = session.config
    if not config.params.done:
        # Страховка на случай прямого вызова в обход _repl(): штатный путь
        # (_repl) сам проверяет done ДО вызова _run_dialog() и в его
        # отсутствие обрабатывает реплику как обычный вопрос вместо того,
        # чтобы звать эту функцию (НАХОДКА 3 code review — раньше именно
        # здесь введённая строка молча терялась). Сюда мы дойти не должны.
        console.warn("mode=dialog требует done — /set done text:<строка> или json:<поле>")
        return None
    try:
        kind, needle = formats.parse_done(config.params.done)
    except ConfigError as error:
        console.warn(str(error))
        return None

    system = _dialog_system_prompt(config, kind, needle)
    # None здесь означает «сброшено через /set max_turns default» — по
    # контракту params.py это трактуется как дефолт 10, а не «без потолка»:
    # без потолка demo-шаг advent record завис бы до таймаута в 180 секунд
    # на первом же вопросе модели, которого нет в фиксированном списке строк.
    max_turns = config.params.max_turns or 10
    dialog_history: list[chat_core.Message] = []
    line = task
    turns = 0

    while True:
        turns += 1
        messages = chat_core.build_messages(line, system=system, history=dialog_history)
        try:
            result = _run(session, messages)
        except AdventError as error:
            console.fail(error)
            log_call(
                chat_core.CallResult(model_requested=config.model),
                messages,
                week=WEEK,
                day=DAY,
                error=error.message,
            )
            return None

        console.footer(result)
        log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)

        dialog_history.append({"role": "user", "content": line})
        if result.text:
            dialog_history.append({"role": "assistant", "content": result.text})

        if formats.is_done(result.text, kind, needle):
            console.note(f"диалог: условие завершения выполнено, ходов: {turns}")
            return None

        if turns >= max_turns:
            # НЕ ошибка и НЕ выход из сессии — просто возврат в обычный REPL,
            # диалог остаётся в логе как незавершённый.
            console.warn(
                f"диалог: достигнут потолок ходов ({max_turns}), условие не "
                "выполнено — возврат в обычный REPL"
            )
            return None

        try:
            line = typer.prompt("\nдиалог", prompt_suffix=" › ", err=True).strip()
        except (EOFError, typer.Abort):
            console.note("диалог прерван")
            return None

        if line and not sys.stdin.isatty():
            console.echo_input(line)
        if not line:
            console.note("пустой ввод — диалог прерван, возврат в обычный REPL")
            return None

        if line.startswith("/"):
            # Слэш-команда — не реплика диалога. "/exit" здесь раньше уходил
            # в API как ответ про салат (НАХОДКА 7): демо-шаг 5
            # (advent_cli/record.py) заканчивается ровно "/exit" внутри
            # диалога, и если модель не сошлась за отведённые ходы (она не
            # обязана сходиться быстро — это не гарантия, а наблюдение),
            # выход срывался перед камерой. Прерываем диалог и отдаём строку
            # наверх — _repl() исполнит её как только что введённую команду,
            # ничего не потеряется и ничего не уйдёт в модель.
            console.note("диалог: получена команда — выходим в обычный REPL")
            return line


def _run(session: Session, messages: list[chat_core.Message]) -> chat_core.CallResult:
    config = session.config
    capabilities = session.capabilities

    # should_stream() — единая точка решения: format=json/schema всегда без
    # стрима (вердикт и поиск маркера завершения возможны только по целому
    # ответу), иначе — как задано --no-stream/config.stream.
    if chat_core.should_stream(config):
        result = chat_core.stream(config, messages, console.write_chunk, capabilities)
        console.finish_answer()
    else:
        result = chat_core.complete(config, messages, capabilities)
        # print_answer сама решает: json/schema с валидным ответом — pretty
        # print с отступами, иначе — сырой текст как есть.
        console.print_answer(result, config.params.format or "text")

    # API возвращает то же имя, что мы прислали, поэтому конкретную версию
    # подставляем из списка моделей.
    if resolved := session.resolved:
        result.model_actual = resolved
    if result.skipped_params:
        console.warn(
            f"{config.model} не поддерживает: {', '.join(result.skipped_params)} — "
            "параметры не отправлены"
        )
    return result


def _load_history() -> list[chat_core.Message]:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, dict) and m.get("role") and m.get("content")]


def _save_history(history: list[chat_core.Message]) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
