"""Неделя 01 — команды `advent w01 ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from advent_core import chat as chat_core
from advent_core import console
from advent_core.client import (
    capabilities_of,
    chat_models,
    find_model,
    list_models,
    model_names,
    resolve_alias,
)
from advent_core.config import LOG_DIR, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.params import REASONING_EFFORTS, ParamError

WEEK = 1
HISTORY_PATH = LOG_DIR / "repl_history_w01.json"

REPL_COMMANDS = [
    ("/help", "показать этот список"),
    ("/model", "текущая модель"),
    ("/model <имя>", "переключить модель"),
    ("/model list", "таблица chat-моделей (list all — вообще все)"),
    ("/model info", "карточка текущей модели: контекст, возможности, t°"),
    ("/params", "текущие параметры генерации"),
    ("/set <параметр> <значение>", "изменить параметр (значение default — сбросить)"),
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

    if question:
        _ask_once(config, question)
    else:
        _repl(config)


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
        console.note(f"model={config.model} stream={config.stream} messages={len(messages)}")

    result = _run(session, messages)
    console.footer(result)
    log_call(result, messages, week=WEEK, day=1)


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

        history, dropped = chat_core.trim_history(history)
        if dropped:
            console.warn(f"история обрезана: выброшено {dropped} старых сообщений")

        messages = chat_core.build_messages(line, system=system, history=history)

        try:
            result = _run(session, messages)
        except AdventError as error:
            # В REPL ошибка не должна убивать сессию — печатаем и живём дальше.
            console.fail(error)
            log_call(
                chat_core.CallResult(model_requested=session.config.model),
                messages,
                week=WEEK,
                day=1,
                error=error.message,
            )
            continue

        console.footer(result)
        log_call(result, messages, week=WEEK, day=1)

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
        console.params_table(session.config.params.describe())
        return False

    if command == "/set":
        _handle_set(args, session)
        return False

    if command == "/model":
        _handle_model(args, session)
        return False

    console.warn(f"неизвестная команда {command}. /help — список")
    return False


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

    console.note(f"{name} = {'сброшен' if value is None else value}")

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


def _run(session: Session, messages: list[chat_core.Message]) -> chat_core.CallResult:
    config = session.config
    capabilities = session.capabilities

    if config.stream:
        result = chat_core.stream(config, messages, console.write_chunk, capabilities)
        console.finish_answer()
    else:
        result = chat_core.complete(config, messages, capabilities)
        console.write_chunk(result.text)
        console.finish_answer()

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
        HISTORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
