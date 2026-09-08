"""Агент недели 02 — отдельная точка входа `adventagent`.

Почему не подкоманда `advent w02` (SPEC-w02d06.md §3): неделя 03
(«Оптимизация агента») продолжает ТОГО ЖЕ агента, и привязка к номеру недели
заставила бы week_03 либо импортировать чужую неделю, либо форкнуть код.

Позиционного вопроса здесь нет сознательно: `adventagent "вопрос"` — это ровно
та «обёртка над одним вызовом API», от которой уводит задание дня. Агент
настраивается (сессия, модель, параметры, режим) и запускается как разговор;
поверхность для скриптового однократного вызова остаётся у недели 01
(`advent w01 chat "…" > answer.txt`).

Слой ответственности здесь ровно один — ввод-вывод: сборка запроса, обрезка
контекста и детект завершения живут в advent_core/agent.py, память — в
advent_core/session.py, счёт токенов — в advent_core/tokens.py. Печатает
только этот модуль (SPEC §5, §14).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console, formats, tokens
from advent_core.agent import Agent, AgentReply
from advent_core.client import (
    capabilities_of,
    chat_models,
    find_model,
    list_models,
    model_names,
    resolve_alias,
)
from advent_core.config import DEFAULT_SYSTEM_PROMPT, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.params import (
    AGENT_COMMAND,
    AGENT_PARAMS,
    BY_NAME,
    FORMAT_CHOICES,
    MODE_CHOICES,
    REASONING_EFFORTS,
    ParamError,
    defaults_for,
)
from advent_core.session import DEFAULT_SESSION, Session, Turn, list_sessions, validate_name
from advent_core.telemetry import CallResult
from advent_core.tokens import TokenCheck, counter_for

WEEK = 2
# Номер дня, который последним трогал ЭТОТ код. Тест сверяет литерал 6, а не
# эту константу: ожидание, взятое из того же источника, что и код под тестом,
# покраснеть не может (CLAUDE.md, «A test whose expected value comes from the
# same source as the code under test»).
DAY = 6

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# Персона агента — своя, а не общекурсовая «лаконичный ассистент, без воды»:
# та персона гасит встречные вопросы, а агент должен уметь их задавать
# (advent_core/prompts/default_system.md против week_02/prompts/agent.md).
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent.md"
# Копия пресета недели 01, а не импорт из неё: у недели своя копия, и правка
# диалога агента не должна менять поведение уже сданного дня 02.
DIALOG_PROMPT_PATH = PROMPTS_DIR / "dialog.md"

# Многострочный ввод — sentinel, а не Shift+Enter: портируемого способа поймать
# Shift+Enter в терминалах не существует, документация aider говорит это прямо
# (SPEC-w02d06.md §13). Тройная кавычка взята у shell_gpt.
MULTILINE_SENTINEL = '"""'


# --------------------------------------------------------------------------
# Реестр слэш-команд
#
# Реестр, а не цепочка if/elif: в неделе 01 _handle_command() — цепочка на 130
# строк, а у агента команд больше (SPEC-w02d06.md §10). Из всей разведки
# расширяемая система команд есть только у gptme — декоратор плюс алиасы,
# ровно этот приём.
# --------------------------------------------------------------------------

# Обработчик получает состояние сессии и аргументы команды, возвращает True,
# если после неё надо выйти. Тип объявлен строкой: AgentShell определён ниже,
# а держать реестр рядом с командами важнее, чем угодить порядку определений.
CommandFn = Callable[["AgentShell", list[str]], bool]


@dataclass(slots=True, frozen=True)
class Command:
    """Одна слэш-команда: как зовётся, что делает, чем её описать в /help."""

    name: str
    help: str
    handler: CommandFn
    usage: str = ""
    aliases: tuple[str, ...] = ()


# Порядок вставки = порядок в /help, поэтому dict, а не отсортированный набор:
# список команд читается на видео сверху вниз, и алфавит там ни при чём.
COMMANDS: dict[str, Command] = {}


def command(
    name: str, help_text: str, *aliases: str, usage: str = ""
) -> Callable[[CommandFn], CommandFn]:
    """Регистрирует обработчик слэш-команды под именем и алиасами."""

    def register(handler: CommandFn) -> CommandFn:
        entry = Command(
            name=name, help=help_text, handler=handler, usage=usage or name, aliases=aliases
        )
        for key in (name, *aliases):
            COMMANDS[key] = entry
        return handler

    return register


def unique_commands() -> list[Command]:
    """Команды по одной записи на обработчик — алиасы не дублируются в /help."""
    seen: dict[int, Command] = {}
    for entry in COMMANDS.values():
        seen.setdefault(id(entry), entry)
    return list(seen.values())


# --------------------------------------------------------------------------
# Состояние запуска
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AgentShell:
    """Всё, что живёт один запуск: конфиг, модель, память, агент, счётчик.

    Отдельно от Agent намеренно: агент ничего не печатает и не знает про диск,
    а здесь собрано ровно то, что нужно интерфейсу — карточка модели для
    `/model info`, рабочая история для следующего хода, счётчик ходов целевого
    диалога и последняя сверка счёта токенов для `/tokens`.
    """

    config: Config
    # Каталог сессий. None — штатный logs/sessions; тесты подставляют свой,
    # чтобы прогон suite'а не писал в настоящую память агента.
    directory: Path | None = None

    models: list[dict] = field(default_factory=list)
    card: dict | None = None
    session: Session = field(init=False)
    history: list[chat_core.Message] = field(default_factory=list)
    agent: Agent = field(init=False)
    counter: tokens.TokenCounter | None = None
    # Последний вопрос обычного хода — только для /again.
    last_question: str | None = None
    # Последняя сверка локального счёта с фактом сервера — показывает /tokens.
    last_check: TokenCheck | None = None
    # Ходов целевого диалога с момента его начала. Считает CLI, а не агент:
    # диалог идёт в ОБЩЕЙ памяти сессии (SPEC-w02d06.md §9), поэтому вывести
    # число ходов из длины истории нельзя — там лежит и всё, что обсуждали до.
    dialog_turns: int = 0

    def __post_init__(self) -> None:
        self.refresh()
        self.session = self.open_session(self.config.params.session or DEFAULT_SESSION)
        self.history = self.session.history()
        # on_notice: скачка токенизатора идёт минуты, и молчащий процесс между
        # строкой про сессию и приглашением читается как зависший — в том
        # числе машинерией записи демо, которая снимает шаг по таймауту и
        # называет причину неверно.
        self.counter, warning = counter_for(self.config.model, on_notice=console.note)
        if warning:
            console.warn(warning)
        self.agent = Agent(
            self.config,
            # Явно, а не дефолтом аргумента: дефолт связывается на импорте, и
            # monkeypatch.setattr(chat_core, "complete", …) до него не
            # дотягивается — ловушка, оплаченная днём 03 (CLAUDE.md).
            complete=chat_core.complete,
            stream=chat_core.stream,
            counter=self.counter,
            capabilities=self.capabilities,
            context_limit=tokens.context_limit(self.card),
            persona=_persona(self.config),
            dialog_preset=_dialog_preset(),
            # Агент не печатает — он отдаёт текст сюда (SPEC §5).
            on_warning=console.warn,
        )
        _warn_text(self.agent.check_done())

    # --- модель ------------------------------------------------------------

    def refresh(self) -> None:
        """Перечитывает список моделей и проверяет, что текущая существует."""
        try:
            self.models = list_models(self.config)
        except AdventError:
            # Список — удобство (отсев параметров, окно контекста, /model
            # info), а не условие работы: ошибка всплывёт на самом запросе и
            # будет переведена в человеческий текст.
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

    def retarget(self) -> None:
        """Перенастраивает всё, что зависит от имени модели.

        Смена модели внутри сессии меняет не только адресата запроса: у другой
        модели другое окно контекста, другие capabilities и другой токенизатор.
        Оставить их от прежней модели значило бы показывать заполненность окна
        и счёт токенов, посчитанные не для той модели, — ровно тот структурный
        риск, на котором стоит goose #6185 (SPEC-w02d06.md §11).
        """
        self.refresh()
        self.counter, warning = counter_for(self.config.model, on_notice=console.note)
        if warning:
            console.warn(warning)
        self.agent.counter = self.counter
        self.agent.capabilities = self.capabilities
        self.agent.context_limit = tokens.context_limit(self.card)
        self.warn_model_drift()

    # --- сессия ------------------------------------------------------------

    def open_session(self, name: str) -> Session:
        """Загружает сессию с диска и говорит вслух обо всём, что с ней не так."""
        session = Session.load(name, directory=self.directory)
        for warning in session.warnings:
            console.warn(warning)
        console.note(f"сессия {session.name}: ходов {len(session.turns)}, начата {session.created}")
        self._warn_drift(session)
        return session

    def warn_model_drift(self) -> None:
        self._warn_drift(self.session)

    def _warn_drift(self, session: Session) -> None:
        """Предупреждение о смене модели относительно последнего сохранённого хода.

        Запрета нет — есть предупреждение (SPEC-w02d06.md §11): контекст
        остаётся валидным, но окно и счёт токенов теперь другие. У simonw/llm
        подмена модели при продолжении сессии происходит молча, и issue #1140
        открыт годами — это ровно тот пробел, который здесь закрыт.
        """
        last = session.last_model()
        if last and last != self.config.model:
            console.warn(
                f"последний ход сессии {session.name} сделан моделью {last}, сейчас "
                f"{self.config.model} — контекст остаётся валидным, но окно "
                "контекста и счёт токенов теперь другие"
            )

    def save(self) -> None:
        """Сохраняет сессию; ошибку записи показывает, а не глотает.

        Session.save() специально не глотает OSError (в отличие от
        journal.log_call, где лог — побочный эффект): потерянная сессия это
        потерянный разговор. Ловить обязана CLI — здесь.
        """
        try:
            self.session.save()
        except OSError as error:
            console.warn(f"сессия не сохранена ({error}) — разговор остаётся только на экране")


def _persona(config: Config) -> str | None:
    """Персона агента, либо None — «взять system prompt проекта».

    `--system` (и ADVENT_SYSTEM_PROMPT) перекрывает персону: явный выбор
    пользователя не наш, чтобы его игнорировать. Ровно то же правило действует
    в week_01/strategies.build_strategy_system() — там оно спасает измерение,
    здесь сохраняет управляемость.
    """
    if config.system_prompt_path != DEFAULT_SYSTEM_PROMPT:
        return None
    try:
        # Не `or None` на пустом файле: None означает не «без персоны», а
        # «взять персону проекта» — ту самую «лаконичный ассистент, без воды»,
        # которая гасит встречные вопросы агента. Пустая строка — осознанный
        # отказ от персоны, и Agent.system_prompt() отличает её от None.
        return AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as error:
        console.warn(
            f"персона агента не прочиталась ({error}) — идём вовсе без system prompt; "
            "персона проекта не подставляется, она гасит встречные вопросы"
        )
        return ""


def _dialog_preset() -> str | None:
    try:
        return DIALOG_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        # Не падаем: диалог без пресета работает на одной инструкции про
        # маркер (agent.system_prompt() сам скажет об этом вслух).
        console.warn(f"пресет диалога не прочитался ({error})")
        return None


def _warn_text(text: str | None) -> None:
    if text:
        console.warn(text)


# --------------------------------------------------------------------------
# Ход разговора
# --------------------------------------------------------------------------


def _turn(shell: AgentShell, question: str) -> None:
    """Один ход: спросить, напечатать, посчитать, запомнить."""
    shell.last_question = question
    reply = _ask(shell, question, shell.history)
    if reply is None:
        return

    # reply.history — УЖЕ обрезанная история, и именно её надо отдать в
    # следующий ask(): полная запись живёт в файле сессии, а кормить агента
    # session.history() каждый ход значило бы пересчитывать обрезку заново.
    shell.history = reply.history
    _remember(shell, question, reply)
    # Панель — ПОСЛЕ обновления истории и записи хода, иначе оба её числа
    # отстают на ход: «сессия» не считала бы только что полученный usage, а
    # «контекст» показывал бы окно без этого обмена, хотя обещает следующий
    # запрос.
    _token_panel(shell, reply)

    if shell.agent.mode == "dialog":
        _dialog_progress(shell, reply)


def _ask(shell: AgentShell, question: str, history: list[chat_core.Message]) -> AgentReply | None:
    """Вызов агента плюс весь вывод вокруг него. None — вызов не состоялся."""
    streaming = chat_core.should_stream(shell.config)
    try:
        reply = shell.agent.ask(
            question, history, on_chunk=console.write_chunk if streaming else None
        )
    except AdventError as error:
        # Ошибка печатается и НЕ убивает сессию — как в REPL недели 01.
        console.fail(error)
        # messages реального запроса построил и потерял упавший вызов;
        # придумывать их здесь значило бы записать в журнал то, чего не
        # отправляли.
        log_call(
            CallResult(model_requested=shell.config.model),
            [{"role": "user", "content": question}],
            week=WEEK,
            day=DAY,
            error=error.message,
        )
        return None

    # result.stream, а не streaming: решение о стриме принимает агент, и
    # печатать ответ второй раз на несовпадении этих двух значений — самый
    # дешёвый способ получить дубль в кадре.
    if reply.result.stream:
        console.finish_answer()
    else:
        console.print_answer(reply.result, shell.config.params.format or "text")

    # API возвращает то же имя модели, что мы прислали (-latest так и остаётся
    # -latest), поэтому конкретную версию подставляем из списка моделей.
    if resolved := shell.resolved:
        reply.result.model_actual = resolved
    if reply.result.skipped_params:
        console.warn(
            f"{shell.config.model} не поддерживает: {', '.join(reply.result.skipped_params)} — "
            "параметры не отправлены"
        )
    if reply.dropped:
        # Токены названы вслух, а не только сообщения (SPEC-w02d06.md §8):
        # «выброшено 6» не отличает освобождённые 200 токенов от 20 000, а
        # именно это число и было смыслом перехода с символьного порога
        # недели 01 на токенный. Счёт не удался — «—», не ноль.
        console.warn(
            f"контекст обрезан: выброшено старых сообщений {reply.dropped}, "
            f"освобождено токенов {_num(reply.dropped_tokens)} — "
            "модель их больше не видит, файл сессии не тронут"
        )

    console.footer(reply.result)
    # reconcile() — чистая функция: калибровку счётчика уже сделал агент, здесь
    # только показ. Сверять надо с тем, что РЕАЛЬНО ушло в API (sent_messages):
    # слой формата дописывает инструкцию внутри chat._payload(), и сверка «до
    # слоя» дала бы стабильную ложную дельту.
    shell.last_check = tokens.reconcile(
        shell.counter, reply.result.sent_messages or [], reply.result.usage.prompt_tokens
    )
    log_call(
        reply.result,
        reply.result.sent_messages or [{"role": "user", "content": question}],
        week=WEEK,
        day=DAY,
    )
    return reply


def _remember(shell: AgentShell, question: str, reply: AgentReply) -> None:
    """Кладёт обмен в память сессии и сохраняет её на диск.

    Текст ответа берётся из reply.history[-1], а НЕ из reply.text: при
    прерывании стрима агент дописывает в историю пометку об обрыве, и в файле
    сессии она должна быть — иначе следующий запуск подсунет модели её же
    оборванный ответ как законченный (SPEC-w02d06.md §12).
    """
    last = reply.history[-1] if reply.history else None
    assistant_text = last["content"] if last and last["role"] == "assistant" else ""
    usage = reply.result.usage
    # usage не пришёл — в файл идёт None, а не нулевой Usage: «неизвестно» не
    # превращается в ноль ни в одной точке (SPEC-w02d06.md §7.3).
    shell.session.record(
        question,
        assistant_text,
        model=shell.config.model,
        usage=None if usage.is_empty() else usage,
    )
    shell.save()


def _dialog_progress(shell: AgentShell, reply: AgentReply) -> None:
    """Учёт ходов целевого диалога: сработал маркер или упёрлись в потолок."""
    shell.dialog_turns += 1
    if reply.done:
        console.note(f"диалог: условие завершения выполнено, ходов: {shell.dialog_turns}")
        shell.dialog_turns = 0
        # Цель достигнута — эпизод закончен, и режим возвращается в chat.
        #
        # Иначе следующая же реплика, включая «спасибо», трактуется как НОВАЯ
        # цель, и агент начинает допрос заново: поймано на живом dry-run
        # 2026-09-07, где после выданного рецепта агент тут же спросил про
        # ингредиенты. Это ещё и регрессия относительно недели 01: там
        # `_run_dialog()` после срабатывания маркера возвращал управление в
        # обычный REPL, то есть эпизод всегда был ограничен.
        #
        # Отличие от ветки потолка ходов ниже принципиальное. Там условие НЕ
        # выполнено, разговор не закончен, и переключать режим значило бы
        # решить за пользователя, что попытка провалена. Здесь наоборот:
        # done — это и есть объявленный им признак конца. Переключение
        # называется вслух, а не делается молча — настройку меняем не мы, её
        # исчерпал результат.
        shell.config.params.mode = "chat"
        console.note("режим вернулся в chat; /mode dialog начнёт новый диалог")
        return
    if shell.agent.turn_limit_reached(shell.dialog_turns):
        # Не ошибка и не выход из сессии: счётчик обнуляется, режим остаётся
        # тем, который выставил пользователь. Молча переключать mode обратно
        # значило бы менять его настройку за него.
        console.warn(
            f"диалог: достигнут потолок ходов ({shell.agent.max_turns}), условие не "
            "выполнено — счётчик сброшен; /mode chat вернёт обычный режим"
        )
        shell.dialog_turns = 0


# --------------------------------------------------------------------------
# Панель токенов
# --------------------------------------------------------------------------


def _num(value: int | None) -> str:
    """Число либо «—». Ноль печатается только тогда, когда он пришёл с сервера."""
    return "—" if value is None else str(value)


def _next_context_tokens(shell: AgentShell) -> int | None:
    """Сколько займёт СЛЕДУЮЩИЙ запрос — до отправки, а не после.

    Пустое сообщение пользователя стоит вместо ещё не набранного вопроса:
    историю, оканчивающуюся ответом ассистента, токенизатор отказывается
    кодировать вовсе (замер 2026-09-07: count() возвращает None), а разница
    между пустым и отсутствующим ходом — это накладные расходы шаблона чата,
    которые в следующем запросе всё равно будут.
    """
    if shell.counter is None:
        return None
    messages = chat_core.build_messages(
        "", system=shell.agent.system_prompt(), history=shell.history
    )
    return shell.counter.count(messages)


def _context_label(shell: AgentShell) -> str:
    used = _next_context_tokens(shell)
    if used is None:
        return "контекст —"
    # Оценка обязана быть помечена как оценка везде, где показывается
    # (SPEC-w02d06.md §7.2) — иначе она читается как точный счёт.
    mark = "" if shell.counter is not None and shell.counter.exact else "~"
    limit = shell.agent.context_limit
    if not limit:
        return f"контекст {mark}{used}/— (окно модели неизвестно)"
    return f"контекст {mark}{used}/{limit} ({used * 100 // limit}%)"


def _session_tokens_label(shell: AgentShell) -> str:
    total = shell.session.token_total()
    label = "—" if total is None else str(total)
    missing = shell.session.missing_usage()
    if missing:
        # Без этой пометки неполная сумма выглядит полной.
        label += f" (без usage: {missing})"
    return label


def _token_panel(shell: AgentShell, reply: AgentReply) -> None:
    """Панель токенов после каждого хода — в stderr, как и весь не-продукт.

    Три числа, и они отвечают на три разных вопроса: сколько стоил этот ход
    (факт с сервера), сколько стоила сессия целиком и сколько займёт следующий
    запрос ДО отправки (SPEC-w02d06.md §7.4).
    """
    usage = reply.result.usage
    turn = f"{_num(usage.prompt_tokens)}/{_num(usage.completion_tokens)}"
    console.note(
        f"токены · ход {turn} · сессия {_session_tokens_label(shell)} · {_context_label(shell)}"
    )


# --------------------------------------------------------------------------
# Слэш-команды
# --------------------------------------------------------------------------


@command("/help", "показать этот список")
def _cmd_help(shell: AgentShell, args: list[str]) -> bool:
    # target=console.err у всех таблиц агента: продукт агента — ответ модели, а
    # список команд, карточка модели и таблица параметров это служебные сводки
    # (SPEC-w02d06.md §14). В неделе 01 та же таблица идёт в stdout законно —
    # там `advent w01 models` печатает её как продукт команды.
    console.commands_help(
        [(entry.usage, entry.help) for entry in unique_commands()], target=console.err
    )
    return False


@command("/exit", "выход", "/quit")
def _cmd_exit(shell: AgentShell, args: list[str]) -> bool:
    console.note("пока")
    return True


@command(
    "/model",
    "текущая модель; с именем — переключить, list all — вообще все модели",
    # Без квадратных скобок: console.commands_help кладёт строку в таблицу
    # Rich, а Rich-markup молча съедает «[all]» как незакрытый тег — ровно
    # этот кусок подписи и пропал бы с экрана.
    usage="/model <имя> | list | list all | info",
)
def _cmd_model(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.note(f"текущая модель: {shell.config.model}")
        return False

    sub = args[0]
    if sub == "list":
        if not shell.models:
            console.warn("список моделей недоступен — не отвечает API")
            return False
        show_all = len(args) > 1 and args[1] == "all"
        shown = shell.models if show_all else chat_models(shell.models)
        console.models_table(shown, highlight=shell.config.model, target=console.err)
        console.note(f"показано {len(shown)} из {len(shell.models)}")
        return False

    if sub == "info":
        if shell.card is None:
            console.warn("карточка модели недоступна — не отвечает API")
            return False
        console.model_card(shell.card, shell.config.model, target=console.err)
        return False

    previous = shell.config.model
    shell.config.model = sub
    try:
        shell.retarget()
    except (ConfigError, AdventError) as error:
        # Опечатка в имени модели не должна ронять сессию: чинится следующей
        # же командой.
        shell.config.model = previous
        shell.retarget()
        console.warn(str(getattr(error, "message", error)))
        return False

    resolved = shell.resolved
    console.note(f"модель переключена на {sub}" + (f" → {resolved}" if resolved else ""))
    _, skipped = shell.config.params.as_payload(shell.capabilities)
    if skipped:
        console.warn(f"{sub} не поддерживает: {', '.join(skipped)} — параметры не отправляются")
    return False


@command("/params", "параметры, которые читает агент, и итоговый system prompt")
def _cmd_params(shell: AgentShell, args: list[str]) -> bool:
    # describe(AGENT_PARAMS), а не describe(): агент не читает problem/runs/
    # judge/temps/models, и показывать их значило бы предлагать выставить
    # параметр, который ни на что не влияет (SPEC-w02d06.md §16).
    rows = []
    for name, value, help_text in shell.config.params.describe(AGENT_PARAMS):
        if BY_NAME[name].local:
            # Скобки круглые: Rich съедает квадратные как незакрытый тег.
            help_text = f"{help_text} (local — не уходит на сервер)"
        rows.append((name, value, help_text))
    console.params_table(rows, target=console.err)

    system = _final_system_prompt(shell)
    if system:
        # rich_escape: в system попадает инструкция формата и, при
        # format=schema, дамп JSON Schema — квадратные скобки массивов Rich
        # иначе молча съест вместе с куском текста.
        console.note(f"итоговый system prompt:\n{rich_escape(system)}")
    else:
        console.note("system prompt не задан")
    return False


def _final_system_prompt(shell: AgentShell) -> str | None:
    """Ровно то, что уедет в system-сообщение запроса.

    Агент собирает персону и пресет режима, а слой формата дописывает
    chat._payload() уже перед отправкой — здесь зовётся та же
    formats.build_system(), чтобы /params не расходился с реальным запросом
    (та же ловушка, что чинили в неделе 01, НАХОДКА 1).
    """
    base = shell.agent.system_prompt()
    format_name = shell.config.params.format or "text"
    schema = None
    if format_name == "schema" and shell.config.params.schema_file:
        try:
            schema = formats.load_schema(shell.config.params.schema_file)
        except ConfigError:
            schema = None  # покажем то, что есть — ошибка всплывёт на запросе
    return formats.build_system(format_name, base, schema)


@command(
    "/set",
    "изменить параметр (default — вернуть умолчание команды)",
    usage="/set <параметр> <значение>",
)
def _cmd_set(shell: AgentShell, args: list[str]) -> bool:
    if len(args) < 2:
        console.warn("нужно: /set <параметр> <значение>. /params — что есть")
        return False

    name, raw = args[0], " ".join(args[1:])
    if name not in AGENT_PARAMS:
        console.warn(
            f"агент не читает параметр {name!r} — /params показывает те, что читает. "
            "Выставленный параметр, ни на что не влияющий, выглядит как поломка"
        )
        return False

    try:
        shell.config.params.set(name, raw)
    except ParamError as error:
        console.warn(str(error))
        return False

    # `default` означает «умолчание ЭТОЙ команды», а не «пусто навсегда»:
    # правило живёт в реестре параметров, а не здесь.
    shell.config.params.apply_defaults(AGENT_COMMAND)
    value = getattr(shell.config.params, name)
    console.note(f"{name} = {_shown(value)}")

    if value is not None:
        try:
            _check_local_param(name, value)
        except ConfigError as error:
            # Параметр остаётся выставленным: пользователь видит предупреждение
            # и правит следующей командой.
            console.warn(str(error))
            return False

    if name == "session":
        _switch_session(shell, str(value or DEFAULT_SESSION))
        return False
    if name == "mode":
        shell.dialog_turns = 0
    if name in ("mode", "done", "stop"):
        # Негодный или съедаемый stop'ом маркер надо поймать сейчас, а не
        # тогда, когда диалог не закончится ни разу.
        _warn_text(shell.agent.check_done())

    _, skipped = shell.config.params.as_payload(shell.capabilities)
    if name in skipped:
        console.warn(f"{shell.config.model} не поддерживает {name} — параметр не отправляется")
    return False


def _shown(value: object) -> object:
    if value is None:
        return "сброшен"
    if isinstance(value, bool):
        return "вкл" if value else "выкл"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, str):
        # done=text:[ГОТОВО] содержит квадратные скобки — Rich съест их как
        # незакрытый тег, и маркер пропадёт с экрана.
        return rich_escape(value)
    return value


@command("/mode", "режим разговора: chat или dialog (целевой диалог)", usage="/mode chat | dialog")
def _cmd_mode(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.note(f"режим: {shell.agent.mode}, ходов диалога: {shell.dialog_turns}")
        return False
    # Делегируем /set: правило разбора и валидации значения должно жить в
    # одном месте, иначе `/mode dialg` и `/set mode dialg` ответят по-разному.
    return _cmd_set(shell, ["mode", *args])


@command("/again", "повторить последний вопрос с текущими настройками, без истории")
def _cmd_again(shell: AgentShell, args: list[str]) -> bool:
    """Независимый прогон того же вопроса: ни истории, ни записи в сессию.

    При повторе «через историю» модель видит собственный прошлый ответ и
    отвечает «как я говорил выше» — сравнивать настройки становится не с чем.
    В память сессии результат тоже не идёт: это проба, а не ход разговора.
    """
    if shell.agent.mode == "dialog":
        console.warn("/again недоступна в mode=dialog — там история и есть суть диалога")
        return False
    if shell.last_question is None:
        console.warn("нечего повторять — сначала задай вопрос")
        return False
    reply = _ask(shell, shell.last_question, [])
    if reply is not None:
        # Панель печатается по неизменной истории — и это правда: рабочий
        # контекст этот прогон не пополнил.
        _token_panel(shell, reply)
    return False


@command("/reset", "очистить рабочий контекст запуска (файл сессии остаётся)")
def _cmd_reset(shell: AgentShell, args: list[str]) -> bool:
    shell.history = []
    shell.dialog_turns = 0
    console.note(
        "рабочий контекст очищен — модель забыла разговор; файл сессии не тронут, стирает его /new"
    )
    return False


@command(
    "/new",
    "начать пустую сессию: без имени — текущую заново, с именем — другую (содержимое стирается)",
    usage="/new <имя>",
)
def _cmd_new(shell: AgentShell, args: list[str]) -> bool:
    name = args[0] if args else shell.session.name
    _switch_session(shell, name, fresh=True)
    return False


@command("/sessions", "список сессий: дата, число ходов, токены")
def _cmd_sessions(shell: AgentShell, args: list[str]) -> bool:
    infos, warnings = list_sessions(shell.directory)
    for warning in warnings:
        console.warn(warning)
    if not infos:
        console.note("сессий пока нет")
        return False

    table = Table(title="Сессии агента")
    table.add_column("сессия", style="cyan", no_wrap=True)
    table.add_column("начата", style="dim")
    table.add_column("ходов", justify="right")
    table.add_column("токенов", justify="right")
    for info in infos:
        tokens_label = _num(info.tokens)
        if info.missing_usage:
            tokens_label += f" (без usage: {info.missing_usage})"
        turns_label = str(info.turns)
        if info.broken:
            # «Ходов 0» про нечитаемый файл — это утверждение о разговоре,
            # которого мы не видели. Файл при листинге не тронут (session.py),
            # так что его ещё можно починить руками.
            turns_label = tokens_label = "нечитаема"
        table.add_row(
            info.name,
            info.created,
            turns_label,
            tokens_label,
            style="bold green" if info.name == shell.session.name else None,
        )
    # В stderr, а не в stdout: продукт агента — ответ модели, список сессий это
    # служебная сводка (SPEC-w02d06.md §14).
    console.err.print(table)
    return False


def _component_label(turns: list[Turn], field_name: str) -> str:
    """Сумма одного поля usage по ходам сессии, с честной пометкой о пробелах.

    Покомпонентно и с отслеживанием None, а не `value or 0`: usage бывает
    частичным. Локальный OpenAI-совместимый сервер (`--base-url`, SPEC §7.2)
    присылает {"total_tokens": 51} без prompt/completion, и `is_empty()`
    возвращает для такого usage False — total пришёл. Сложение через `or 0`
    напечатало бы «0/0», то есть выдало бы неизвестное за точный ноль ровно в
    той команде, ради которой заведён день (SPEC §7.3).
    """
    total = 0
    known = unknown = 0
    for turn in turns:
        usage = turn.usage
        if usage is None or usage.is_empty():
            # Ход вовсе без usage учтён отдельной оговоркой «(без usage: N)»
            # у суммы за сессию — дублировать её здесь нечего.
            continue
        value = getattr(usage, field_name)
        if value is None:
            unknown += 1
            continue
        known += 1
        total += value
    if not known:
        return "—"
    return f"{total} (+{unknown} неизвестно)" if unknown else str(total)


@command("/tokens", "разбивка по токенам: сессия, окно контекста, сверка с сервером")
def _cmd_tokens(shell: AgentShell, args: list[str]) -> bool:
    prompt_label = _component_label(shell.session.turns, "prompt_tokens")
    completion_label = _component_label(shell.session.turns, "completion_tokens")

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column()

    counter = shell.counter
    if counter is None:
        table.add_row("счёт токенов", "нет счётчика — считать нечем")
    elif counter.exact:
        table.add_row("счёт токенов", f"{counter.name}, точный")
    else:
        # Оценка помечается как оценка везде, где показывается: выдать её за
        # точный счёт хуже, чем не показать вовсе (SPEC-w02d06.md §7.2).
        table.add_row("счёт токенов", f"{counter.name} (~), калибруется по usage сервера")
    table.add_row("сессия", f"{shell.session.name}, ходов {len(shell.session.turns)}")
    table.add_row("prompt/completion", f"{prompt_label}/{completion_label}")
    table.add_row("всего за сессию", _session_tokens_label(shell))
    table.add_row("следующий запрос", _context_label(shell))
    table.add_row("окно модели", _num(shell.agent.context_limit))

    check = shell.last_check
    if check is None:
        # «Сверять не с чем» — отдельный исход, а не «сошлось»: в этом запуске
        # ещё не было ни одного ответа.
        table.add_row("сверка с сервером", "в этом запуске ходов ещё не было")
    elif check.delta is None:
        mark = "" if check.exact else "~"
        table.add_row(
            "сверка с сервером", f"локально {mark}{_num(check.local)}, сервер не прислал usage"
        )
    elif check.exact:
        verdict = "сошлось" if check.matches else "разошлось"
        table.add_row(
            "сверка с сервером",
            f"локально {check.local}, сервер {check.server} (дельта {check.delta:+d}) — {verdict}",
        )
    else:
        # У оценки расхождение — это норма и повод откалиброваться, а не
        # поломка: то же правило, по которому молчит TokenCheck.warning().
        # Слово «разошлось» здесь означало бы «таблица токенизаторов
        # разъехалась с моделью» (SPEC §7.1) — вердикт про то, чего не
        # измеряли. Плюс `~`: оценка помечается везде, где показывается (§7.2).
        table.add_row(
            "сверка с сервером",
            f"оценка ~{check.local} против {check.server} у сервера "
            f"(дельта {check.delta:+d}) — коэффициент калибруется по usage, "
            "в том числе по этому же ходу",
        )
    console.err.print(table)
    return False


def _switch_session(shell: AgentShell, name: str, *, fresh: bool = False) -> None:
    """Переключение сессии: текущая сохраняется, история подменяется.

    `fresh` стирает содержимое целевой сессии — это и есть `/new`. Отказ вместо
    стирания выглядел бы безопаснее, но сорвал бы второй дубль записи: демо
    начинается с пустой сессии, и повторный прогон обязан приводить в то же
    состояние, что и первый.
    """
    validate_name(name)
    shell.save()
    session = shell.open_session(name)
    if fresh:
        dropped = len(session.turns)
        session.clear()
        if dropped:
            console.warn(f"сессия {session.name}: удалено ходов {dropped}")
    shell.session = session
    shell.history = session.history()
    shell.config.params.session = session.name
    shell.dialog_turns = 0
    shell.last_question = None
    shell.last_check = None
    if fresh:
        shell.save()


def _dispatch(line: str, shell: AgentShell) -> bool:
    """Исполняет слэш-команду. True означает «выходим»."""
    parts = line.split()
    entry = COMMANDS.get(parts[0])
    if entry is None:
        console.warn(f"неизвестная команда {parts[0]}. /help — список")
        return False
    try:
        return entry.handler(shell, parts[1:])
    except ConfigError as error:
        # Опечатка в /set или в имени сессии предупреждает, а не убивает
        # разговор — как и ошибка вызова API.
        console.warn(str(error))
        return False
    except AdventError as error:
        console.fail(error)
        return False


# --------------------------------------------------------------------------
# Ввод
# --------------------------------------------------------------------------


def _read_line(label: str) -> str | None:
    """Одна строка от пользователя. None — ввод закончился (EOF/Ctrl+C)."""
    try:
        line = typer.prompt(f"\n{label}", prompt_suffix=" › ", err=True)
    except (EOFError, typer.Abort):
        return None
    line = line.strip()
    # При запуске из скрипта записи stdin — это pipe, и терминал не показывает
    # то, что в него подали: без эха в кадре видно приглашение без вопроса.
    if line and not sys.stdin.isatty():
        console.echo_input(line)
    return line


def _read_input() -> str | None:
    """Реплика пользователя, возможно многострочная. None — конец ввода."""
    line = _read_line("ты")
    if line != MULTILINE_SENTINEL:
        return line
    return _read_multiline()


def _read_multiline() -> str:
    """Копит строки до закрывающего sentinel. Пустая строка внутри — законна."""
    lines: list[str] = []
    while True:
        try:
            # default="": без него click переспрашивает на пустой строке, а
            # пустая строка внутри многострочного текста — это абзац, а не
            # отсутствие ввода.
            raw = typer.prompt("", prompt_suffix="… ", err=True, default="", show_default=False)
        except (EOFError, typer.Abort):
            break
        if raw and not sys.stdin.isatty():
            console.echo_input(raw)
        if raw.strip() == MULTILINE_SENTINEL:
            break
        lines.append(raw)
    return "\n".join(lines).strip()


def _loop(shell: AgentShell) -> None:
    """Разговор: реплики, слэш-команды, выход."""
    console.note(
        "агент. /help — список команд, "
        f"{MULTILINE_SENTINEL} — многострочный ввод, Ctrl+C прерывает генерацию."
    )
    if shell.history:
        console.note(f"подхвачен контекст сессии: сообщений {len(shell.history)}")

    while True:
        line = _read_input()
        if line is None:
            console.note("пока")
            return
        if not line:
            continue
        if line.startswith("/"):
            if _dispatch(line, shell):
                return
            continue
        _turn(shell, line)


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def _default_hint(param: str) -> str:
    """«(по умолчанию X)» из реестра параметров, а не литералом в сигнатуре.

    Умолчание, напечатанное рядом литералом, становится вторым источником
    истины и однажды разъезжается с настоящим — так в `--temperature` полгода
    висел несуществующий потолок 2.0 (CLAUDE.md).
    """
    value = defaults_for(AGENT_COMMAND).get(param)
    return f"(по умолчанию {value})"


def _check_local_param(name: str, value: object) -> None:
    """Ранняя проверка смысла значения, а не только его формы.

    Spec в params.py валидирует форму; существование файла схемы, понятность
    префикса done и допустимость имени сессии знают formats.py и session.py.
    Зовутся они здесь ради самой дешёвой точки отказа: сразу, а не на первом
    запросе к модели и не посреди демо.
    """
    if name == "schema_file":
        formats.load_schema(str(value))
    elif name == "done":
        formats.parse_done(str(value))
    elif name == "session":
        validate_name(str(value))


def _apply_local_flags(config: Config, **flags: object) -> None:
    """Кладёт локальные флаги в config.params той же машинерией, что и /set."""
    try:
        for name, raw in flags.items():
            if raw is None:
                continue
            config.params.set(name, raw)
            _check_local_param(name, raw)
    except ParamError as exc:
        raise ConfigError(str(exc)) from exc


app = typer.Typer(
    help="Агент курса AI Advent 9: разговор с памятью, учётом токенов и режимами.",
    add_completion=False,
)


@app.command()
def agent(
    session: str | None = typer.Option(
        None, "--session", "-s", help=f"Имя сессии {_default_hint('session')}."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Имя модели."),
    system: Path | None = typer.Option(
        None, "--system", help="Файл с system prompt — перекрывает персону агента."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Базовый URL OpenAI-совместимого endpoint "
        "(например http://127.0.0.1:1234 для LM Studio).",
    ),
    mode: str | None = typer.Option(
        None, "--mode", help=f"Режим: {', '.join(MODE_CHOICES)} {_default_hint('mode')}."
    ),
    done: str | None = typer.Option(
        None,
        "--done",
        help=f'Условие завершения целевого диалога: "text:<строка>" или "json:<поле>" '
        f"{_default_hint('done')}.",
    ),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help=f"Потолок ходов диалога {_default_hint('max_turns')}."
    ),
    # help берётся из реестра, а не пишется здесь второй раз: тот же текст
    # печатает /params, и «0..2» однажды уже разошлось с потолком API 1.5.
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help=BY_NAME["temperature"].help
    ),
    top_p: float | None = typer.Option(None, "--top-p", help=BY_NAME["top_p"].help),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help=BY_NAME["max_tokens"].help),
    seed: int | None = typer.Option(None, "--seed", help=BY_NAME["random_seed"].help),
    stop: str | None = typer.Option(None, "--stop", help=BY_NAME["stop"].help),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=f"Глубина рассуждения: {', '.join(REASONING_EFFORTS)}. Нужна capability reasoning.",
    ),
    format_: str | None = typer.Option(
        None, "--format", help=f"Пресет формата ответа: {', '.join(FORMAT_CHOICES)}."
    ),
    schema_file: str | None = typer.Option(
        None, "--schema-file", help="Путь к .json со схемой ответа — для --format schema."
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Печатать ответ по мере генерации."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запуска в stderr."),
) -> None:
    """Разговор с агентом: память сессии, учёт токенов, режимы chat и dialog."""
    config = Config.resolve(
        model=model,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        random_seed=seed,
        stop=stop,
        reasoning_effort=reasoning_effort,
        stream=stream,
        verbose=verbose,
        base_url=base_url,
    )
    # Config.resolve() не знает про локальные параметры — кладём их той же
    # машинерией, что и /set, с той же валидацией и теми же ParamError.
    _apply_local_flags(
        config,
        session=session,
        mode=mode,
        done=done,
        max_turns=max_turns,
        format=format_,
        schema_file=schema_file,
    )
    # Умолчания приходят из реестра, а не литералами в сигнатуре typer: иначе
    # одно правило записано в двух местах и разъедется при первой правке.
    config.params.apply_defaults(AGENT_COMMAND)

    shell = AgentShell(config)
    if config.verbose:
        counter_name = shell.counter.name if shell.counter else "нет"
        console.note(
            f"model={config.model} session={shell.session.name} mode={shell.agent.mode} "
            f"stream={chat_core.should_stream(config)} счётчик={counter_name} "
            f"окно={_num(shell.agent.context_limit)}"
        )
    _loop(shell)


def main() -> None:
    """Точка входа `adventagent`: ошибки текстом, а не traceback."""
    # Первым делом: без этого консоль Windows падает в cp1251 и кириллица в
    # ответе превращается в мусор прямо в кадре.
    console.force_utf8()
    try:
        app()
    except (AdventError, ConfigError) as error:
        if isinstance(error, ConfigError):
            wrapped = AdventError(str(error))
            wrapped.exit_code = ConfigError.exit_code
            error = wrapped
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


# `python -m week_02.cli` — так агента зовёт машинерия записи демо
# (advent_cli/record.py, Step.module).
if __name__ == "__main__":
    main()
