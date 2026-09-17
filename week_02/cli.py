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

import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console, formats, profiles, tokens
from advent_core.agent import Agent, AgentReply, Compaction, FactsFailure, FactsUpdate
from advent_core.client import (
    capabilities_of,
    chat_models,
    find_model,
    list_models,
    model_names,
    resolve_alias,
)
from advent_core.compact import summary_messages
from advent_core.config import (
    DEFAULT_SYSTEM_PROMPT,
    Config,
    ConfigError,
    configured_secrets,
    redact,
)
from advent_core.errors import AdventError
from advent_core.facts import format_facts, render_note, validate_key
from advent_core.invariants import (
    INVARIANTS_STATE_KEY,
    Invariant,
    InvariantAssessment,
    InvariantError,
    InvariantSet,
    invariant_messages,
)
from advent_core.journal import log_call, log_internal_call
from advent_core.memory import (
    MemoryFailure,
    MemorySnapshot,
    MemoryStore,
    MemoryUpdate,
    ShortTermMemory,
    StructuredMemory,
    is_credential_like,
    manual_set,
    render_delta,
    render_memory,
    split_key,
)
from advent_core.memory import (
    validate_key as validate_memory_key,
)
from advent_core.params import (
    AGENT_COMMAND,
    AGENT_PARAMS,
    BY_NAME,
    CONTEXT_STRATEGY_CHOICES,
    FORMAT_CHOICES,
    MODE_CHOICES,
    REASONING_EFFORTS,
    ParamError,
    defaults_for,
)
from advent_core.session import (
    BRANCH_SEP,
    DEFAULT_SESSION,
    ROLE_ASSISTANT,
    Session,
    Turn,
    build_tree,
    checkpoint_file_name,
    delete_branch,
    list_sessions,
    make_branch,
    make_checkpoint,
    parse_name,
    root_of,
    validate_name,
)
from advent_core.task_state import (
    TASK_STATE_KEY,
    TaskState,
    TaskStateError,
    task_messages,
)
from advent_core.telemetry import CallResult
from advent_core.tokens import TokenCheck, counter_for

WEEK = 2
# Day that last touched THIS code. The test asserts the literal 9, not this
# constant — an expectation drawn from the same source as the code under
# test can never go red (CLAUDE.md, "A test whose expected value comes from
# the same source as the code under test").
#
# No separate constant for compaction, unlike week 01's TEMP_DAY: there one
# module held commands from two different days (`chat` day 03, `temp` day
# 04); here the week is ONE app growing by day, same command throughout.
# A compaction call is distinguished from a regular turn by kind=compact in
# the journal, not by a day number.
DAY = 10
# Day 14 is additive to the long-lived agent.  Existing conversation and
# service calls keep their historical journal provenance; only the new
# invariant preflight records its Week 03 / Day 14 origin.
INVARIANT_WEEK = 3
INVARIANT_DAY = 14

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# Персона агента — своя, а не общекурсовая «лаконичный ассистент, без воды»:
# та персона гасит встречные вопросы, а агент должен уметь их задавать
# (advent_core/prompts/default_system.md против week_02/prompts/agent.md).
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent.md"
# Копия пресета недели 01, а не импорт из неё: у недели своя копия, и правка
# диалога агента не должна менять поведение уже сданного дня 02.
DIALOG_PROMPT_PATH = PROMPTS_DIR / "dialog.md"
# The summarizer prompt lives in advent_core/prompts, not in the week: history
# compaction is core mechanics (inherited by anything built on Agent), not a
# week-02 persona like agent.md/dialog.md above.
SUMMARY_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT.parent / "summary.md"
# Same reasoning for the facts extractor (day 10, SPEC-w02d10.md §5): core
# mechanics, so the prompt lives in advent_core/prompts, not week_02.
FACTS_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT.parent / "facts.md"

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
_FACTS_ALIAS_SHOWN = False


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
    # Имена local-параметров, заданных флагами запуска явно. Явный флаг бьёт
    # файл сессии (приоритет «флаг > файл > дефолт реестра»): явный выбор
    # пользователя не наш, чтобы его игнорировать — то же правило, по которому
    # --system перекрывает персону проекта.
    explicit: frozenset[str] = frozenset()

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
    # Summary of the old part of the conversation, currently in effect. Lives
    # here, not in the agent: the agent holds no memory at all
    # (advent_core/agent.py) — it takes the summary in ask() and returns the
    # current one in AgentReply, same as history.
    summary: str | None = None
    # Cost of compactions this run. Summarizer calls deliberately don't land
    # in session.turns (they're service calls, not conversation turns), so
    # neither "total for session" nor the growth table sees them — but they
    # are real spend, and something has to name it (SPEC-w02d09.md §7).
    compact_calls: int = 0
    compact_prompt: int = 0
    compact_completion: int = 0

    # Sticky facts (day 10): current dict/pins/coverage cursor, mirroring
    # summary/summary_upto above — same "content lives here, agent has no
    # memory of its own" split.
    facts: dict[str, str] = field(default_factory=dict)
    facts_pinned: list[str] = field(default_factory=list)
    facts_upto: int = 0
    # Extractor call cost this run — mirrors compact_calls/compact_prompt/
    # compact_completion, same reason: a service call, invisible to both
    # "total for session" and the growth table.
    facts_calls: int = 0
    facts_prompt_tokens: int = 0
    facts_completion_tokens: int = 0
    # Paid extractor calls that came back unusable (truncated/invalid delta) —
    # counted separately from facts_calls above so /tokens can name them
    # instead of folding them into a number that reads as "N successful
    # updates" (review finding #5: these calls cost money and were going
    # unjournalled).
    facts_failures: int = 0
    # Explicit Day 11 memory.  The store is deliberately owned by the shell,
    # not Agent: Agent remains stateless and can be used by offline tests.
    memory_store: MemoryStore = field(init=False)
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    memory_upto: int = 0
    memory_dirty_working: bool = False
    memory_dirty_long_term: bool = False
    memory_calls: int = 0
    memory_prompt_tokens: int = 0
    memory_completion_tokens: int = 0
    memory_failures: int = 0
    # Runtime-only (never persisted): whether `/set context_strategy <value>`
    # was issued explicitly THIS run. Deliberately not derived from the
    # session file — otherwise `/set compact off|on`, day 09's own alias,
    # would behave differently the moment a session survives a restart, since
    # _save_state always writes context_strategy (SPEC-w02d10.md §3).
    context_strategy_explicit: bool = False
    # Last `window_dropped` value already announced this run. None means
    # "never announced yet" — distinct from 0, which IS a value the window
    # strategy can legitimately report. Without this, "окно: выпало N
    # сообщений" repeats the same N every turn once the window fills (review
    # finding T1) — reset whenever the conversation identity changes
    # (_switch_session, covering /new, /switch, /branch, /set session),
    # otherwise a stale number would suppress a genuine new drop.
    window_dropped_reported: int | None = None
    active_profile: str | None = None
    profile_values: dict[str, str] = field(default_factory=dict)
    task: TaskState | None = None
    task_needs_healing: bool = False
    # Invariants are Session settings rather than conversation content: they
    # outlive /new, while Session.clear() still removes summaries/facts/task.
    invariants: InvariantSet = field(default_factory=InvariantSet)
    invariants_needs_healing: bool = False
    invariant_assessment_calls: int = 0
    invariant_assessment_prompt_tokens: int = 0
    invariant_assessment_completion_tokens: int = 0
    invariant_assessment_failures: int = 0

    def __post_init__(self) -> None:
        try:
            self.refresh()
        except ConfigError:
            # Модели из конфига нет на сервере. Фатально это только когда чинить
            # не на месте: не-tty (пайп, запись демо, CI) — спрашивать некого,
            # и по-прежнему падаем ConfigError'ом с текстом из console. Спрашивать
            # можно ТОЛЬКО здесь, на старте: retarget() из /model <опечатка>
            # продолжает откатываться к прежней модели с warning'ом.
            if not sys.stdin.isatty():
                raise
            # Сервер без capabilities (LM Studio): chat_models() вернёт
            # пусто, и выбор не предложат вовсе — тогда предлагаем всё, что
            # сервер отдал. Отфильтрованный список предпочтительнее: облачный
            # список содержит и эмбеддинги.
            candidates = chat_models(self.models) or self.models
            chosen = console.choose_model(
                candidates,
                current=self.config.model,
                base_url=self.config.base_url,
            )
            if chosen is None:
                raise
            self.config.model = chosen
            # counter_for(), capabilities и context_limit ниже по __post_init__
            # читают config.model уже новый — пересчитывать ничего не надо, было
            # бы надо, зови выбор ПОСЛЕ счётчика.
            self.refresh()
        target_session = self.config.params.session or DEFAULT_SESSION
        # Checkpoints are file-snapshots, not sessions — the same rule /switch
        # enforces (SPEC-w02d10.md §7.1). Checked here too: `--session <cp>`
        # used to reach open_session() directly, landing the user live-editing
        # an immutable snapshot (review finding P1). No one to ask at startup,
        # so this is fatal, same as an unknown model name below.
        refusal = _checkpoint_switch_refusal(self, target_session)
        if refusal:
            raise ConfigError(refusal)
        self.session = self.open_session(target_session)
        self.history = self.session.history()
        # Подхватываем служебное состояние сессии ДО сборки агента: его
        # property mode читает config.params, поэтому порядок безопасен, а
        # анонс — после создания agent, где уже есть max_turns.
        resumed_dialog = self._apply_session_state(self.session, check_explicit=True)
        self._load_profile()
        self._load_memory()
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
            context_limit=self.effective_context_limit(),
            persona=_persona(self.config),
            dialog_preset=_dialog_preset(),
            summary_prompt=_summary_prompt(),
            facts_prompt=_facts_prompt(),
            memory_prompt=_memory_prompt(),
            # Агент не печатает — он отдаёт текст сюда (SPEC §5).
            on_warning=console.warn,
        )
        if self.config.params.context_limit is not None:
            # Override — инструмент демо, и молча живущий заниженный лимит
            # после перезапуска — ловушка: на старте его называют вслух,
            # с обоими числами (SPEC-w02d08.md §4).
            console.note(
                f"лимит окна переопределён: {self.config.params.context_limit} "
                f"(карточка: {_num(tokens.context_limit(self.card))})"
            )
        if resumed_dialog:
            console.note(
                f"подхвачен целевой диалог: ход {self.dialog_turns} из {self.agent.max_turns}"
            )
            if self.task is not None:
                console.warn(
                    "Persisted task несовместима с mode=dialog — "
                    "используй /mode chat либо /task clear"
                )
        # check_done() зовётся и на применённом из файла маркере: негодный done,
        # оставшийся в сессии с прошлого запуска, ловим сейчас, а не тогда,
        # когда диалог не закончится ни разу.
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
                console.model_unavailable_text(
                    self.config.model, sorted(names), base_url=self.config.base_url
                )
            )
        self.card = find_model(self.models, self.config.model)

    @property
    def capabilities(self) -> dict | None:
        return capabilities_of(self.models, self.config.model) if self.models else None

    def effective_context_limit(self) -> int | None:
        """Лимит окна с учётом override: заданный вручную бьёт карточку модели.

        Приоритет «параметр > карточка» — тот же порядок, что у mode/done
        (SPEC-w02d07 §3, SPEC-w02d08.md §4). Минимум реестра — 1, поэтому
        `or` безопасен: нулевое значение сюда не доходит, а None означает
        «карточка».
        """
        return self.config.params.context_limit or tokens.context_limit(self.card)

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
        self.agent.context_limit = self.effective_context_limit()
        if self.config.params.context_limit is not None:
            # Override — явный выбор пользователя: смена модели его НЕ
            # переопределяет, но молча показывать лимит от чужой карточки
            # нельзя — предупреждаем, что окно не из карточки новой модели
            # (SPEC-w02d08.md §4).
            console.warn(
                f"лимит окна переопределён: {self.config.params.context_limit} — "
                "это не окно новой модели "
                f"(карточка: {_num(tokens.context_limit(self.card))}); "
                "/set context_limit default вернёт лимит из карточки"
            )
        self.warn_model_drift()

    # --- сессия ------------------------------------------------------------

    def open_session(self, name: str, *, quarantine: bool = True) -> Session:
        """Загружает сессию с диска и говорит вслух обо всём, что с ней не так."""
        session = Session.load(name, directory=self.directory, quarantine=quarantine)
        for warning in session.warnings:
            console.warn(warning)
        console.note(f"сессия {session.name}: ходов {len(session.turns)}, начата {session.created}")
        self._warn_drift(session)
        return session

    def _load_profile(self) -> None:
        """Load the session's optional global profile without breaking a run."""
        raw = self.session.state.get("active_profile")
        self.active_profile = raw if isinstance(raw, str) and raw else None
        self.profile_values = {}
        if self.active_profile:
            try:
                self.profile_values = profiles.load(
                    self.active_profile, self.memory_root / "profiles"
                )
            except ConfigError as error:
                console.warn(str(error) + " — запросы идут без profile")

    def _save_profile_state(self) -> None:
        self.session.state["active_profile"] = self.active_profile

    @property
    def memory_root(self) -> Path:
        """Root for structured memory, parallel to the session directory.

        Tests pass a temporary session directory, so using it as the root keeps
        all state isolated.  The normal run uses ``logs`` and therefore writes
        exactly ``logs/memory/...``.
        """
        from advent_core.config import LOG_DIR

        return self.directory or LOG_DIR

    def _memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            ShortTermMemory(tuple(self.history), 0, len(self.session.turns)),
            self.memory.working,
            self.memory.long_term,
        )

    def _load_memory(self, *, load_working: bool = True) -> None:
        self.memory_store = MemoryStore(self.memory_root)
        long_result = self.memory_store.load_long_term()
        working_result = (
            self.memory_store.load_working(self.session.name, turns_count=len(self.session.turns))
            if load_working
            else None
        )
        warnings = long_result.warnings + (working_result.warnings if working_result else ())
        for warning in warnings:
            console.warn(warning)

        working = working_result.value if working_result else StructuredMemory()
        migrated = False
        if working_result is not None and not working_result.exists:
            # Legacy facts are session content and never become global memory.
            mapped: dict[str, str] = {}
            mapped_pins: set[str] = set()
            for key, value in self.facts.items():
                category, _, name = key.partition(".")
                field_name = {
                    "цель": "goal",
                    "ограничения": "constraints",
                    "предпочтения": "constraints",
                    "решения": "decisions",
                    "договорённости": "decisions",
                }.get(category)
                if field_name is None or not name:
                    continue
                suffix = name
                if category == "предпочтения":
                    suffix = f"legacy_preference.{name}"
                elif category == "договорённости":
                    suffix = f"legacy_agreement.{name}"
                canonical = f"{field_name}.{suffix}"
                mapped[canonical] = value
                if key in self.facts_pinned:
                    mapped_pins.add(canonical)
            working = StructuredMemory(mapped, frozenset(mapped_pins))
            migrated = bool(mapped) or "facts" in self.session.state

        upto = working_result.upto if working_result else 0
        if working_result is not None and not working_result.exists:
            # The Day 10 facts cursor is the only legacy coverage boundary.
            # Carry it into the new per-session working-memory file, but never
            # trust a malformed value to trigger a paid catch-up over history.
            legacy_upto = self.session.state.get("facts_upto", 0)
            valid_legacy = (
                isinstance(legacy_upto, int)
                and not isinstance(legacy_upto, bool)
                and 0 <= legacy_upto <= len(self.session.turns)
            )
            if valid_legacy:
                upto = legacy_upto
            elif "facts_upto" in self.session.state:
                upto = len(self.session.turns)
                console.warn(
                    f"в сессии {self.session.name} негодный facts_upto={legacy_upto!r} — "
                    f"legacy cursor перенесён как {upto}; пересобрать заново: /memory backfill"
                )
        if (
            not isinstance(upto, int)
            or isinstance(upto, bool)
            or not 0 <= upto <= len(self.session.turns)
        ):
            console.warn(
                f"в memory working {self.session.name} негодный cursor {upto!r} — "
                f"считаем покрытым {len(self.session.turns)} сообщений; backfill — явно"
            )
            upto = len(self.session.turns)
        self.memory = MemorySnapshot(
            ShortTermMemory(tuple(self.history), 0, len(self.session.turns)),
            working,
            long_result.value,
        )
        self.memory_upto = upto
        if migrated:
            self.memory_dirty_working = True
            self._save_memory()

    def _save_memory(self) -> None:
        """Persist dirty structured layers independently; failed layers stay dirty."""
        if self.memory_dirty_long_term:
            try:
                self.memory_store.save_long_term(self.memory.long_term)
                self.memory_dirty_long_term = False
            except OSError as error:
                console.warn(f"long-term memory не сохранена ({error})")
        if self.memory_dirty_working:
            try:
                self.memory_store.save_working(
                    self.session.name, self.memory.working, self.memory_upto
                )
                self.memory_dirty_working = False
            except OSError as error:
                console.warn(f"working memory не сохранена ({error})")

    def _mark_memory_update(self, update: MemoryUpdate) -> None:
        old_upto = self.memory_upto
        self.memory = update.snapshot
        self.memory_upto = update.memory_upto
        layers = {layer for layer, _op in update.applied}
        # A successful extractor call advances the working cursor even when
        # it produced only long-term changes (or an empty delta).  The cursor
        # is persisted in the working file, so that file is dirty on cursor
        # movement as well as on a working operation.
        self.memory_dirty_working |= "working" in layers or self.memory_upto != old_upto
        self.memory_dirty_long_term |= "long_term" in layers

    def _mark_memory_failure(self, failure: MemoryFailure) -> None:
        self.memory_failures += 1
        usage = getattr(failure.call_result, "usage", None)
        if usage is not None:
            self.memory_prompt_tokens += usage.prompt_tokens or 0
            self.memory_completion_tokens += usage.completion_tokens or 0

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

    def _apply_session_state(self, session: Session, *, check_explicit: bool) -> bool:
        """Применяет служебное состояние сессии: mode/done, счётчик диалога, override окна.

        Возвращает True, если из файла подхвачен режим dialog (вызывающий код
        анонсирует это после создания агента). Файлу не доверяем: каждое
        значение идёт через params.set() — валидация choices достаётся бесплатно,
        — а не подходящее значение предупреждает и пропускается, а не падает.

        `check_explicit` — только для старта: явный флаг --mode/--done/
        --context-limit бьёт файл. При переключении в существующую сессию
        (`/session`, `/new`) explicit не проверяется: это «продолжить тот
        разговор как оставили», а флаг был про старт этого запуска.
        """
        resumed_dialog = False
        for name in ("mode", "done"):
            if check_explicit and name in self.explicit:
                continue
            value = session.state.get(name)
            if not isinstance(value, str):
                continue
            try:
                self.config.params.set(name, value)
            except ParamError as error:
                console.warn(
                    f"в файле сессии {session.name} негодное {name}={value!r} — игнор: {error}"
                )
                continue
            if name == "mode" and value == "dialog":
                resumed_dialog = True
        turns = session.state.get("dialog_turns")
        # bool в Python — это int: True прошёл бы проверку и напечатался как
        # «ход True из 10», поэтому исключаем его явно.
        self.dialog_turns = (
            turns if isinstance(turns, int) and not isinstance(turns, bool) and turns >= 0 else 0
        )
        # Три исхода, не два. Ключа нет вовсе — файлы сессий тегов w02d06/
        # w02d07, где context_limit никогда не писался: молчим и оставляем
        # текущее значение, та же дисциплина, что у mode/done выше. Ключ есть
        # и None — это ЧЕСТНЫЙ «override снят» (именно так его пишет
        # _save_state), и не различать это с «ключа нет» значило бы, что
        # override одной сессии переживает переключение на сессию, где
        # override явно снят.
        if (
            not (check_explicit and "context_limit" in self.explicit)
            and "context_limit" in session.state
        ):
            raw_limit = session.state["context_limit"]
            # None и int-не-bool идут через ту же set(): "none"/значение
            # разбирает один код, а не два параллельных пути.
            if raw_limit is None or (
                isinstance(raw_limit, int) and not isinstance(raw_limit, bool)
            ):
                try:
                    self.config.params.set("context_limit", raw_limit)
                except ParamError as error:
                    console.warn(
                        f"в файле сессии {session.name} негодное "
                        f"context_limit={raw_limit!r} — игнор: {error}"
                    )
            else:
                # bool (True/False) и прочий мусор — не лимит окна и не
                # «снято»: isinstance(int) молча пропускал бы True без
                # единого слова (bool — подкласс int в Python).
                console.warn(
                    f"в файле сессии {session.name} негодное context_limit={raw_limit!r} — игнор"
                )
        if (
            not (check_explicit and "context_strategy" in self.explicit)
            and "context_strategy" in session.state
        ):
            raw_strategy = session.state["context_strategy"]
            if isinstance(raw_strategy, str):
                try:
                    self.config.params.set("context_strategy", raw_strategy)
                except ParamError as error:
                    console.warn(
                        f"в файле сессии {session.name} негодное "
                        f"context_strategy={raw_strategy!r} — игнор: {error}"
                    )
            else:
                console.warn(
                    f"в файле сессии {session.name} негодное "
                    f"context_strategy={raw_strategy!r} — игнор"
                )
        self._apply_session_summary(session)
        self._apply_session_facts(session)
        self._apply_session_task(session)
        self._apply_session_invariants(session)
        return resumed_dialog

    def _apply_session_task(self, session: Session) -> None:
        """Load the optional task independently from the rest of session state."""
        self.task = None
        self.task_needs_healing = False
        if TASK_STATE_KEY not in session.state:
            return
        raw = session.state.get(TASK_STATE_KEY)
        try:
            task = TaskState.from_json(raw)
            task.validate_privacy(_task_configured_secrets(self))
        except TaskStateError as error:
            session.state.pop(TASK_STATE_KEY, None)
            self.task_needs_healing = True
            console.warn(f"в сессии {session.name} task повреждена — отключена ({error})")
            return
        self.task = task
        status = "paused" if task.paused else task.phase
        console.note(f"подхвачена task: {status} · {task.current_step}")

    def _apply_session_invariants(self, session: Session) -> None:
        """Load all-or-nothing policy state without breaking a valid Session."""
        self.invariants = InvariantSet()
        self.invariants_needs_healing = False
        if INVARIANTS_STATE_KEY not in session.state:
            return
        try:
            self.invariants = InvariantSet.from_json(
                session.state[INVARIANTS_STATE_KEY],
                configured_secrets=_task_configured_secrets(self),
            )
        except InvariantError as error:
            self.invariants_needs_healing = True
            console.warn(f"в сессии {session.name} invariants повреждены — отключены ({error})")
            return
        if self.invariants.rules:
            console.note(f"подхвачены invariants: {len(self.invariants.rules)}")

    def _apply_session_summary(self, session: Session) -> None:
        """Loads the summary and rewinds the working history to its boundary.

        The session file stores the FULL conversation (the growth table and
        token sums depend on that), so the summary alone isn't enough: without
        a coverage boundary a restart would send the model both the summary
        and the summarized part at once — doubling what compaction had just
        shrunk.

        Called after the caller has put the session's full history into
        self.history, and trims it. No explicit flag here: the summary isn't
        a run setting, there's nothing to override it from the command line.
        """
        raw_summary = session.state.get("summary")
        if not isinstance(raw_summary, str) or not raw_summary.strip():
            # Key absent (session files tagged w02d06-w02d08) or empty —
            # no summary, leave history alone.
            self.summary = None
            return

        raw_upto = session.state.get("summary_upto")
        # bool excluded explicitly: True is an int in Python, and "1 message
        # covered" would come out of a value that isn't actually a boundary.
        if not isinstance(raw_upto, int) or isinstance(raw_upto, bool):
            # A summary whose boundary is missing or of the wrong type used to
            # fall back to 0 — the one outcome this method exists to prevent:
            # the summary AND the whole history it summarizes in the same
            # request. Dropping the summary costs one compaction; keeping it
            # without a boundary doubles the context silently.
            console.warn(
                f"в сессии {session.name} есть пересказ, но граница summary_upto={raw_upto!r} "
                "негодная — пересказ отброшен, история идёт целиком"
            )
            self.summary = None
            return
        upto = raw_upto

        if not 0 <= upto <= len(session.turns):
            # Boundary doesn't match the history: the file was hand-edited or
            # turns were partially cleared. A summary rewound to the wrong
            # spot is quieter and worse than a lost one — it substitutes
            # someone else's content for the conversation.
            console.warn(
                f"в сессии {session.name} пересказ покрывает {upto} сообщений при "
                f"{len(session.turns)} в истории — пересказ отброшен, история идёт целиком"
            )
            self.summary = None
            return

        self.summary = raw_summary
        self.history = session.history()[upto:]
        console.note(
            f"подхвачен пересказ: {upto} старых сообщений заменены им, "
            f"в рабочей истории осталось {len(self.history)}"
        )

    def _apply_session_facts(self, session: Session) -> None:
        """Loads facts/facts_pinned/facts_upto — three cases each (SPEC §8).

        `facts_upto` is the one place this differs from `_apply_session_summary`:
        garbage there falls back to `len(session.turns)`, not 0. Zero would
        mean "nothing extracted yet", and the next turn would pay for one
        extractor call over the ENTIRE history — the exact re-extraction cost
        a corrupted key must not trigger. `/facts backfill` is the explicit,
        opt-in way to pay that cost.
        """
        raw_facts = session.state.get("facts")
        if "facts" not in session.state:
            self.facts = {}
        elif isinstance(raw_facts, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw_facts.items()
        ):
            cleaned: dict[str, str] = {}
            rejected: list[str] = []
            for key, value in raw_facts.items():
                try:
                    validate_key(key)
                except ValueError:
                    rejected.append(key)
                    continue
                cleaned[key] = value
            self.facts = cleaned
            if rejected:
                console.warn(
                    f"в сессии {session.name} отброшены ключи facts вне категорий: "
                    f"{', '.join(rejected)}"
                )
        else:
            console.warn(f"в сессии {session.name} facts повреждены — начинаем с пустого блока")
            self.facts = {}

        raw_pinned = session.state.get("facts_pinned")
        if "facts_pinned" not in session.state:
            self.facts_pinned = []
        elif isinstance(raw_pinned, list) and all(isinstance(k, str) for k in raw_pinned):
            self.facts_pinned = [key for key in raw_pinned if key in self.facts]
        else:
            console.warn(f"в сессии {session.name} facts_pinned повреждены — закрепления сброшены")
            self.facts_pinned = []

        raw_upto = session.state.get("facts_upto")
        if "facts_upto" not in session.state:
            self.facts_upto = 0
        elif (
            isinstance(raw_upto, int)
            and not isinstance(raw_upto, bool)
            and 0 <= raw_upto <= len(session.turns)
        ):
            self.facts_upto = raw_upto
        else:
            self.facts_upto = len(session.turns)
            console.warn(
                f"в сессии {session.name} негодный facts_upto={raw_upto!r} — считаем "
                f"догон полным ({self.facts_upto}) вместо повторного извлечения по всей "
                "истории; пересобрать заново: /facts backfill"
            )

    def save(self) -> bool:
        """Сохраняет сессию; ошибку записи показывает, а не глотает.

        Session.save() специально не глотает OSError (в отличие от
        journal.log_call, где лог — побочный эффект): потерянная сессия это
        потерянный разговор. Ловить обязана CLI — здесь.
        """
        try:
            self.session.save()
        except OSError as error:
            console.warn(f"сессия не сохранена ({error}) — разговор остаётся только на экране")
            return False
        return True


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


def _summary_prompt() -> str | None:
    try:
        return SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        # Don't crash, but don't compact either: a summarizer with no
        # instruction would summarize arbitrarily, and what compaction
        # forgets doesn't come back. Agent disables compaction on None
        # itself and says so out loud (SPEC §12).
        console.warn(f"промпт суммаризатора не прочитался ({error}) — сжатие выключено")
        return None


def _facts_prompt() -> str | None:
    try:
        return FACTS_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        # Same degrade-with-a-warning shape as _summary_prompt: Agent turns
        # strategy="facts" into a plain window itself and says so.
        console.warn(f"промпт экстрактора фактов не прочитался ({error}) — блок facts не растёт")
        return None


def _memory_prompt() -> str | None:
    path = DEFAULT_SYSTEM_PROMPT.parent / "memory.md"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        console.warn(f"промпт memory extractor не прочитался ({error}) — routing выключен")
        return None


def _warn_text(text: str | None) -> None:
    if text:
        console.warn(text)


def _warn_extra_args(name: str, args: list[str]) -> None:
    """An argument-less command got arguments — say so, don't swallow it.

    A silently ignored argument reads as honored: "/compact 4" looks like a
    request to compact down to four messages, and without this line the user
    would believe it was.
    """
    if args:
        console.warn(f"{name} не принимает аргументов — «{' '.join(args)}» проигнорировано")


# --------------------------------------------------------------------------
# Ход разговора
# --------------------------------------------------------------------------


def _turn(shell: AgentShell, question: str) -> None:
    """Один ход: спросить, напечатать, посчитать, запомнить."""
    if shell.task is not None and shell.task.paused:
        console.warn("Task paused — prompt blocked; используй /task resume или /task clear")
        return
    if shell.task is not None and shell.agent.mode == "dialog":
        console.warn(
            "Task State Machine несовместима с mode=dialog — используй /mode chat либо /task clear"
        )
        return
    shell.last_question = question
    reply = _ask(shell, question, shell.history)
    if reply is None:
        return

    # reply.history — УЖЕ обрезанная история, и именно её надо отдать в
    # следующий ask(): полная запись живёт в файле сессии, а кормить агента
    # session.history() каждый ход значило бы пересчитывать обрезку заново.
    shell.history = reply.history
    # Summary holds until the next compaction and survives a restart via
    # session state (_save_state) — like the working history, it's memory.
    shell.summary = reply.summary
    if reply.facts is not None:
        shell.facts = reply.facts
        shell.facts_upto = reply.facts_upto
    if reply.memory is not None:
        shell.memory = reply.memory
        shell.memory_upto = reply.memory_upto
        if reply.memory_update is not None:
            shell._mark_memory_update(reply.memory_update)
    _remember(shell, question, reply)
    # Панель — ПОСЛЕ обновления истории и записи хода, иначе оба её числа
    # отстают на ход: «сессия» не считала бы только что полученный usage, а
    # «контекст» показывал бы окно без этого обмена, хотя обещает следующий
    # запрос.
    _token_panel(shell, reply)
    if reply.invariant_assessment is not None:
        # stderr preserves machine-readable stdout for format=json/schema.
        console.note("Invariant check: compliant")

    if shell.agent.mode == "dialog":
        _dialog_progress(shell, reply)


def _report_invariant_assessment(
    shell: AgentShell, assessment: InvariantAssessment, result: CallResult
) -> None:
    """Account and journal an internal preflight without its private payload."""
    shell.invariant_assessment_calls += 1
    shell.invariant_assessment_prompt_tokens += result.usage.prompt_tokens or 0
    shell.invariant_assessment_completion_tokens += result.usage.completion_tokens or 0
    if assessment.decision is None:
        shell.invariant_assessment_failures += 1
    log_internal_call(
        result,
        week=INVARIANT_WEEK,
        day=INVARIANT_DAY,
        kind="invariant_assessment",
        status=assessment.decision or "failed",
        extra={"rule_ids": list(assessment.rule_ids)},
    )


def _render_invariant_refusal(assessment: InvariantAssessment) -> None:
    """A local refusal: never print model JSON or fabricate an answer call."""
    if assessment.decision is None:
        console.warn(
            "Invariant check failed closed — ответ не сгенерирован: "
            f"{redact(assessment.error or 'unknown assessment error')}"
        )
        console.note("Проверь invariants через /invariant list или исправь их explicit command")
        return
    ids = ", ".join(assessment.rule_ids)
    if assessment.decision == "policy_conflict":
        console.warn(
            f"Invariant policy conflict ({ids}) — ответ не сгенерирован: "
            f"{redact(assessment.explanation)}"
        )
        console.note(
            f"Разреши конфликт explicit: /invariant remove <id> (rules: {ids}) или /invariant clear"
        )
        return
    console.warn(f"Invariant conflict ({ids}) — запрос отклонён: {redact(assessment.explanation)}")
    console.note(f"Safe alternative: {redact(assessment.safe_alternative or '')}")


def _ask(shell: AgentShell, question: str, history: list[chat_core.Message]) -> AgentReply | None:
    """Вызов агента плюс весь вывод вокруг него. None — вызов не состоялся."""
    streaming = chat_core.should_stream(shell.config)
    facts_before = dict(shell.facts)
    try:
        reply = shell.agent.ask(
            question,
            history,
            summary=shell.summary,
            facts=shell.facts,
            facts_pinned=shell.facts_pinned,
            facts_upto=shell.facts_upto,
            memory=shell._memory_snapshot(),
            memory_upto=shell.memory_upto,
            memory_history=shell.session.history(),
            profile=shell.profile_values,
            task=shell.task,
            invariants=shell.invariants,
            on_chunk=console.write_chunk if streaming else None,
        )
    except AdventError as error:
        # Compaction/facts, if either happened, happened BEFORE the failure —
        # so both are salvaged and reported first, in the order things
        # actually occurred.
        _salvage_compaction(shell)
        _salvage_facts(shell, facts_before)
        _salvage_facts_failure(shell)
        _salvage_memory(shell)
        _salvage_memory_failure(shell)
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

    if reply.invariant_assessment is not None and reply.invariant_result is not None:
        _report_invariant_assessment(shell, reply.invariant_assessment, reply.invariant_result)
        if reply.invariant_assessment.blocked:
            _render_invariant_refusal(reply.invariant_assessment)
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
    if reply.window_dropped != shell.window_dropped_reported:
        if reply.window_dropped:
            # The "window"/"facts" strategy's own deliberate forgetting —
            # distinct from the budget-trim safety net reported below
            # (SPEC §4). Only on CHANGE (review finding T1): printing the
            # same count every turn once the window fills turns a one-time
            # warning into noise nobody reads.
            console.note(f"окно: выпало {reply.window_dropped} сообщений")
        shell.window_dropped_reported = reply.window_dropped
    if reply.facts_update is not None:
        _report_facts_update(shell, reply.facts_update, before=facts_before)
    if reply.facts_failed is not None:
        # Paid but unusable (truncated/invalid delta) — Agent already warned
        # WHY aloud; this only accounts for the cost (review finding #5).
        _report_facts_failure(shell, reply.facts_failed)
    if reply.memory_update is not None:
        _report_memory_update(shell, reply.memory_update)
    if reply.memory_failed is not None:
        _report_memory_failure(shell, reply.memory_failed)
    if reply.compaction is not None:
        # BEFORE the trim warning: compaction happens earlier, and the
        # on-screen order should match reality — otherwise it reads as if
        # the summary replaced what trimming had already dropped.
        _report_compaction(shell, reply.compaction)
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

    console.footer(reply.result, completion_estimate=_completion_estimate(shell, reply))
    # reconcile() — чистая функция: калибровку счётчика уже сделал агент, здесь
    # только показ. Сверять надо с тем, что РЕАЛЬНО ушло в API (sent_messages):
    # слой формата дописывает инструкцию внутри chat._payload(), и сверка «до
    # слоя» дала бы стабильную ложную дельту.
    shell.last_check = tokens.reconcile(
        shell.counter, reply.result.sent_messages or [], reply.result.usage.prompt_tokens
    )
    log_call(
        reply.result,
        _journal_messages(
            reply.result.sent_messages,
            task=shell.task,
            profile=shell.profile_values,
            invariants=shell.invariants,
        )
        or [{"role": "user", "content": question}],
        week=WEEK,
        day=DAY,
    )
    return reply


def _journal_messages(
    messages: list[chat_core.Message] | None,
    *,
    task: TaskState | None = None,
    profile: dict[str, str] | None = None,
    invariants: InvariantSet | None = None,
) -> list[chat_core.Message]:
    """Exclude generated private context without deleting lookalike user text."""
    if not messages:
        return []
    task_pair = task_messages(task)
    synthetic_indices: set[int] = set()
    protected_index = 1 if messages[0].get("role") == "system" else 0
    invariant_pair = invariant_messages(invariants)
    if (
        invariant_pair
        and [dict(message) for message in messages[protected_index:]][: len(invariant_pair)]
        == invariant_pair
    ):
        synthetic_indices.update(range(protected_index, protected_index + len(invariant_pair)))
        protected_index += len(invariant_pair)
    profile_pair = profiles.messages(profile) if profile else []
    if (
        profile_pair
        and [dict(message) for message in messages[protected_index:]][: len(profile_pair)]
        == profile_pair
    ):
        protected_index += len(profile_pair)
    if (
        task_pair
        and [dict(message) for message in messages[protected_index:]][: len(task_pair)] == task_pair
    ):
        synthetic_indices.update(range(protected_index, protected_index + len(task_pair)))

    kept: list[chat_core.Message] = []
    for index, message in enumerate(messages):
        if index in synthetic_indices:
            continue
        content = str(message.get("content", ""))
        if not content.startswith("Профиль пользователя (учитывай") and not content.startswith(
            "Профиль учтён"
        ):
            kept.append(message)
    return kept


def _salvage_compaction(shell: AgentShell) -> None:
    """Keeps a compaction whose turn then failed.

    The summarizer call is a separate, already paid call. When the turn's own
    call fails afterwards, the compaction never reaches the reply — and
    without this it would be lost whole: no `kind=compact` row in the journal,
    no line in `/tokens` (against §7, where every before/after comparison
    includes the price of compacting), and a second summarizer call on the
    next attempt, since the history it was built from would be unchanged.
    Applying it here is what makes the retry cheap instead of paid twice.
    """
    compaction = shell.agent.take_pending_compaction()
    if compaction is None:
        return
    _report_compaction(shell, compaction)
    shell.summary = compaction.summary
    shell.history = list(compaction.tail)
    # Straight to disk, for the same reason `/compact` does it: the summary is
    # memory, and the turn that would have saved it did not happen.
    _save_state(shell)


def _report_compaction(shell: AgentShell, compaction: Compaction) -> None:
    """Announces the compaction out loud and logs the service call.

    Gain and cost in one line. "Request got 4940 tokens lighter" without the
    second number is a profit report that never names the cost: the summary
    came from a model call, that call has usage, and it's paid for (SPEC §7).
    """
    usage = compaction.result.usage
    shell.compact_calls += 1
    # `or 0`: usage may not come back at all, undercounting this compaction's
    # cost. Zero here isn't "free", it's "server didn't say" — the nearby
    # "summary itself cost —/—" line says that.
    shell.compact_prompt += usage.prompt_tokens or 0
    shell.compact_completion += usage.completion_tokens or 0

    saved = compaction.saved()
    gain = f" (−{saved})" if saved is not None else ""
    console.note(
        f"история сжата: {compaction.covered} старых сообщений заменены пересказом; "
        f"запрос {_num(compaction.tokens_before)} → {_num(compaction.tokens_after)}"
        f"{gain} токенов; сам пересказ стоил "
        f"{_num(usage.prompt_tokens)}/{_num(usage.completion_tokens)}"
    )
    log_call(
        compaction.result,
        compaction.result.sent_messages or [],
        week=WEEK,
        day=DAY,
        # kind=compact distinguishes a service call from a conversation turn.
        # Without it any journal-based count — day 08's growth table
        # included — would count compaction as a regular turn, and its
        # prompt (the old history!) would skew exactly the curve day 08
        # shows.
        extra={"kind": "compact", "covered": compaction.covered},
    )


def _salvage_facts(shell: AgentShell, before: dict[str, str]) -> None:
    """Keeps a paid facts update whose turn then failed. Mirrors _salvage_compaction.

    `facts_upto` after salvage is `len(shell.history)`, not the cursor
    `_run_facts` computed: the failed turn's question never entered history
    (the exchange is not recorded), so the only truthful claim is "facts
    reflect everything currently in history" — nothing about a turn that
    never happened.
    """
    update = shell.agent.take_pending_facts()
    if update is None:
        return
    _report_facts_update(shell, update, before=before)
    shell.facts = update.facts
    shell.facts_upto = len(shell.history)
    _save_state(shell)


def _salvage_facts_failure(shell: AgentShell) -> None:
    """Keeps a paid-but-unusable extractor call whose turn then failed.

    Mirrors _salvage_compaction/_salvage_facts: the call already happened and
    was already billed, distinct from the successful-update case above — only
    one of the two can happen per turn. Without this it would vanish with no
    journal row at all, under-reporting the day's spend (review finding #5).
    """
    failure = shell.agent.take_pending_facts_failed()
    if failure is None:
        return
    _report_facts_failure(shell, failure)


def _salvage_memory(shell: AgentShell) -> None:
    update = shell.agent.take_pending_memory()
    if update is None:
        return
    persisted_turns = len(shell.session.turns)
    _mark = shell._mark_memory_update
    _mark(update)
    # The extractor includes the failed turn's user message in its own
    # coordinate space. That message was never recorded, so it must not be
    # persisted as covered by working memory.
    shell.memory_upto = min(shell.memory_upto, persisted_turns)
    _report_memory_update(shell, update)
    shell._save_memory()


def _salvage_memory_failure(shell: AgentShell) -> None:
    failure = shell.agent.take_pending_memory_failed()
    if failure is None:
        return
    _report_memory_failure(shell, failure)


def _report_memory_update(shell: AgentShell, update: MemoryUpdate) -> None:
    usage = getattr(update.call_result, "usage", None)
    shell.memory_calls += 1
    if usage is not None:
        shell.memory_prompt_tokens += usage.prompt_tokens or 0
        shell.memory_completion_tokens += usage.completion_tokens or 0
    note = render_delta(update)
    if note:
        console.note(note)
    for layer, _operation, _reason in update.blocked:
        console.note(f"memory: {layer} — pinned value сохранено")
    log_call(
        update.call_result,
        getattr(update.call_result, "sent_messages", None) or [],
        week=3,
        day=11,
        extra={
            "kind": "memory",
            "covered": update.memory_upto,
            "applied": len(update.applied),
            "blocked": len(update.blocked),
            "rejected": len(update.rejected),
        },
    )


def _report_memory_failure(shell: AgentShell, failure: MemoryFailure) -> None:
    shell.memory_calls += 1
    shell.memory_failures += 1
    usage = getattr(failure.call_result, "usage", None)
    if usage is not None:
        shell.memory_prompt_tokens += usage.prompt_tokens or 0
        shell.memory_completion_tokens += usage.completion_tokens or 0
    log_call(
        failure.call_result,
        getattr(failure.call_result, "sent_messages", None) or [],
        week=3,
        day=11,
        extra={"kind": "memory_failed", "reason": failure.reason},
    )


def _report_facts_failure(
    shell: AgentShell, failure: FactsFailure, *, backfill: bool = False
) -> None:
    """Journals a paid extractor call that came back unusable (review finding #5).

    Distinguished from a successful `kind="facts"` row by `kind="facts_failed"`
    plus the reason (truncated/invalid) — Agent already said why aloud via
    on_warning, this only makes sure the cost isn't silently absent from the
    journal or from /tokens.
    """
    shell.facts_calls += 1
    shell.facts_failures += 1
    usage = failure.result.usage
    shell.facts_prompt_tokens += usage.prompt_tokens or 0
    shell.facts_completion_tokens += usage.completion_tokens or 0
    extra: dict[str, object] = {"kind": "facts_failed", "reason": failure.reason}
    if backfill:
        extra["backfill"] = True
    log_call(
        failure.result,
        failure.result.sent_messages or [],
        week=WEEK,
        day=DAY,
        extra=extra,
    )


def _report_facts_update(
    shell: AgentShell, update: FactsUpdate, *, before: dict[str, str], backfill: bool = False
) -> None:
    """Announces the extractor call and logs it as a service call (SPEC §5, §9)."""
    shell.facts_calls += 1
    usage = update.result.usage
    shell.facts_prompt_tokens += usage.prompt_tokens or 0
    shell.facts_completion_tokens += usage.completion_tokens or 0

    note = render_note(before, update.facts)
    if note:
        console.note(note)
    for key in update.delta.blocked:
        # A pinned key the extractor tried to touch: the user's own edit wins
        # and stays untouched — SPEC-w02d10.md §5.4's own example wording.
        _, _, name = key.partition(".")
        console.note(f"facts: {name} — правка пользователя сохранена")
    # update.truncated is already spoken aloud by the agent itself
    # (self._warn in _run_facts, via on_warning=console.warn) — repeating it
    # here would just be an echo.
    extra: dict[str, object] = {"kind": "facts", "covered": update.covered}
    if backfill:
        extra["backfill"] = True
    log_call(
        update.result,
        update.result.sent_messages or [],
        week=WEEK,
        day=DAY,
        extra=extra,
    )


def _remember(shell: AgentShell, question: str, reply: AgentReply) -> None:
    """Кладёт обмен в память сессии и сохраняет её на диск.

    Текст ответа берётся из reply.history[-1], а НЕ из reply.text: при
    прерывании стрима агент дописывает в историю пометку об обрыве, и в файле
    сессии она должна быть — иначе следующий запуск подсунет модели её же
    оборванный ответ как законченный (SPEC-w02d06.md §12).

    Goes through `_save_state()`, not a bare `shell.save()`: since day 09 the
    state holds the summary, which is MEMORY, not a setting. A saved turn
    without a saved summary would mean history goes in full after a restart
    as if compaction never happened — the day would only work until the
    first exit.
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
    # Agent's cursor is measured before this exchange is recorded. After a
    # successful memory update, the working file must cover the full durable
    # transcript, including both newly recorded messages.
    if reply.memory_update is not None:
        shell.memory_upto = len(shell.session.turns)
        shell.memory_dirty_working = True
    _save_state(shell)


def _save_state(shell: AgentShell) -> None:
    """Снимок режима и счётчика диалога — в файл сессии.

    Save при КАЖДОМ изменении, а не только после хода: иначе `/mode dialog` →
    `/exit` без единого хода терял бы режим — в файле остался бы прежний
    слепок, и перезапуск не продолжил бы диалог «как будто не выключался».
    """
    shell.session.state["mode"] = shell.agent.mode
    shell.session.state["done"] = shell.config.params.done
    shell.session.state["dialog_turns"] = shell.dialog_turns
    # None — честное «карточка»: перечитывание файла вернёт лимит из карточки,
    # и ключ не нужно удалять, достаточно пустого значения.
    shell.session.state["context_limit"] = shell.config.params.context_limit
    shell.session.state["active_profile"] = shell.active_profile
    # Summary and its coverage boundary are conversation CONTENT, not a
    # setting: Session.clear() wipes them along with the turns
    # (CONTENT_STATE_KEYS), unlike mode/done/context_limit above.
    #
    # summary_upto — how many of the session file's leading messages are
    # outside the working history. Computed as a difference, not a separate
    # counter: working history is always a suffix of the session record
    # (both trimming and compaction only remove from the front), so a
    # difference can't drift from fact the way a counter could. Turns
    # dropped by trimming count here alongside compacted ones, which is
    # correct — a restart isn't obligated to revive what already fell out of
    # the working context.
    shell.session.state["summary"] = shell.summary
    shell.session.state["summary_upto"] = max(0, len(shell.session.turns) - len(shell.history))
    # Day 10 (SPEC §8): context_strategy is a setting (like mode/context_limit
    # above); facts/facts_pinned/facts_upto are CONTENT, dropped by
    # session.clear() (CONTENT_STATE_KEYS) the same way summary is.
    shell.session.state["context_strategy"] = shell.config.params.context_strategy
    shell.session.state["facts"] = shell.facts
    shell.session.state["facts_pinned"] = list(shell.facts_pinned)
    shell.session.state["facts_upto"] = shell.facts_upto
    if shell.task is None:
        shell.session.state.pop(TASK_STATE_KEY, None)
    else:
        shell.session.state[TASK_STATE_KEY] = shell.task.to_json()
    if shell.invariants.rules:
        shell.session.state[INVARIANTS_STATE_KEY] = shell.invariants.to_json()
    else:
        # Missing is the canonical empty form.  It both survives /new and
        # makes `/invariant clear` an explicit, durable removal.
        shell.session.state.pop(INVARIANTS_STATE_KEY, None)
    # Structured Day 11 layers have independent atomic files.  Save them
    # before the transcript so a successful session write never falsely claims
    # that a dirty memory layer was persisted.
    shell._save_memory()
    if shell.save():
        shell.task_needs_healing = False
        shell.invariants_needs_healing = False


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
        # Слепок пишем ПОСЛЕ flip'а: save уже случился в _remember ДО этой
        # функции, и без отдельного save файл при выходе сразу после done
        # хранил бы протухший режим dialog — перезапуск воскресил бы
        # законченный эпизод.
        _save_state(shell)
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
    _save_state(shell)


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
        "",
        system=shell.agent.system_prompt(),
        history=[*invariant_messages(shell.invariants), *shell.history],
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
    if shell.config.params.context_limit is not None:
        # Override (SPEC-w02d08.md §4): процент заполнения при заниженном
        # лимите вторичен, важно само слово — иначе «450/2500» после недель
        # жизни с честным окном читается как поломка карточки.
        return f"контекст {mark}{used}/{limit} (override)"
    return f"контекст {mark}{used}/{limit} ({used * 100 // limit}%)"


def _session_tokens_label(shell: AgentShell) -> str:
    total = shell.session.token_total()
    label = "—" if total is None else str(total)
    missing = shell.session.missing_usage()
    if missing:
        # Без этой пометки неполная сумма выглядит полной.
        label += f" (без usage: {missing})"
    return label


def _completion_estimate(shell: AgentShell, reply: AgentReply) -> int | None:
    """Локальная оценка ответа, когда сервер не прислал completion_tokens.

    LM Studio в стриме usage не присылает вовсе (TODO №2), и футер деградировал
    бы в «?». Оценка всегда с тильдой — её ставит footer (SPEC-w02d08.md §6);
    счётчика нет — честный «?», притворяться нечем.

    Текст ответа кодируется формой role=user, а не role=assistant: точный
    токенизатор (MistralCounter) отказывается считать сообщение с
    role=assistant вовсе — count() возвращает None (тот же отказ, что уже
    задокументирован у _next_context_tokens), и на модели с точным счётом
    оценка ответа молча деградировала бы в «?» ровно там, где SPEC §6 обещает
    «~N». Оверхед шаблона чата (system-токены, спецсимволы) вычитается тем же
    приёмом, что и в _next_context_tokens, — иначе он считался бы частью
    длины ответа.
    """
    if reply.result.usage.completion_tokens is not None:
        return None
    if shell.counter is None:
        return None
    with_text = shell.counter.count([{"role": "user", "content": reply.text}])
    overhead = shell.counter.count([{"role": "user", "content": ""}])
    if with_text is None or overhead is None:
        return None
    return max(0, with_text - overhead)


def _branch_label(shell: AgentShell) -> str | None:
    """Own segment name if the current session is a branch, else None."""
    parent, segment, _is_checkpoint = parse_name(shell.session.name)
    return segment if parent is not None else None


def _token_panel(shell: AgentShell, reply: AgentReply) -> None:
    """Панель токенов после каждого хода — в stderr, как и весь не-продукт.

    Три числа, и они отвечают на три разных вопроса: сколько стоил этот ход
    (факт с сервера), сколько стоила сессия целиком и сколько займёт следующий
    запрос ДО отправки (SPEC-w02d06.md §7.4). Day 10 (SPEC §11) adds two more
    segments — the strategy in effect always, the branch name only if there
    is one.
    """
    usage = reply.result.usage
    turn = f"{_num(usage.prompt_tokens)}/{_num(usage.completion_tokens)}"
    parts = [
        f"токены · ход {turn}",
        f"сессия {_session_tokens_label(shell)}",
        _context_label(shell),
        f"стратегия {shell.agent.context_strategy}",
    ]
    branch = _branch_label(shell)
    if branch:
        parts.append(f"ветка {branch}")
    console.note(" · ".join(parts))


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
    "текущая модель; без имени — выбор из списка (интерактивно), "
    "с именем — переключить, list all — вообще все модели",
    # Без квадратных скобок: console.commands_help кладёт строку в таблицу
    # Rich, а Rich-markup молча съедает «[all]» как незакрытый тег — ровно
    # этот кусок подписи и пропал бы с экрана.
    usage="/model <имя> | list | list all | info",
)
def _cmd_model(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        # Пикер — только при интерактивном stdin: запись демо гоняет REPL через
        # pipe, и typer.prompt пикера съел бы следующую строку сценария как
        # «ответ» и порвал бы кадр (та же причина, по которой /set пикерный
        # только при isatty). Не-tty и пустой список — прежняя печать.
        if sys.stdin.isatty() and shell.models:
            candidates = chat_models(shell.models) or shell.models
            chosen = console.choose_model(
                candidates,
                current=shell.config.model,
                base_url=shell.config.base_url,
            )
            if chosen is None:
                console.note("модель не изменена")
                return False
            return _switch_model(shell, chosen)
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

    return _switch_model(shell, sub)


def _switch_model(shell: AgentShell, name: str) -> bool:
    """Общий хвост `/model <имя>` и пикерной голой `/model`.

    Одно место для retarget/rollback и предупреждения о skipped-параметрах —
    иначе голая /model и /model с именем разошлись бы на первой же правке.
    """
    previous = shell.config.model
    shell.config.model = name
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
    console.note(f"модель переключена на {name}" + (f" → {resolved}" if resolved else ""))
    _, skipped = shell.config.params.as_payload(shell.capabilities)
    if skipped:
        console.warn(f"{name} не поддерживает: {', '.join(skipped)} — параметры не отправляются")
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
    "изменить параметр (без аргументов — выбор из списка, интерактивно; "
    "default — вернуть умолчание команды)",
    usage="/set <параметр> <значение>",
)
def _cmd_set(shell: AgentShell, args: list[str]) -> bool:
    # Пикер — только при интерактивном stdin и голой команде: запись демо
    # гоняет REPL через pipe, и промпт пикера съел бы следующую строку
    # сценария как «ответ». Не-tty или неполная команда — прежнее предупреждение.
    if not args and sys.stdin.isatty():
        picked = _pick_param(shell)
        if picked is None:
            console.note("отменено — параметры не изменены")
            return False
        return _apply_set(shell, *picked)
    if len(args) < 2:
        console.warn("нужно: /set <параметр> <значение>. /params — что есть")
        return False
    return _apply_set(shell, args[0], " ".join(args[1:]))


def _pick_param(shell: AgentShell) -> tuple[str, str] | None:
    """Двухшаговый пикер для голой /set: сначала параметр, потом значение.

    Подписи берём из params.describe(AGENT_PARAMS) — тот же источник, что у
    /params: пикер не должен показывать параметры, которых агент не читает.
    None на любом шаге — отмена без изменений.
    """
    described = shell.config.params.describe(AGENT_PARAMS)
    options = [f"{name} = {value} — {help_text}" for name, value, help_text in described]
    index = console.choose("параметр", options)
    if index is None:
        return None
    name, current, _ = described[index]

    spec = BY_NAME[name]
    if spec.choices:
        # Выбор из фиксированного списка — тоже пикером: свободным вводом
        # сюда нечего вводить, кроме опечатки. «default» передаётся как есть —
        # реестр сам вернёт умолчание команды.
        choice_options = [*spec.choices, "default"]
        choice_index = console.choose(name, choice_options)
        if choice_index is None:
            return None
        return name, choice_options[choice_index]

    try:
        raw = typer.prompt(
            f"{name} (сейчас: {current}; default — вернуть умолчание)",
            default="",
            show_default=False,
            err=True,
        )
    except (EOFError, typer.Abort, KeyboardInterrupt):
        return None
    value = raw.strip()
    if not value:
        return None
    return name, value


def _maybe_backfill_facts(shell: AgentShell, previous_strategy: str) -> None:
    """Auto-backfill on switching TO facts with turns already on disk (SPEC §5.6).

    Fires from whatever code path actually changes the EFFECTIVE strategy to
    "facts" — called from both `/set context_strategy facts` and the `compact`
    alias's own params.set(), so a future alias spelling can't reintroduce the
    review's C3 gap by forgetting a duplicate check (there's only ever one).
    Without this, switching mid-conversation looks like amnesia: the agent
    suddenly has no memory of a conversation it's still having. `/facts
    backfill` (`_facts_backfill`) is the explicit form; this calls the exact
    same function so the two never drift.
    """
    if previous_strategy == "facts" or shell.agent.context_strategy != "facts":
        return
    if shell.facts or not shell.session.turns:
        return
    console.note("context_strategy → facts: сессия не пуста, а facts пуст — авто-backfill")
    _facts_backfill(shell)


def _apply_set(shell: AgentShell, name: str, raw: str) -> bool:
    """Общий хвост `/set <name> <value>` и двухшагового пикера.

    Одно место для валидации, apply_defaults и веток session/mode/done — иначе
    пикерная и аргументная формы /set разошлись бы на первой же правке.
    """
    if name not in AGENT_PARAMS:
        console.warn(
            f"агент не читает параметр {name!r} — /params показывает те, что читает. "
            "Выставленный параметр, ни на что не влияющий, выглядит как поломка"
        )
        return False

    if name == "mode" and raw.strip().lower() == "dialog" and shell.task is not None:
        console.warn("mode=dialog несовместим с retained task — сначала /task clear")
        return False

    # Captured BEFORE .set() mutates it: _maybe_backfill_facts needs to know
    # what the strategy was to tell "just switched to facts" from "already
    # was facts" (SPEC §5.6, review finding C3).
    previous_strategy = shell.agent.context_strategy if name == "context_strategy" else None
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
    if name == "context_limit":
        # Значение применилось — сразу пересчитываем лимит агента: trim идёт
        # по нему уже со следующего хода, а «вступит после перезапуска»
        # читалось бы как поломка.
        shell.agent.context_limit = shell.effective_context_limit()
    if name == "context_strategy":
        # "default"/"none"/"" is params.set()'s own reset sentinel — treated
        # as "not explicit" so `/set compact off|on` regains the alias.
        shell.context_strategy_explicit = raw.strip().lower() not in ("", "none", "default")
        assert previous_strategy is not None  # name == "context_strategy" set it above
        _maybe_backfill_facts(shell, previous_strategy)
    if name == "compact" and value is not None:
        _apply_compact_alias(shell, bool(value))
    if name in ("mode", "done", "stop"):
        # Негодный или съедаемый stop'ом маркер надо поймать сейчас, а не
        # тогда, когда диалог не закончится ни разу.
        _warn_text(shell.agent.check_done())
    if name in ("mode", "done", "context_limit", "context_strategy", "compact"):
        # Значение применилось — сразу в файл: `/mode dialog` → `/exit` без
        # хода иначе терял бы режим (save после хода тут не случится). Точка
        # записи context_limit — та же (SPEC-w02d08.md §4).
        _save_state(shell)

    _, skipped = shell.config.params.as_payload(shell.capabilities)
    if name in skipped:
        console.warn(f"{shell.config.model} не поддерживает {name} — параметр не отправляется")
    return False


def _apply_compact_alias(shell: AgentShell, enabled: bool) -> None:
    """`/set compact off|on` maps onto context_strategy (SPEC-w02d10.md §3).

    Skipped once context_strategy was set explicitly this run — an explicit
    choice loses to nothing, compact included. "Explicit" is tracked only
    within this process (see AgentShell.context_strategy_explicit): day 09's
    own demo and README call `/set compact off` on a freshly started REPL and
    expect it to behave exactly as before, restart after restart —
    persisting explicitness across a save/load cycle would break that on the
    very first resume.
    """
    if shell.context_strategy_explicit:
        console.warn(
            f"context_strategy={shell.agent.context_strategy} уже выбран явно — "
            "compact на сборку запроса не влияет, им управляет /set context_strategy"
        )
        return
    target = "summary" if enabled else "window"
    previous_strategy = shell.agent.context_strategy
    if previous_strategy == target:
        return
    shell.config.params.set("context_strategy", target)
    console.note(f"context_strategy → {target} (алиас /set compact {'on' if enabled else 'off'})")
    # target is never "facts" today (on|off maps only to summary|window), but
    # routed through the one shared check anyway — review finding C3 asked
    # for a single place that can't be bypassed by a future alias spelling.
    _maybe_backfill_facts(shell, previous_strategy)


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


@command(
    "/strategy",
    "стратегия контекста: без значения — текущая и список, со значением — переключить",
    usage="/strategy window | facts | branch | summary",
)
def _cmd_strategy(shell: AgentShell, args: list[str]) -> bool:
    """Shorthand for `/set context_strategy` — the day's switch under the day's name.

    Delegates like /mode: parsing, validation and the side effects of a switch
    (auto-backfill, the note, saving the state) stay in one place, so
    `/strategy fcts` and `/set context_strategy fcts` cannot answer differently.
    """
    if not args:
        choices = ", ".join(CONTEXT_STRATEGY_CHOICES)
        console.note(f"стратегия контекста: {shell.agent.context_strategy} (есть: {choices})")
        return False
    return _cmd_set(shell, ["context_strategy", *args])


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
    shell.summary = None
    # Short-term is the in-process view of history; working/long-term are
    # durable structured layers and intentionally survive /reset.
    shell.memory = MemorySnapshot(ShortTermMemory(), shell.memory.working, shell.memory.long_term)
    shell.dialog_turns = 0
    # Счётчик обнулился — сразу и в файл: иначе перезапуск воскресил бы его
    # из протухшего слепка state.
    _save_state(shell)
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
    path = Session.path_for(validate_name(name), shell.directory)
    if path.is_file():
        peek = Session.load(name, directory=shell.directory, quarantine=False)
        if peek.state.get("kind") == "checkpoint":
            console.warn(
                f"{name!r} — checkpoint, /new его не стирает: /branch <имя> --from {name} "
                "ответвится от него"
            )
            return False
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


@command(
    "/checkpoint",
    "снимок сессии целиком (можно только ответвиться от него, не /switch); "
    "без имени — список checkpoint'ов этой сессии",
    usage="/checkpoint [имя]",
)
def _cmd_checkpoint(shell: AgentShell, args: list[str]) -> bool:
    infos, warnings = list_sessions(shell.directory)
    for warning in warnings:
        console.warn(warning)
    if not args:
        entries = [
            entry
            for entry in build_tree([info.name for info in infos])
            if entry.is_checkpoint and entry.parent == shell.session.name
        ]
        if not entries:
            console.note(f"у сессии {shell.session.name} нет checkpoint'ов")
            return False
        for entry in entries:
            console.note(entry.name)
        return False
    name = args[0]
    _warn_extra_args("/checkpoint", args[1:])
    if not _flush_memory_for_identity_change(shell):
        return False
    _save_state(shell)
    checkpoint = make_checkpoint(shell.session, name, directory=shell.directory)
    try:
        shell.memory_store.copy_working(shell.session.name, checkpoint.name)
    except OSError as error:
        console.warn(f"working memory checkpoint не скопирована ({error})")
    console.note(f"checkpoint {checkpoint.name}: ходов {len(checkpoint.turns)}")
    return False


@command(
    "/branch",
    "ответвиться в новую сессию: <имя> [--from <checkpoint>] | удалить: --delete <имя>",
    usage="/branch <имя> [--from <checkpoint>] | --delete <имя>",
)
def _cmd_branch(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.warn("нужно: /branch <имя> [--from <checkpoint>] | --delete <имя>")
        return False

    if args[0] == "--delete":
        if len(args) < 2:
            console.warn("нужно: /branch --delete <имя>")
            return False
        name = args[1]
        _warn_extra_args("/branch --delete", args[2:])
        if name == shell.session.name:
            console.warn("нельзя удалить текущую ветку — сначала /switch на другую")
            return False
        delete_branch(name, directory=shell.directory)
        console.note(f"ветка {name} удалена")
        return False

    name = args[0]
    rest = args[1:]
    from_arg: str | None = None
    if rest and rest[0] == "--from":
        if len(rest) < 2:
            console.warn("нужно: /branch <имя> --from <checkpoint>")
            return False
        from_arg = rest[1]
        rest = rest[2:]
    _warn_extra_args("/branch", rest)

    if not _flush_memory_for_identity_change(shell):
        return False

    if from_arg is not None:
        cp_name = (
            from_arg
            if BRANCH_SEP in from_arg
            else checkpoint_file_name(shell.session.name, from_arg)
        )
        if not Session.path_for(cp_name, shell.directory).is_file():
            console.warn(f"checkpoint {cp_name!r} не найден")
            return False
        source = Session.load(cp_name, directory=shell.directory, quarantine=False)
        if source.state.get("kind") != "checkpoint":
            console.warn(f"{cp_name!r} — не checkpoint, --from ждёт снимок из /checkpoint")
            return False
    else:
        infos, warnings = list_sessions(shell.directory)
        for warning in warnings:
            console.warn(warning)
        checkpoints = [
            entry
            for entry in build_tree([info.name for info in infos])
            if entry.is_checkpoint and entry.parent == shell.session.name
        ]
        if len(checkpoints) > 1:
            console.warn(
                "у сессии несколько checkpoint'ов — укажи --from <имя>: "
                + ", ".join(entry.name for entry in checkpoints)
            )
            return False
        source = (
            Session.load(checkpoints[0].name, directory=shell.directory, quarantine=False)
            if checkpoints
            else shell.session
        )

    branch = make_branch(source, name, directory=shell.directory)
    try:
        shell.memory_store.copy_working(source.name, branch.name)
    except OSError as error:
        console.warn(f"working memory branch не скопирована ({error})")
    console.note(f"ветка {branch.name} создана от {source.name}")
    _switch_session(shell, branch.name, announce_diff=True)
    return False


@command(
    "/switch",
    "перейти в сессию или ветку (в checkpoint нельзя — подсказка ответвиться)",
    usage="/switch <имя>",
)
def _cmd_switch(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.warn("нужно: /switch <имя>")
        return False
    name = validate_name(args[0])
    _warn_extra_args("/switch", args[1:])
    if not Session.path_for(name, shell.directory).is_file():
        console.warn(f"сессия {name!r} не найдена")
        return False
    # Checkpoint refusal lives in _switch_session now — shared with /set
    # session, /new and startup (review finding P1).
    _switch_session(shell, name, announce_diff=True)
    return False


@command("/branches", "дерево веток и checkpoint'ов текущего корня")
def _cmd_branches(shell: AgentShell, args: list[str]) -> bool:
    _warn_extra_args("/branches", args)
    infos, warnings = list_sessions(shell.directory)
    for warning in warnings:
        console.warn(warning)
    root = root_of(shell.session.name)
    entries = [entry for entry in build_tree([info.name for info in infos]) if entry.root == root]
    if not entries:
        console.note(f"у сессии {root} нет веток")
        return False
    entries.sort(key=lambda entry: (entry.depth, entry.name))
    table = Table(title=f"Дерево сессии {root}", show_header=False, box=None)
    table.add_column()
    for entry in entries:
        indent = "  " * entry.depth
        marker = " (checkpoint)" if entry.is_checkpoint else ""
        current = " ← текущая" if entry.name == shell.session.name else ""
        orphan = " [сирота]" if entry.orphaned else ""
        table.add_row(f"{indent}{entry.name}{marker}{current}{orphan}")
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


@command("/profile", "создать, выбрать и изменить user profile")
def _cmd_profile(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.warn("использование: /profile create|list|use|show|set|del|delete <...>")
        return False
    action, *rest = args
    directory = shell.memory_root / "profiles"
    if action == "list":
        names = profiles.list_names(directory)
        console.err.print("\n".join(names) if names else "profile list пуст")
        return False
    if action in {"del", "delete"}:
        if len(rest) != 1:
            raise ConfigError("использование: /profile del|delete <name>")
        name = profiles.validate_name(rest[0])
        profiles.delete(name, directory)
        if shell.active_profile == name:
            shell.active_profile = None
            shell.profile_values = {}
            _save_state(shell)
        console.note(f"profile {name} удалён")
        return False
    if action == "create":
        if not rest:
            raise ConfigError("использование: /profile create <name> [key=value ...]")
        name, *pairs = rest
        values = _profile_pairs(pairs)
        profiles.save(name, values, directory)
        shell.active_profile = profiles.validate_name(name)
        shell.profile_values = values
        _save_state(shell)
        console.note(f"profile {name} создан и активирован")
        return False
    if action == "use":
        if len(rest) != 1:
            raise ConfigError("использование: /profile use <name>")
        name = profiles.validate_name(rest[0])
        shell.profile_values = profiles.load(name, directory)
        shell.active_profile = name
        _save_state(shell)
        console.note(f"profile {name} активирован")
        return False
    if action == "show":
        name = rest[0] if rest else shell.active_profile
        if not name:
            console.err.print("active profile отсутствует")
            return False
        values = profiles.load(name, directory)
        console.err.print(f"profile {name}")
        for key, value in sorted(values.items()):
            console.err.print(f"  {key} = {value}")
        return False
    if action == "set":
        if not shell.active_profile:
            raise ConfigError("нет active profile; сначала /profile create")
        if not rest:
            raise ConfigError("использование: /profile set <key=value> ...")
        values = dict(shell.profile_values)
        values.update(_profile_pairs(rest))
        profiles.save(shell.active_profile, values, directory)
        shell.profile_values = values
        _save_state(shell)
        return False
    raise ConfigError("неизвестное действие profile")


def _profile_pairs(pairs: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"ожидалось key=value, получено {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError("ключ preference не может быть пустым")
        # validate through save's same privacy gate before returning.
        profiles.validate_value(value)
        values[key] = value
    return values


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
    table.add_row("active profile", shell.active_profile or "нет")
    if shell.profile_values:
        profile_tokens = _profile_tokens(shell)
        table.add_row("profile tokens", _num(profile_tokens))
    if shell.task is None:
        table.add_row("task", "нет")
        table.add_row("task tokens", "—")
    else:
        status = (
            "done" if shell.task.phase == "done" else ("paused" if shell.task.paused else "running")
        )
        table.add_row("task", f"{shell.task.phase}, {status}")
        table.add_row(
            "task tokens",
            "not injected"
            if shell.task.paused or shell.task.phase == "done"
            else _num(_task_tokens(shell)),
        )
    table.add_row("invariants", _invariant_label(shell))
    table.add_row("prompt/completion", f"{prompt_label}/{completion_label}")
    table.add_row("всего за сессию", _session_tokens_label(shell))
    table.add_row("следующий запрос", _context_label(shell))
    table.add_row("стратегия", shell.agent.context_strategy)
    branch = _branch_label(shell)
    if branch:
        table.add_row("ветка", branch)
    table.add_row("сжатие истории", _compact_label(shell))
    table.add_row("facts", _facts_label(shell))
    table.add_row("memory", _memory_label(shell))
    override = shell.config.params.context_limit
    if override is not None:
        # Override обязан быть виден и здесь: «окно 2500» без второго числа
        # после недель жизни с честным окном читалось бы как поломка карточки
        # (SPEC-w02d08.md §4).
        table.add_row(
            "окно модели",
            f"{shell.agent.context_limit} (override; карточка: "
            f"{_num(tokens.context_limit(shell.card))})",
        )
    else:
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
    _print_growth_table(shell)
    return False


def _task_segments(args: list[str], expected: int, usage: str) -> list[str] | None:
    text = " ".join(args)
    parts = [part.strip() for part in text.split(" :: ")]
    if len(parts) != expected or any(not part for part in parts):
        console.warn(f"нужно: {usage}; delimiter — literal ' :: '")
        return None
    return parts


def _task_configured_secrets(shell: AgentShell) -> tuple[str, ...]:
    """Combine every configured secret source used by this process."""
    values = list(configured_secrets())
    if len(shell.config.api_key) >= 8 and shell.config.api_key not in values:
        values.append(shell.config.api_key)
    return tuple(values)


def _task_save(shell: AgentShell, replacement: TaskState | None) -> bool:
    """Persist one task mutation transactionally, rolling runtime and state back."""
    previous_task = shell.task
    previous_healing = shell.task_needs_healing
    previous_state = copy.deepcopy(shell.session.state)
    if replacement is not None:
        replacement.validate_privacy(_task_configured_secrets(shell))
    shell.task = replacement
    shell.task_needs_healing = False
    if replacement is None:
        shell.session.state.pop(TASK_STATE_KEY, None)
    else:
        shell.session.state[TASK_STATE_KEY] = replacement.to_json()
    try:
        shell.session.save()
    except OSError as error:
        shell.task = previous_task
        shell.task_needs_healing = previous_healing
        shell.session.state = previous_state
        console.warn(f"task не сохранена ({error}) — mutation отменена")
        return False

    return True


def _task_required(shell: AgentShell) -> TaskState | None:
    if shell.task is None:
        console.warn("Task отсутствует — создай её через /task start")
        return None
    return shell.task


def _task_show(shell: AgentShell) -> None:
    task = shell.task
    if task is None:
        console.note("task: нет")
        return
    table = Table(title="Task State Machine", show_header=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("goal", rich_escape(task.goal))
    table.add_row("phase", task.phase)
    status = "paused" if task.paused else ("done" if task.phase == "done" else "running")
    table.add_row("status", status)
    table.add_row("current step", rich_escape(task.current_step))
    table.add_row("expected action", rich_escape(task.expected_action))
    if task.pause_reason:
        table.add_row("pause reason", rich_escape(task.pause_reason))
    if task.result:
        table.add_row("result", rich_escape(task.result))
    console.err.print(table)


@command(
    "/task",
    "formal task state: start/show/update/advance/pause/resume/complete/clear",
    usage="/task <subcommand>",
)
def _cmd_task(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.warn("нужно: /task start|show|update|advance|pause|resume|complete|clear")
        return False
    sub, rest = args[0], args[1:]
    try:
        if sub == "show":
            if rest:
                console.warn("нужно: /task show (без arguments)")
                return False
            _task_show(shell)
            return False
        if sub == "clear":
            if rest:
                console.warn("нужно: /task clear (без arguments)")
                return False
            if shell.task is None and not shell.task_needs_healing:
                console.note("task: уже нет")
                return False
            _task_save(shell, None)
            return False
        if sub == "start":
            if shell.agent.mode == "dialog":
                console.warn("Task нельзя начать в mode=dialog — сначала /mode chat")
                return False
            if shell.task is not None:
                console.warn("Task уже существует — сначала /task clear")
                return False
            if shell.task_needs_healing:
                console.warn("Persisted task повреждена — сначала explicit /task clear")
                return False
            parts = _task_segments(rest, 3, "/task start <goal> :: <step> :: <expected>")
            if parts is None:
                return False
            task = TaskState.start(*parts, configured_secrets=_task_configured_secrets(shell))
            _task_save(shell, task)
            return False

        task = _task_required(shell)
        if task is None:
            return False
        if sub == "update":
            parts = _task_segments(rest, 2, "/task update <step> :: <expected>")
            if parts is None:
                return False
            replacement = task.update(*parts, configured_secrets=_task_configured_secrets(shell))
            _task_save(shell, replacement)
            return False
        if sub == "advance":
            if not rest:
                console.warn("нужно: /task advance <phase> <step> :: <expected>")
                return False
            phase, body = rest[0], rest[1:]
            parts = _task_segments(body, 2, "/task advance <phase> <step> :: <expected>")
            if parts is None:
                return False
            replacement = task.advance(
                phase, *parts, configured_secrets=_task_configured_secrets(shell)
            )
            _task_save(shell, replacement)
            return False
        if sub == "pause":
            reason = " ".join(rest).strip() or None
            replacement = task.pause(reason, configured_secrets=_task_configured_secrets(shell))
            _task_save(shell, replacement)
            return False
        if sub == "resume":
            if rest:
                console.warn("нужно: /task resume (без arguments)")
                return False
            replacement = task.resume(configured_secrets=_task_configured_secrets(shell))
            _task_save(shell, replacement)
            return False
        if sub == "complete":
            result = " ".join(rest).strip()
            if not result:
                console.warn("нужно: /task complete <result>")
                return False
            replacement = task.complete(result, configured_secrets=_task_configured_secrets(shell))
            _task_save(shell, replacement)
            return False
        console.warn(f"неизвестная /task subcommand {sub!r}")
    except TaskStateError as error:
        console.warn(str(error))
    return False


def _invariant_save(shell: AgentShell, replacement: InvariantSet) -> bool:
    """Persist one invariant mutation transactionally, mirroring task state."""
    previous_invariants = shell.invariants
    previous_healing = shell.invariants_needs_healing
    previous_state = copy.deepcopy(shell.session.state)
    shell.invariants = replacement
    shell.invariants_needs_healing = False
    if replacement.rules:
        shell.session.state[INVARIANTS_STATE_KEY] = replacement.to_json()
    else:
        shell.session.state.pop(INVARIANTS_STATE_KEY, None)
    try:
        shell.session.save()
    except OSError as error:
        shell.invariants = previous_invariants
        shell.invariants_needs_healing = previous_healing
        shell.session.state = previous_state
        console.warn(f"invariants не сохранены ({error}) — mutation отменена")
        return False
    return True


def _invariant_configured_secrets(shell: AgentShell) -> tuple[str, ...]:
    return _task_configured_secrets(shell)


def _invariant_list(shell: AgentShell) -> None:
    if not shell.invariants.rules:
        console.note("invariants: нет")
        return
    table = Table(title="Session invariants", show_header=True)
    table.add_column("id", style="dim")
    table.add_column("requirement")
    for rule in shell.invariants.rules:
        table.add_row(rule.id, rich_escape(rule.requirement))
    console.err.print(table)


@command(
    "/invariant",
    "session policy: add/list/remove/clear",
    usage="/invariant <subcommand>",
)
def _cmd_invariant(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        console.warn("нужно: /invariant add|list|remove|clear")
        return False
    sub, rest = args[0], args[1:]
    try:
        if sub == "list":
            if rest:
                console.warn("нужно: /invariant list (без arguments)")
                return False
            _invariant_list(shell)
            return False
        if sub == "clear":
            if rest:
                console.warn("нужно: /invariant clear (без arguments)")
                return False
            _invariant_save(shell, InvariantSet())
            return False
        if sub == "add":
            parts = _task_segments(rest, 2, "/invariant add <id> :: <requirement>")
            if parts is None:
                return False
            replacement = shell.invariants.add(
                Invariant(*parts), configured_secrets=_invariant_configured_secrets(shell)
            )
            _invariant_save(shell, replacement)
            return False
        if sub == "remove":
            if len(rest) != 1:
                console.warn("нужно: /invariant remove <id>")
                return False
            # Reuse the authoritative slug validation without treating the
            # supplied id as free-form text.
            Invariant(rest[0], "validates id")
            _invariant_save(shell, shell.invariants.remove(rest[0]))
            return False
        console.warn(f"неизвестная /invariant subcommand {sub!r}")
    except InvariantError as error:
        console.warn(str(error))
    return False


def _profile_tokens(shell: AgentShell) -> int | None:
    if shell.counter is None or not shell.profile_values:
        return None
    fragment = profiles.messages(shell.profile_values)
    empty = [{"role": "user", "content": ""}]
    total = shell.counter.count([*fragment, *empty])
    overhead = shell.counter.count(empty)
    if total is None or overhead is None:
        return None
    return max(0, total - overhead)


def _task_tokens(shell: AgentShell) -> int | None:
    if shell.counter is None or shell.task is None:
        return None
    fragment = task_messages(shell.task)
    if not fragment:
        return None
    empty = [{"role": "user", "content": ""}]
    total = shell.counter.count([*fragment, *empty])
    overhead = shell.counter.count(empty)
    if total is None or overhead is None:
        return None
    return max(0, total - overhead)


def _invariant_tokens(shell: AgentShell) -> int | None:
    if shell.counter is None or not shell.invariants.rules:
        return None
    fragment = invariant_messages(shell.invariants)
    empty = [{"role": "user", "content": ""}]
    total = shell.counter.count([*fragment, *empty])
    overhead = shell.counter.count(empty)
    if total is None or overhead is None:
        return None
    return max(0, total - overhead)


def _invariant_label(shell: AgentShell) -> str:
    if not shell.invariants.rules:
        return "нет; assessment calls 0"
    mark = "" if shell.counter is not None and shell.counter.exact else "~"
    parts = [
        f"rules {len(shell.invariants.rules)}, block {mark}{_num(_invariant_tokens(shell))} tokens"
    ]
    failures = (
        f", failed {shell.invariant_assessment_failures}"
        if shell.invariant_assessment_failures
        else ""
    )
    parts.append(
        f"assessment calls {shell.invariant_assessment_calls}{failures}, cost "
        f"{shell.invariant_assessment_prompt_tokens}/{shell.invariant_assessment_completion_tokens}"
    )
    return "; ".join(parts)


def _summary_tokens(shell: AgentShell) -> int | None:
    """Size of the summary pseudo-pair, counted in a shape the counter accepts.

    `counter.count(summary_messages(...))` looks right and returns None on the
    exact tokenizer: the pair ends with role=assistant, and MistralCounter
    refuses that shape outright (the same refusal documented for
    _completion_estimate). Both /tokens and /summary then printed a dash
    instead of the number — measured on a w02d09 dry-run. Counting a probe
    request that ENDS with an empty user message keeps the shape legal;
    subtracting that empty request removes the chat-template overhead, so what
    is left is the pair's own cost.
    """
    if shell.counter is None or not shell.summary:
        return None
    empty = [{"role": "user", "content": ""}]
    with_pair = shell.counter.count([*summary_messages(shell.summary), *empty])
    overhead = shell.counter.count(empty)
    if with_pair is None or overhead is None:
        return None
    return max(0, with_pair - overhead)


def _facts_tokens(shell: AgentShell) -> int | None:
    """Size of the current facts block. Same probe-and-subtract as `_summary_tokens`.

    Deliberately re-implemented here rather than calling Agent's own private
    `_facts_block_tokens` — the CLI owns display math, the same split already
    used for `_summary_tokens`/`_completion_estimate`.
    """
    if shell.counter is None or not shell.facts:
        return None
    if shell.agent.context_strategy == "memory":
        block = render_memory(shell._memory_snapshot(), layer="working")
    else:
        block = format_facts(shell.facts, shell.facts_pinned)
    if not block:
        return 0
    empty = [{"role": "user", "content": ""}]
    with_block = shell.counter.count([{"role": "user", "content": block}, *empty])
    overhead = shell.counter.count(empty)
    if with_block is None or overhead is None:
        return None
    return max(0, with_block - overhead)


def _facts_label(shell: AgentShell) -> str:
    """/tokens line about facts: block size, pins, extractor calls and price."""
    if shell.agent.context_strategy != "facts":
        return f"неприменимо: context_strategy={shell.agent.context_strategy} (не facts)"
    if not shell.facts:
        parts = ["блок пуст"]
    else:
        size = _facts_tokens(shell)
        mark = "" if shell.counter is not None and shell.counter.exact else "~"
        parts = [f"ключей {len(shell.facts)}, блок {mark}{_num(size)} токенов"]
    if shell.facts_calls:
        failed = f", из них неуспешных {shell.facts_failures}" if shell.facts_failures else ""
        parts.append(
            f"вызовов экстрактора {shell.facts_calls}{failed}, стоили "
            f"{shell.facts_prompt_tokens}/{shell.facts_completion_tokens}"
        )
    return "; ".join(parts)


def _memory_label(shell: AgentShell) -> str:
    if shell.agent.context_strategy != "memory":
        return f"неприменимо: context_strategy={shell.agent.context_strategy} (не memory)"
    working_size = len(shell.memory.working.entries)
    long_size = len(shell.memory.long_term.entries)
    dirty = (
        " (есть unsaved layers)"
        if shell.memory_dirty_working or shell.memory_dirty_long_term
        else ""
    )
    return (
        f"working {working_size}, long-term {long_size}, short-term {len(shell.history)}; "
        f"extractor {shell.memory_calls} calls, failed {shell.memory_failures}, "
        f"cost {shell.memory_prompt_tokens}/{shell.memory_completion_tokens}{dirty}"
    )


def _compact_label(shell: AgentShell) -> str:
    """/tokens line about compaction: on or off, summary present, its cost.

    Compaction cost is named here because there's nowhere else to name it:
    service calls land in neither "total for session" nor the growth table
    (SPEC §7).
    """
    if not shell.config.params.compact:
        return "выключено (/set compact on)"
    if shell.agent.context_strategy != "summary":
        # compact_enabled() would also say False here, but for a different
        # reason than "no summarizer prompt" — naming the actual strategy
        # avoids sending the user to fix a prompt file that isn't the issue.
        return f"неприменимо: context_strategy={shell.agent.context_strategy} (не summary)"
    if not shell.agent.compact_enabled():
        # Param is on but there's nothing to compact with (prompt didn't
        # load) — not "disabled by the user", needs different wording.
        return "включено, но суммаризатора нет — работает обычная обрезка"

    parts = [f"порог {shell.agent.compact_every} сообщений, хвост {shell.agent.keep_last}"]
    if shell.summary:
        covered = max(0, len(shell.session.turns) - len(shell.history))
        size = _summary_tokens(shell)
        mark = "" if shell.counter is not None and shell.counter.exact else "~"
        parts.append(f"пересказ есть: {covered} сообщений в {mark}{_num(size)} токенов")
    else:
        parts.append("пересказа ещё нет")
    if shell.compact_calls:
        parts.append(
            f"сжатий {shell.compact_calls}, стоили "
            f"{shell.compact_prompt}/{shell.compact_completion}"
        )
    return "; ".join(parts)


@command("/compact", "сжать историю прямо сейчас, не дожидаясь порога")
def _cmd_compact(shell: AgentShell, args: list[str]) -> bool:
    _warn_extra_args("/compact", args)
    compaction = shell.agent.compact_now(shell.summary, shell.history)
    if compaction is None:
        # Agent already named the reason via on_warning: disabled, nothing
        # to compact, or the call failed. Repeating it here is just noise.
        return False
    shell.summary = compaction.summary
    shell.history = list(compaction.tail)
    _report_compaction(shell, compaction)
    # Straight to disk: the summary is a form of memory, and losing it to
    # Ctrl+C between compaction and the next turn would lose the whole
    # conversation up to the tail — the compacted part won't return to the
    # working history.
    _save_state(shell)
    return False


@command("/summary", "показать действующий пересказ целиком")
def _cmd_summary(shell: AgentShell, args: list[str]) -> bool:
    _warn_extra_args("/summary", args)
    if not shell.summary:
        console.note("пересказа нет: история идёт в модель целиком")
        return False
    covered = max(0, len(shell.session.turns) - len(shell.history))
    size = _summary_tokens(shell)
    mark = "" if shell.counter is not None and shell.counter.exact else "~"
    console.note(f"пересказ заменяет {covered} сообщений и занимает {mark}{_num(size)} токенов:")
    # stderr, like all REPL service output: a command's product is the
    # model's answer, not a status report (stdout/stderr contract, CLAUDE.md).
    # rich_escape: the summary came from the model, and any [something] in
    # it would be taken by Rich as markup.
    console.err.print(rich_escape(shell.summary))
    return False


def _memory_layer_name(raw: str) -> str:
    value = raw.strip().lower()
    aliases = {
        "short-term": "short",
        "short_term": "short",
        "long-term": "long",
        "long_term": "long",
    }
    return aliases.get(value, value)


def _memory_value(shell: AgentShell, layer: str) -> StructuredMemory:
    if layer == "working":
        return shell.memory.working
    if layer == "long":
        return shell.memory.long_term
    raise ValueError("memory operations доступна только для working или long")


def _replace_memory_layer(shell: AgentShell, layer: str, value: StructuredMemory) -> None:
    if layer == "working":
        shell.memory = MemorySnapshot(shell.memory.short_term, value, shell.memory.long_term)
        shell.memory_dirty_working = True
    else:
        shell.memory = MemorySnapshot(shell.memory.short_term, shell.memory.working, value)
        shell.memory_dirty_long_term = True
    _save_state(shell)


@command(
    "/memory",
    "показать или изменить short-term, working и long-term memory",
    usage="/memory [short|working|long|set|del|pin|unpin|move|backfill|retry]",
)
def _cmd_memory(shell: AgentShell, args: list[str]) -> bool:
    if not args:
        working_path = shell.memory_store.working_path(shell.session.name)
        long_term_path = shell.memory_store.long_term_path
        console.err.print(
            rich_escape(
                "memory: short-term "
                f"{len(shell.history)} messages; "
                f"working {len(shell.memory.working.entries)} entries "
                f"({len(shell.memory.working.pinned)} pinned); long-term "
                f"{len(shell.memory.long_term.entries)} entries "
                f"({len(shell.memory.long_term.pinned)} pinned); "
                f"working path: {working_path}; long-term path: {long_term_path}"
            )
        )
        return False
    sub = _memory_layer_name(args[0])
    if sub in ("short", "working", "long"):
        if len(args) > 1:
            _warn_extra_args(f"/memory {args[0]}", args[1:])
        if sub == "short":
            console.err.print(rich_escape(f"short-term: {len(shell.history)} messages"))
        else:
            layer = "long_term" if sub == "long" else "working"
            console.err.print(rich_escape(render_memory(shell._memory_snapshot(), layer=layer)))
        return False
    if sub == "retry":
        shell._save_memory()
        if not shell.memory_dirty_working and not shell.memory_dirty_long_term:
            console.note("memory: dirty layers отсутствуют")
        return False
    if sub == "backfill":
        _memory_backfill(shell)
        return False
    if sub not in ("set", "del", "pin", "unpin", "move"):
        console.warn("нужно: /memory [short|working|long|set|del|pin|unpin|move|backfill|retry]")
        return False
    try:
        if sub == "set":
            if len(args) < 4:
                raise ValueError("нужно: /memory set working|long <field.key> <value>")
            layer, key, value = _memory_layer_name(args[1]), args[2], " ".join(args[3:])
            if layer == "long" and is_credential_like(key, value):
                raise ValueError("credential-like data нельзя сохранять в long-term")
            snapshot = manual_set(
                shell._memory_snapshot(), "long_term" if layer == "long" else layer, key, value
            )
            _replace_memory_layer(
                shell, layer, snapshot.long_term if layer == "long" else snapshot.working
            )
        elif sub in ("del", "pin", "unpin"):
            if len(args) != 3:
                raise ValueError(f"нужно: /memory {sub} working|long <field.key>")
            layer, key = _memory_layer_name(args[1]), args[2]
            target = _memory_value(shell, layer)
            field_name, name = split_key(key)
            canonical = validate_memory_key(
                "long_term" if layer == "long" else "working", field_name, name
            )
            entries, pins = dict(target.entries), set(target.pinned)
            if sub == "del":
                if canonical not in entries:
                    raise ValueError(f"memory key {canonical!r} не найден")
                entries.pop(canonical)
                pins.discard(canonical)
            elif canonical not in entries:
                raise ValueError(f"memory key {canonical!r} не найден")
            elif sub == "pin":
                pins.add(canonical)
            else:
                pins.discard(canonical)
            _replace_memory_layer(shell, layer, StructuredMemory(entries, frozenset(pins)))
        else:  # move
            if len(args) != 5:
                raise ValueError(
                    "нужно: /memory move working|long working|long <source-key> <target-key>"
                )
            source_layer, target_layer, source_key, target_key = map(_memory_layer_name, args[1:3])
            source_key, target_key = args[3], args[4]
            source = _memory_value(shell, source_layer)
            source_field, source_name = split_key(source_key)
            source_canonical = validate_memory_key(
                "long_term" if source_layer == "long" else "working", source_field, source_name
            )
            if source_canonical not in source.entries:
                raise ValueError(f"memory key {source_canonical!r} не найден")
            target_field, target_name = split_key(target_key)
            target_canonical = validate_memory_key(
                "long_term" if target_layer == "long" else "working", target_field, target_name
            )
            value = source.entries[source_canonical]
            if target_layer == "long" and is_credential_like(target_canonical, value):
                raise ValueError("credential-like data нельзя сохранять в long-term")
            src_entries, src_pins = dict(source.entries), set(source.pinned)
            src_entries.pop(source_canonical)
            src_pins.discard(source_canonical)
            dst = _memory_value(shell, target_layer)
            dst_entries, dst_pins = dict(dst.entries), set(dst.pinned)
            dst_entries[target_canonical] = value
            dst_pins.add(target_canonical)
            if source_layer == target_layer:
                _replace_memory_layer(
                    shell,
                    source_layer,
                    StructuredMemory(dst_entries | src_entries, frozenset(dst_pins | src_pins)),
                )
            else:
                _replace_memory_layer(
                    shell, source_layer, StructuredMemory(src_entries, frozenset(src_pins))
                )
                _replace_memory_layer(
                    shell, target_layer, StructuredMemory(dst_entries, frozenset(dst_pins))
                )
        console.note(f"memory: {sub} выполнено")
    except (ValueError, KeyError) as error:
        console.warn(str(error))
    return False


def _memory_backfill(shell: AgentShell) -> None:
    history = shell.session.history()
    if not history:
        console.note("сессия пуста — memory backfill не нужен")
        return
    seed = StructuredMemory(
        {
            key: value
            for key, value in shell.memory.working.entries.items()
            if key in shell.memory.working.pinned
        },
        shell.memory.working.pinned,
    )
    snapshot = MemorySnapshot(ShortTermMemory(), seed, shell.memory.long_term)
    try:
        _snapshot, _upto, update, failure = shell.agent._run_memory(
            snapshot, history, "", 0, catchup_max=None
        )
    except TypeError:
        # Compatibility with a test double implementing the older private seam.
        _snapshot, _upto, update, failure = shell.agent._run_memory(snapshot, history, "", 0)
    if failure is not None:
        _report_memory_failure(shell, failure)
        return
    if update is None:
        console.warn("memory backfill не вернул update")
        return
    shell.memory = update.snapshot
    shell.memory_upto = len(history)
    shell.memory_dirty_working = True
    shell.memory_dirty_long_term = bool(
        update.applied and any(layer == "long_term" for layer, _ in update.applied)
    )
    _report_memory_update(shell, update)
    _save_state(shell)


def _facts_backfill(shell: AgentShell) -> None:
    """One extractor pass over the WHOLE session (SPEC-w02d10.md §5.6).

    Seeded with pinned facts only, not the full current block: apply_delta()
    blocks any extractor set/delete touching a pinned key regardless of
    whether that key is already present — seeding with everything would let
    the extractor silently confirm stale non-pinned values instead of
    actually re-deriving them from scratch, which is the whole point of an
    explicit rebuild.

    Calls Agent._run_facts() directly — a private method, not part of
    advent_core/agent.py's public surface (no public "run the extractor once
    over arbitrary history" exists, only the full per-turn ask()). Documented
    as a deliberate exception rather than reimplementing the catch-up/
    truncation rules a second time in the CLI (see followups).
    """
    history = shell.session.history()
    if not history:
        console.note("сессия пуста — извлекать нечего")
        return
    seed = {key: value for key, value in shell.facts.items() if key in shell.facts_pinned}
    facts_before = dict(shell.facts)
    # catchup_max=None (review finding P3): backfill means "the whole file",
    # not "the last FACTS_CATCHUP_MAX messages" — the per-turn catch-up cap
    # exists for the ongoing-conversation case, not for an explicit rebuild.
    _facts, _upto, update = shell.agent._run_facts(
        seed, shell.facts_pinned, history, "", 0, catchup_max=None
    )
    if update is None:
        failure = shell.agent.take_pending_facts_failed()
        if failure is not None:
            _report_facts_failure(shell, failure, backfill=True)
        # Agent already warned the reason (no prompt / call failed / bad delta).
        return
    shell.facts = update.facts
    shell.facts_upto = len(history)
    _report_facts_update(shell, update, before=facts_before, backfill=True)
    _save_state(shell)


@command(
    "/facts",
    "показать блок facts целиком; `backfill` — пересобрать одним вызовом по всей сессии",
    usage="/facts [backfill]",
)
def _cmd_facts(shell: AgentShell, args: list[str]) -> bool:
    global _FACTS_ALIAS_SHOWN
    if not _FACTS_ALIAS_SHOWN:
        console.note("/facts — compatibility alias для /memory working")
        _FACTS_ALIAS_SHOWN = True
    if args and args[0] == "backfill":
        _warn_extra_args("/facts backfill", args[1:])
        _facts_backfill(shell)
        return False
    _warn_extra_args("/facts", args)
    blocks: list[str] = []
    working_block = render_memory(shell._memory_snapshot(), layer="working")
    if shell.memory.working.entries:
        blocks.append(working_block)
    # Keep the legacy block visible for old sessions and scripts while the
    # alias itself is now backed by working memory for every strategy.
    legacy_block = format_facts(shell.facts, shell.facts_pinned)
    if legacy_block:
        blocks.append(legacy_block)
    if not blocks:
        console.note("фактов пока нет")
        return False
    # rich_escape: facts values come from the extractor/model, [что-то] would
    # otherwise be eaten by Rich as markup — same reasoning as /summary.
    console.err.print(rich_escape("\n".join(blocks)))
    return False


@command(
    "/fact",
    "правка facts вручную: set <категория.имя> <значение> (и закрепляет) | del <ключ> | "
    "unpin <ключ>",
    usage="/fact set|del|unpin <ключ> [значение]",
)
def _cmd_fact(shell: AgentShell, args: list[str]) -> bool:
    # Keep the old surface while routing every edit to session-scoped working
    # memory. Legacy facts are mirrored for tagged scripts and old sessions;
    # the canonical target is working memory regardless of strategy.
    if args:
        if args and args[0] == "set" and len(args) >= 3:
            raw_key, value = args[1], " ".join(args[2:])
            category, _, name = raw_key.partition(".")
            field_name = {
                "цель": "goal",
                "ограничения": "constraints",
                "предпочтения": "constraints",
                "решения": "decisions",
                "договорённости": "decisions",
            }.get(category)
            if field_name is None:
                console.warn(
                    f"неизвестная категория {category!r}: legacy /fact key должен "
                    "начинаться с цель., ограничения., решения. или договорённости."
                )
                return False
            suffix = name
            if category == "предпочтения":
                suffix = f"legacy_preference.{name}"
            elif category == "договорённости":
                suffix = f"legacy_agreement.{name}"
            try:
                legacy_key = validate_key(raw_key)
            except ValueError as error:
                console.warn(str(error))
                return False
            before = dict(shell.facts)
            shell.facts[legacy_key] = value
            if legacy_key not in shell.facts_pinned:
                shell.facts_pinned.append(legacy_key)
            note = render_note(before, shell.facts)
            if note:
                console.note(f"{note} (закреплено)")
            return _cmd_memory(shell, ["set", "working", f"{field_name}.{suffix}", value])
        if args and args[0] in ("del", "unpin") and len(args) >= 2:
            raw_key = args[1]
            category, _, name = raw_key.partition(".")
            field_name = {
                "цель": "goal",
                "ограничения": "constraints",
                "предпочтения": "constraints",
                "решения": "decisions",
                "договорённости": "decisions",
            }.get(category)
            if field_name is not None:
                if category == "предпочтения":
                    name = f"legacy_preference.{name}"
                elif category == "договорённости":
                    name = f"legacy_agreement.{name}"
                try:
                    legacy_key = validate_key(raw_key)
                except ValueError as error:
                    console.warn(str(error))
                    return False
                if legacy_key not in shell.facts:
                    if args[0] == "unpin":
                        console.warn(f"{legacy_key!r} не был закреплён")
                        return False
                    console.warn(f"ключ {legacy_key!r} не найден")
                    return False
                if args[0] == "del":
                    shell.facts.pop(legacy_key, None)
                    if legacy_key in shell.facts_pinned:
                        shell.facts_pinned.remove(legacy_key)
                elif legacy_key in shell.facts_pinned:
                    shell.facts_pinned.remove(legacy_key)
                return _cmd_memory(shell, [args[0], "working", f"{field_name}.{name}"])
    if not args:
        console.warn("нужно: /fact set|del|unpin <ключ> ...")
        return False
    sub, *rest = args
    if sub == "set":
        if len(rest) < 2:
            console.warn("нужно: /fact set <категория.имя> <значение>")
            return False
        raw_key, value = rest[0], " ".join(rest[1:])
        try:
            key = validate_key(raw_key)
        except ValueError as error:
            console.warn(str(error))
            return False
        before = dict(shell.facts)
        shell.facts[key] = value
        if key not in shell.facts_pinned:
            shell.facts_pinned.append(key)
        note = render_note(before, shell.facts)
        console.note(f"{note} (закреплено)" if note else f"{key} закреплён")
        _save_state(shell)
        return False
    if sub == "del":
        if not rest:
            console.warn("нужно: /fact del <ключ>")
            return False
        key = rest[0]
        if key not in shell.facts:
            console.warn(f"ключ {key!r} не найден")
            return False
        before = dict(shell.facts)
        del shell.facts[key]
        if key in shell.facts_pinned:
            shell.facts_pinned.remove(key)
        note = render_note(before, shell.facts)
        if note:
            console.note(note)
        _save_state(shell)
        return False
    if sub == "unpin":
        if not rest:
            console.warn("нужно: /fact unpin <ключ>")
            return False
        key = rest[0]
        if key not in shell.facts_pinned:
            console.warn(f"{key!r} не был закреплён")
            return False
        shell.facts_pinned.remove(key)
        console.note(f"{key} больше не закреплён")
        _save_state(shell)
        return False
    console.warn(f"неизвестное действие /fact {sub!r}: set | del | unpin")
    return False


# Сколько ходов показывает таблица роста: длинные сессии режутся, иначе таблица
# уезжает за экран — на видео видны были бы только последние строки и так.
GROWTH_TABLE_LIMIT = 30


def _print_growth_table(shell: AgentShell) -> None:
    """Таблица роста по ходам (SPEC-w02d08.md §3) — история, а не только итоги.

    Источник — usage ходов сессии, факт сервера. «накопительно» — сумма
    total_tokens (или prompt+completion, когда total сервер не прислал): история
    пересылается целиком, и именно эта колонка показывает, как «стоимость
    диалога в токенах» растёт квадратично его длине. Ход без usage — прочерки:
    «неизвестно» не становится нулём ни в одной колонке — и, раз хоть один
    такой ход пропущен, накопительная сумма после него уже не полная, о чём
    отдельная строка говорит после таблицы: молчание превратило бы неполный
    итог в мнимо точный.
    """
    turns = [turn for turn in shell.session.turns if turn.role == ROLE_ASSISTANT]
    if not turns:
        console.note("ходов с ответами в сессии ещё нет — расти пока нечему")
        return

    rows: list[tuple[int, str, str, str]] = []
    cumulative = 0
    # Ходов, не вошедших в «накопительно», — считаем отдельно: их отсутствие
    # называется вслух после таблицы, а не молчаливо теряется в последней
    # видимой сумме (НАХОДКА 4 ревью w02d08).
    missing = 0
    for number, turn in enumerate(turns, start=1):
        usage = turn.usage
        if usage is None or usage.is_empty():
            missing += 1
            rows.append((number, "—", "—", "—"))
            continue
        prompt = usage.prompt_tokens
        completion = usage.completion_tokens
        total = usage.total_tokens
        if total is None:
            # Ветки «total не пришёл, а prompt или completion тоже нет» здесь
            # нет — она недостижима. Usage.is_empty() (проверка выше) ложна
            # только когда total_tokens пришёл, ИЛИ пришли prompt И
            # completion оба сразу; раз total пуст, значит пришли оба —
            # сложение всегда безопасно. Мёртвая ветка на этом месте молчала
            # бы о невозможном случае вместо того, чтобы объяснить, почему
            # его нет (НАХОДКА 5 ревью w02d08).
            total = prompt + completion
        cumulative += total
        rows.append((number, _num(prompt), _num(completion), str(cumulative)))

    # header_style, а не ручной add_row поверх именованных колонок: у Rich
    # show_header=True по умолчанию, и колонки уже печатают шапку сами —
    # add_row тем же текстом рисовал вторую строку заголовка под первой
    # (проверено рендером, артефакт в кадре шагов 1-3 демо).
    table = Table(box=None, padding=(0, 2, 0, 0), header_style="dim")
    table.add_column("ход", justify="right", style="cyan", no_wrap=True)
    table.add_column("prompt", justify="right")
    table.add_column("completion", justify="right")
    table.add_column("накопительно", justify="right")
    for number, prompt, completion, total in rows[-GROWTH_TABLE_LIMIT:]:
        table.add_row(str(number), prompt, completion, total)
    console.err.print(table)
    if len(rows) > GROWTH_TABLE_LIMIT:
        console.note(f"… показаны последние {GROWTH_TABLE_LIMIT} из {len(rows)}")
    if missing:
        # Без этой строки последнее видимое «накопительно» выглядит точным
        # итогом сессии, хотя не включает вклад ходов без usage — то же
        # правило, что у «(без usage: N)» в /tokens и /sessions.
        console.note(f"в накопительном итоге не учтено ходов: {missing}")


def _checkpoint_switch_refusal(shell: AgentShell, name: str) -> str | None:
    """Message refusing to switch INTO `name`, or None if it's fine.

    Single shared guard (review finding P1): a checkpoint can only be
    branched from, never switched into (SPEC-w02d10.md §7.1). `/switch` used
    to be the only caller that checked this — `/set session <checkpoint>`
    reached `_switch_session()`/`open_session()` directly and bypassed it
    entirely. Every path that can land on a session now calls this one
    function instead of re-deriving the rule: `_switch_session` (covers
    `/set session`, `/branch`'s switch-into-the-new-branch, `/new`'s fallback)
    and startup's `--session`.
    """
    if not Session.path_for(name, shell.directory).is_file():
        return None
    peek = Session.load(name, directory=shell.directory, quarantine=False)
    if peek.state.get("kind") != "checkpoint":
        return None
    return (
        f"{name!r} — checkpoint, в него нельзя переключиться: "
        f"/branch <имя> --from {name} — ответвиться от него"
    )


def _switch_settings_note(
    *,
    before: tuple[str, int | None, str],
    after: tuple[str, int | None, str],
    target: str,
    is_branch: bool,
) -> str | None:
    """Diff-only note for `/switch`/`/branch` (SPEC-w02d10.md §7.4).

    None settings actually differ — a note on every switch would drown the
    rare case that matters: settings genuinely changing underneath the user.
    """
    before_strategy, before_limit, before_mode = before
    after_strategy, after_limit, after_mode = after
    diffs: list[str] = []
    if before_strategy != after_strategy:
        diffs.append(f"context_strategy {after_strategy} (было {before_strategy})")
    if before_limit != after_limit:
        diffs.append(
            f"окно {after_limit if after_limit is not None else 'из карточки'} "
            f"(было {before_limit if before_limit is not None else 'из карточки'})"
        )
    if before_mode != after_mode:
        diffs.append(f"mode {after_mode} (было {before_mode})")
    if not diffs:
        return None
    kind = "ветка" if is_branch else "сессия"
    return f"{kind} {target}: " + ", ".join(diffs)


def _flush_memory_for_identity_change(shell: AgentShell) -> bool:
    """Persist all dirty memory before changing the session identity.

    A failed working save must never be hidden by loading another session, and
    the same is true for the process-global long-term layer: reloading it from
    disk would silently discard a dirty in-memory value.
    """
    if not (shell.memory_dirty_working or shell.memory_dirty_long_term):
        return True
    shell._save_memory()
    if shell.memory_dirty_working or shell.memory_dirty_long_term:
        console.warn("memory не сохранена — смена session отменена")
        return False
    return True


def _switch_session(
    shell: AgentShell, name: str, *, fresh: bool = False, announce_diff: bool = False
) -> None:
    """Переключение сессии: текущая сохраняется, история подменяется.

    `fresh` стирает содержимое целевой сессии — это и есть `/new`. Отказ вместо
    стирания выглядел бы безопаснее, но сорвал бы второй дубль записи: демо
    начинается с пустой сессии, и повторный прогон обязан приводить в то же
    состояние, что и первый.

    `announce_diff` — только для `/switch` и `/branch` (SPEC §7.4): печатает
    ноту, если context_strategy/context_limit/mode реально отличаются у цели.
    `/new` и `/set session` его не просят — тот же переход, но без этой ноты,
    чтобы не менять уже сданное поведение дня 09.
    """
    validate_name(name)
    refusal = _checkpoint_switch_refusal(shell, name)
    if refusal:
        console.warn(refusal)
        # config.params.session was already written by _apply_set's "session"
        # branch before this function ran — undo it so /params doesn't claim
        # a session that was never actually entered.
        shell.config.params.session = shell.session.name
        return
    if not _flush_memory_for_identity_change(shell):
        return
    before = (
        (shell.agent.context_strategy, shell.config.params.context_limit, shell.agent.mode)
        if announce_diff
        else None
    )
    previous_profile = shell.active_profile
    try:
        shell.session.save()
    except OSError as error:
        console.warn(f"сессия не сохранена ({error}) — смена session отменена")
        return
    # `/new` stages a replacement before it is allowed to touch the target.
    # A malformed target must therefore stay in place until the clean atomic
    # save succeeds; ordinary switches keep the established quarantine path.
    session = shell.open_session(name, quarantine=not fresh)
    dropped = 0
    fresh_branch_parent: str | None = None
    if fresh:
        session.state["active_profile"] = previous_profile
        dropped = len(session.turns)
        session.clear()
        # Разговор стёрт — счётчик диалога обнуляем вместе с ним: «ход 2 из 10»
        # без истории врал бы. mode/done не трогаем: это настройка запуска,
        # а не содержимое разговора (clear() state не стирает).
        session.state["dialog_turns"] = 0
        parent, _segment, _is_checkpoint = parse_name(session.name)
        if parent is not None:
            fresh_branch_parent = parent
        try:
            # Save the clean target before changing runtime identity. Session.save
            # uses a sibling temp plus os.replace, so a failure leaves both the
            # current runtime and the previous target file intact.
            session.save()
        except OSError as error:
            console.warn(f"новая сессия не сохранена ({error}) — /new отменена")
            return
        if dropped:
            console.warn(f"сессия {session.name}: удалено ходов {dropped}")
        if fresh_branch_parent is not None:
            # SPEC §7.3: /new inside a branch clears only that branch's own
            # file — the parent must never learn this happened. Announce only
            # after the clean target was saved successfully.
            console.note(
                f"{session.name} — ветка (родитель {fresh_branch_parent}), а не корень; "
                "стёрта только она"
            )
    shell.session = session
    shell.history = session.history()
    shell.config.params.session = session.name
    # Это состояние ТОГО разговора: применяем оптом, explicit-флаги не чекая —
    # они были про старт запуска, а переключение = «продолжить как оставили».
    resumed_dialog = shell._apply_session_state(session, check_explicit=False)
    # A fresh session must not load the old session-scoped working file: its
    # cursor necessarily refers to the conversation that `/new` is clearing.
    # Loading it first would emit a false stale-cursor warning and, more
    # importantly, leave the old entries on disk until a later memory update.
    shell._load_memory(load_working=not fresh)
    shell._load_profile()
    if fresh:
        # `/new` resets session-scoped working memory but leaves global
        # long-term memory intact.
        shell.memory = MemorySnapshot(ShortTermMemory(), StructuredMemory(), shell.memory.long_term)
        shell.memory_upto = 0
        shell.memory_dirty_working = True
        # Persist the lifecycle boundary now.  If the write fails,
        # `_save_memory` deliberately keeps the dirty flag for `/memory retry`;
        # silently leaving the old file would resurrect memory after restart.
        shell._save_memory()
    # state подхвачен — и лимит пересчитываем: у сессии, на которую переключились,
    # override в файле может отличаться от текущего (или отсутствовать), а trim
    # со следующего хода идёт по тому, что в агенте.
    shell.agent.context_limit = shell.effective_context_limit()
    if resumed_dialog:
        console.note(
            f"подхвачен целевой диалог: ход {shell.dialog_turns} из {shell.agent.max_turns}"
        )
        if shell.task is not None:
            console.warn(
                "Persisted task несовместима с mode=dialog — используй /mode chat либо /task clear"
            )
    if before is not None:
        after = (shell.agent.context_strategy, shell.config.params.context_limit, shell.agent.mode)
        parent, _segment, _is_checkpoint = parse_name(session.name)
        note = _switch_settings_note(
            before=before, after=after, target=session.name, is_branch=parent is not None
        )
        if note:
            console.note(note)
    shell.last_question = None
    shell.last_check = None
    # New conversation identity — a stale "already announced N" would
    # suppress a genuine window-drop note in the session just entered.
    shell.window_dropped_reported = None


def _paused_command_allowed(parts: list[str], shell: AgentShell) -> bool:
    command = parts[0]
    if command in {
        "/help",
        "/exit",
        "/tokens",
        "/params",
        "/sessions",
        "/branches",
        "/summary",
    }:
        return len(parts) == 1
    if command in {"/checkpoint", "/new"}:
        return len(parts) in {1, 2}
    if command == "/switch":
        return len(parts) == 2
    if command == "/branch":
        return (
            (len(parts) == 2 and not parts[1].startswith("--"))
            or (len(parts) == 3 and parts[1] == "--delete" and not parts[2].startswith("--"))
            or (
                len(parts) == 4
                and not parts[1].startswith("--")
                and parts[2] == "--from"
                and not parts[3].startswith("--")
            )
        )
    if command == "/facts":
        return len(parts) == 1
    if command == "/memory":
        return len(parts) == 1 or (len(parts) == 2 and parts[1] in {"short", "working", "long"})
    if command == "/profile":
        return parts == ["/profile", "list"] or (
            parts[:2] == ["/profile", "show"] and len(parts) in {2, 3}
        )
    if command == "/task":
        return len(parts) == 2 and parts[1] in {"show", "resume", "clear"}
    if command == "/invariant":
        if len(parts) < 2:
            return False
        sub = parts[1]
        return (
            (sub in {"list", "clear"} and len(parts) == 2)
            or (sub == "remove" and len(parts) == 3)
            or (sub == "add" and len(parts) >= 5)
        )
    if shell.agent.mode == "dialog":
        if command == "/mode" and parts[1:] == ["chat"]:
            return True
        if command == "/set" and parts[1:] == ["mode", "chat"]:
            return True
    return False


def _dispatch(line: str, shell: AgentShell) -> bool:
    """Исполняет слэш-команду. True означает «выходим»."""
    parts = line.split()
    if shell.task is not None and shell.task.paused and not _paused_command_allowed(parts, shell):
        console.warn("Task paused — команда blocked; используй /task resume или /task clear")
        return False
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
        if shell.task is not None and shell.task.paused:
            console.warn("Task paused — prompt blocked; используй /task resume или /task clear")
            continue
        if shell.task is not None and shell.agent.mode == "dialog":
            console.warn(
                "Task State Machine несовместима с mode=dialog — используй /mode chat "
                "либо /task clear"
            )
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


def _explicit_local_flags(
    *, mode: str | None, done: str | None, context_limit: int | None
) -> frozenset[str]:
    """Имена local-параметров, заданных флагом запуска явно, а не файлом сессии.

    Отдельная функция, а не блок кода внутри agent(): agent() — typer-команда,
    прямой вызов которой в обход Typer оставляет непереданные параметры
    объектами typer.Option(), а не их значениями, — и тестировать построение
    приоритета «флаг > файл > дефолт» пришлось бы через CliRunner. Здесь же
    сигнатура берёт только то, что реально решает исход (mode/done/
    context_limit), и проверяется прямым вызовом (ревью w02d08, НАХОДКА 7: до
    этой правки победу флага над файлом сессии не ловил ни один тест).
    """
    return frozenset(
        name
        for name, value in (
            ("mode", mode),
            ("done", done),
            ("context_limit", context_limit),
        )
        if value is not None
    )


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
    context_limit: int | None = typer.Option(
        None, "--context-limit", help=BY_NAME["context_limit"].help
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
        context_limit=context_limit,
        format=format_,
        schema_file=schema_file,
    )
    # Умолчания приходят из реестра, а не литералами в сигнатуре typer: иначе
    # одно правило записано в двух местах и разъедется при первой правке.
    config.params.apply_defaults(AGENT_COMMAND)

    # Какие local-параметры заданы флагами явно — они бьют файл сессии при
    # подхвате state (приоритет «флаг > файл > дефолт реестра»).
    explicit = _explicit_local_flags(mode=mode, done=done, context_limit=context_limit)
    shell = AgentShell(config, explicit=explicit)
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
