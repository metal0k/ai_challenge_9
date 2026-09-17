"""Агент: сборка запроса, вызов, разбор ответа, обрезка контекста, детект done.

Агент — сущность, а не обёртка над одним вызовом API (SPEC-w02d06.md §1): у
него собственный system prompt, собственный учёт токенов, собственное решение
об обрезке контекста и собственное понимание того, когда целевой диалог
закончен.

Две вещи, которых агент НЕ делает, и обе намеренно:

* **не хранит историю внутри себя.** `ask()` принимает историю и возвращает
  новую. Так сделан pydantic-ai; явная история тривиально мокается и
  инспектируется в тестах, что совпадает с принятым в проекте «тесты не ходят
  в сеть». smolagents держит state внутри инстанса — хуже тестируется и мешает
  память с вызовом (SPEC-w02d06.md §5).
* **ничего не печатает.** Ни предупреждений, ни footer. Он возвращает
  `dropped`, `done`, `context_tokens` и отдаёт предупреждения в `on_warning`,
  а что и куда печатать, решает CLI — иначе контракт stdout/stderr
  размазывается по двум слоям (§14).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from advent_core import chat as chat_core
from advent_core import formats
from advent_core.chat import Message
from advent_core.compact import (
    ROLE_LABELS,
    fold_summary,
    should_compact,
    split_history,
    summary_messages,
)
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError, ConfigurationError
from advent_core.facts import DeltaResult, apply_delta, facts_messages, format_facts
from advent_core.memory import (
    MemoryDelta,
    MemoryFailure,
    MemoryOperation,
    MemorySnapshot,
    MemoryUpdate,
    ShortTermMemory,
    StructuredMemory,
    memory_messages,
)
from advent_core.memory import (
    apply_delta as apply_memory_delta,
)
from advent_core.profiles import messages as profile_messages
from advent_core.task_state import TaskState, task_messages
from advent_core.telemetry import CallResult
from advent_core.tokens import TokenCounter, reconcile

# Сигнатуры шва: те же, что у chat.complete/chat.stream. Callable[..., ...] —
# не лень, а осознанное послабление: mypy всё равно не проверит соответствие
# именованных аргументов, а тестовый двойник имеет право быть проще боевой
# функции (например, не принимать capabilities).
CompleteFn = Callable[..., CallResult]
StreamFn = Callable[..., CallResult]

# Потолок ходов целевого диалога, когда max_turns сброшен в None. По контракту
# params.py None означает «дефолт цикла», а не «без потолка»: без потолка
# demo-шаг advent record повис бы до таймаута на первом же вопросе модели.
DEFAULT_MAX_TURNS = 10

# Запас под ответ, когда max_tokens не задан. Порог обрезки — это окно модели
# МИНУС то, что она собирается написать: посчитать «влезает» по полному окну
# значит гарантированно получить 400 на длинном ответе.
RESPONSE_RESERVE_TOKENS = 1024

# How many recent messages stay untouched when keep_last is unset. Six is
# three full pairs: measured on a real session, minus 62% on the request
# while keeping the last three exchanges live (PROBE-w02d09-compact.md §2).
DEFAULT_KEEP_LAST = 6

# Threshold for scheduled compaction, in not-yet-compacted old messages.
DEFAULT_COMPACT_EVERY = 10

# Summary length cap. Set explicitly rather than inherited from the session's
# max_tokens: a predictable compaction cost matters more than letting the
# model write longer — a summary that grows to the size of the original
# history defeats the point of compacting.
SUMMARY_MAX_TOKENS = 600

# Context strategy (day 10, SPEC-w02d10.md §3). Default matches the registry's
# own AGENT_COMMAND default: this constant is what a caller gets when NEITHER
# context_strategy NOR the compact alias were ever set (most unit tests, and
# any config built without apply_defaults()) — day 09's own behavior.
DEFAULT_CONTEXT_STRATEGY = "summary"

# Facts (day 10, SPEC-w02d10.md §5). Fallback when facts_max_tokens is unset,
# mirrors Spec.defaults[AGENT_COMMAND] — see
# test_thresholds_unset_fall_back_to_the_same_numbers_the_registry_hands_out.
DEFAULT_FACTS_MAX_TOKENS = 400

# The extractor's own response ceiling — NOT the block cap above; the block is
# what we STORE, the delta is what the model WRITES to change it. Measured
# worst case (2026-09-12, ministral-14b-latest, t=0): a full FACTS_CATCHUP_MAX
# (20 messages, verbose answers) delta costs 376 completion tokens — 900 is
# ~2.4x headroom, so hitting it means the model looped, not a legitimately
# large delta.
FACTS_RESPONSE_TOKENS = 900

# Day 11 memory extractor and request block caps. These are local settings;
# they must never leak into the Mistral payload.
MEMORY_RESPONSE_TOKENS = 1200
MEMORY_CATCHUP_MAX = 20
DEFAULT_WORKING_MAX_TOKENS = 400
DEFAULT_LONG_TERM_MAX_TOKENS = 300

# Catch-up bound (SPEC-w02d10.md §5.3): past this many not-yet-extracted
# messages, the extractor sees only the tail and the truncation is spoken
# aloud — a silent cut here is exactly the loss this cursor exists to prevent.
FACTS_CATCHUP_MAX = 20

# Fixed schema for the facts delta — lives in advent_core, not injected by the
# caller: unlike summary_prompt/dialog_preset (per-week prose), this is a
# structural contract the extractor call always uses, day 10's PROBE picked
# the pairs-array shape specifically for this file (PROBE-w02d10-facts.md).
FACTS_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "facts_delta.json"
MEMORY_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "memory_delta.json"

# Пометка об обрыве, дописываемая к сохранённой части ответа. Приём взят у
# gptme (INTERRUPT_CONTENT): следующий ход модели должен видеть, что её
# оборвали, а не считать обрубок законченной мыслью.
#
# Пометка кладётся ВНУТРЬ сообщения ассистента, а не отдельным system-ходом
# посреди истории: system в середине диалога — это то, что каждый провайдер
# трактует по-своему, а обрыв касается ровно этого ответа и должен ехать
# вместе с ним (в том числе в файл сессии).
INTERRUPT_NOTE = "[ответ прерван пользователем и остался незаконченным]"


@dataclass(slots=True, frozen=True)
class Compaction:
    """One completed history compaction.

    A separate type rather than a pair of numbers on AgentReply: compaction is
    a MODEL CALL with its own usage, and it must be visible as one. Showing
    "the request dropped from 6369 tokens to 1429" while staying silent about
    what the summary itself cost would be misleading — this type exists to
    prevent exactly that (SPEC-w02d09.md §7).

    `covered` — how many messages went into the summary this round.
    `tokens_before`/`tokens_after` — request size before and after
    substitution; None means "could not be counted", not zero.

    `tail` — the history left untouched. `ask()`'s caller doesn't need it (it
    gets the finished `AgentReply.history`), but `/compact` compacts outside a
    turn, and without the tail it would have to repeat split_history itself —
    a second place knowing the pair-boundary rule.
    """

    summary: str
    covered: int
    result: CallResult
    # A list on a frozen dataclass: immutability here is about the fact of
    # compaction, not about protecting the buffer. The tail is copied on
    # input; the CLI owns it after that.
    tail: list[Message] = field(default_factory=list)
    tokens_before: int | None = None
    tokens_after: int | None = None

    def saved(self) -> int | None:
        """Tokens shaved off the request. None — nothing to count from."""
        if self.tokens_before is None or self.tokens_after is None:
            return None
        return self.tokens_before - self.tokens_after


@dataclass(slots=True, frozen=True)
class FactsUpdate:
    """One extractor call's outcome. Mirrors Compaction (SPEC-w02d10.md §5.2-5.3).

    A second paid call per turn must be visible as one, not folded into a bare
    "facts changed" boolean: `result` carries its own usage for the journal
    and `/tokens`, `delta` carries added/updated/removed/blocked/rejected for
    the per-turn note, `covered` is how many not-yet-extracted messages (not
    counting the new user turn) this call swept up.

    `truncated` — the catch-up window exceeded FACTS_CATCHUP_MAX and only the
    tail was sent: some exchange in the middle was never seen by the
    extractor, and the caller must say so, not just show the note.
    """

    facts: dict[str, str]
    delta: DeltaResult
    covered: int
    result: CallResult
    truncated: bool = False


@dataclass(slots=True, frozen=True)
class FactsFailure:
    """A paid extractor call that produced nothing usable — must still be billed."""

    result: CallResult
    reason: str  # "truncated" (hit FACTS_RESPONSE_TOKENS) or "invalid" (bad JSON/schema)


@dataclass(slots=True, frozen=True)
class MemoryFailureResult:
    """Paid memory extractor call that did not produce an applicable delta."""

    result: CallResult
    reason: str


@dataclass(slots=True, frozen=True)
class AgentReply:
    """Результат одного хода агента.

    frozen: ход уже состоялся, и правка полей означала бы, что напечатанное
    разошлось с тем, что реально произошло.

    `history` — история ПОСЛЕ хода, включая этот обмен, и УЖЕ обрезанная:
    именно её надо передать в следующий `ask()`. Полная запись разговора
    живёт в файле сессии (advent_core/session.py) и обрезкой не затрагивается —
    выбрасывание из рабочего контекста и забывание разговора это разные вещи.

    `context_tokens` — сколько заняли сообщения, отправленные в этом ходу;
    None — посчитать не удалось (не ноль!). `context_exact=False` означает
    оценку, и показывать её надо как оценку.

    `dropped_tokens` — сколько токенов освободила обрезка (SPEC §8 требует
    называть выброшенное вслух и в ходах, и в токенах). None — счёт не удался;
    ноль означает ровно «ничего не выбрасывали».
    """

    text: str
    history: list[Message]
    result: CallResult
    dropped: int = 0
    done: bool = False
    context_tokens: int | None = None
    context_exact: bool = False
    dropped_tokens: int | None = None
    # Summary in effect AFTER this turn: same one passed into ask(), or a new
    # one if compaction fired. Lives next to history for the same reason — the
    # agent keeps no memory of its own, the caller does. `history` does NOT
    # include the summary: only real turns; the pseudo-pair is rebuilt fresh
    # from this text on every turn.
    summary: str | None = None
    # Set only when compaction happened on this exact turn.
    compaction: Compaction | None = None

    # Facts in effect AFTER this turn (day 10): the dict to persist as
    # session.state["facts"], regardless of strategy — echoed back unchanged
    # when the strategy isn't "facts", same reasoning as `summary` above.
    facts: dict[str, str] | None = None
    # Cursor to persist as session.state["facts_upto"] — how much of the
    # history handed to the NEXT ask() call (this reply's own `history`) is
    # already reflected in `facts`. Meaningless outside strategy="facts", but
    # always returned so the caller has one consistent field to carry forward.
    facts_upto: int = 0
    # Set only when the extractor call happened on this exact turn.
    facts_update: FactsUpdate | None = None
    # Set only when the extractor call happened AND produced nothing usable —
    # a paid call that must still show up in the token table (§9).
    facts_failed: FactsFailure | None = None
    # How many messages the "window"/"facts" strategy cut from its own tail
    # this turn — the note material for "окно: выпало N сообщений" (SPEC §4).
    # Distinct from `dropped`/`dropped_tokens`: those are the budget-trim
    # safety net underneath ALL FOUR strategies, this is the strategy's own,
    # deliberate forgetting.
    window_dropped: int = 0
    # Day 11 explicit memory strategy. Safe defaults keep all older callers
    # compatible and leave these fields empty outside context_strategy=memory.
    memory: MemorySnapshot | None = None
    memory_upto: int = 0
    memory_update: MemoryUpdate | None = None
    memory_failed: MemoryFailure | None = None


def marker_instruction(kind: str, needle: str) -> str:
    """Инструкция про признак завершения целевого диалога.

    Формат признака задаётся параметром `done`, а не зашит: text — подстрока в
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


def done_conflicts_with_stop(done: str | None, stop: Sequence[str] | None) -> str | None:
    """Проверяет, не съест ли `stop` маркер завершения. Текст проблемы или None.

    Ловушка из CLAUDE.md, оплаченная в неделе 01: «The output will not contain
    the stop sequence» — API вырезает стоп-последовательность из ответа. Если
    маркер завершения совпадает со стоп-строкой ИЛИ содержит её как подстроку,
    до вызывающего кода маркер не доедет никогда, и целевой диалог не
    закончится ни разу — молча, без единой ошибки.

    Пересечение проверяется в обе стороны: stop внутри маркера обрежет ответ
    на полуслове, маркер внутри stop — тот же случай под другим углом.

    Для `done=json:<поле>` опасна ЛЮБАЯ непустая стоп-строка, а не только
    пересекающаяся с именем поля: маркер вынимается из разобранного JSON
    целиком, а обрезанный по stop ответ (`{"ready": true, "result": {`) до
    formats.is_done() доезжает как JSONDecodeError, который тот штатно
    трактует как «ещё не готово». Диалог не закончится ни разу — молча, без
    единой ошибки, ровно то, ради чего эта проверка и написана.
    """
    if not done or not stop:
        return None
    try:
        kind, needle = formats.parse_done(done)
    except ConfigError:
        # Негодный done — отдельная беда, про неё говорит check_done().
        return None
    if kind == "json":
        items = [item for item in stop if item]
        if not items:
            return None
        return (
            f"маркер завершения ищется в поле {needle!r} разобранного JSON, а стоп-строка "
            f"{items[0]!r} обрежет ответ на полуслове — разбор станет невозможен и диалог "
            "не закончится ни разу; убери stop"
        )
    for item in stop:
        if not item:
            continue
        if item in needle or needle in item:
            return (
                f"стоп-строка {item!r} пересекается с маркером завершения {needle!r} — "
                "API вырезает stop из ответа, и детектировать станет нечего; "
                "разведи их или убери stop"
            )
    return None


@dataclass(slots=True)
class _Trimmed:
    """Результат обрезки: что осталось, сколько выброшено, сколько заняло.

    `tokens` — размер ИТОГОВОГО запроса, `dropped_tokens` — сколько токенов
    освободила обрезка. Это разные числа, и наружу нужны оба: SPEC §8 требует
    называть выброшенное вслух «сколько ходов и сколько токенов», а без второго
    числа «выброшено 6 сообщений» не отличает 200 освобождённых токенов от
    20 000. None — посчитать не удалось (не ноль).
    """

    history: list[Message]
    dropped: int
    tokens: int | None
    dropped_tokens: int | None = None


class Agent:
    """Один агент: конфиг, шов вызова, счётчик токенов, режим.

    `complete`/`stream` передаются в конструктор ЯВНО и не имеют дефолта.
    Дефолт аргумента связывается на импорте, и
    `monkeypatch.setattr(chat_core, "complete", fake)` до него не дотягивается
    — ровно эта ловушка стоила дня 03 (CLAUDE.md, «A default argument binds at
    import time»). Повторять её нельзя.
    """

    def __init__(
        self,
        config: Config,
        *,
        complete: CompleteFn,
        stream: StreamFn,
        counter: TokenCounter | None = None,
        capabilities: dict | None = None,
        context_limit: int | None = None,
        persona: str | None = None,
        dialog_preset: str | None = None,
        summary_prompt: str | None = None,
        facts_prompt: str | None = None,
        memory_prompt: str | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._complete = complete
        self._stream = stream
        self.counter = counter
        self.capabilities = capabilities
        # Размер окна модели из карточки `/v1/models`. None — неизвестно, и
        # тогда обрезка идёт по прежнему символьному бюджету с пометкой об
        # этом. Выдуманный лимит был бы хуже отсутствующего.
        self.context_limit = context_limit
        # Персона агента. None — берём system prompt проекта; пустая строка —
        # осознанный отказ от персоны.
        self.persona = persona
        # Пресет режима dialog: шаблон с плейсхолдером %%MARKER%%. Живёт в
        # неделе (week_02/prompts), а не в core: core не знает про промпты
        # недели и не должен читать её файлы.
        self.dialog_preset = dialog_preset
        # System prompt for the summarizer model. None — nothing to compact
        # with, and the mode turns itself off with a warning: a silently
        # "compacting" agent that actually just trims is worse than one that's
        # off.
        #
        # The agent's persona is deliberately NOT mixed in here: "ask one
        # clarifying question at a time" has nothing to do with summarizing,
        # and the rule "a day that measures quality must drop the persona"
        # was already paid for in week 01 (CLAUDE.md).
        self.summary_prompt = summary_prompt
        # System prompt for the facts extractor (advent_core/prompts/facts.md,
        # read by the caller — core doesn't read week-specific files, and
        # unlike FACTS_SCHEMA_PATH this prose could plausibly change per
        # deployment). None — strategy="facts" degrades to a plain window
        # with a warning, same shape as summary_prompt above.
        self.facts_prompt = facts_prompt
        self.memory_prompt = memory_prompt
        # A compaction that has been paid for but whose turn hasn't finished
        # yet. ask() parks it here so a failing model call can't take it down
        # with it — see the comment at the assignment.
        self.pending_compaction: Compaction | None = None
        # Same idea, for the facts extractor call (SPEC-w02d10.md §5.3).
        self.pending_facts: FactsUpdate | None = None
        # A paid extractor call that produced nothing usable — same parking
        # reasoning, so it still gets billed even if the turn then fails.
        self.pending_facts_failed: FactsFailure | None = None
        self.pending_memory: MemoryUpdate | None = None
        self.pending_memory_failed: MemoryFailure | None = None
        self._on_warning = on_warning
        # Одинаковые предупреждения не повторяются каждый ход: в разговоре на
        # двадцать реплик «лимит окна неизвестен» двадцать раз — это шум, в
        # котором тонет то, что случилось только что.
        self._said: set[str] = set()

    # --- предупреждения наверх -------------------------------------------

    def _warn(self, text: str | None, *, once: bool = False) -> None:
        if not text or self._on_warning is None:
            return
        if once:
            if text in self._said:
                return
            self._said.add(text)
        self._on_warning(text)

    # --- режим и промпт ---------------------------------------------------

    @property
    def mode(self) -> str:
        return self.config.params.mode or "chat"

    @property
    def max_turns(self) -> int:
        """Потолок ходов целевого диалога. None в параметрах — это дефолт."""
        return self.config.params.max_turns or DEFAULT_MAX_TURNS

    def turn_limit_reached(self, turns_done: int) -> bool:
        """Пора ли останавливать целевой диалог.

        Считает ходы вызывающий код, а не агент, и это не лень: целевой диалог
        идёт в ОБЩЕЙ памяти сессии (SPEC-w02d06.md §9), поэтому «сколько ходов
        сделано» нельзя вывести из длины истории — там лежит и всё, что
        обсуждали до диалога. Правило же (сколько всего можно и что значит
        None) живёт здесь, в одном месте.
        """
        return turns_done >= self.max_turns

    def check_done(self) -> str | None:
        """Проблема с настройкой завершения диалога, либо None.

        Зовётся и из ask(), и из CLI — на старте и после `/set done`, чтобы
        пользователь узнал о негодном маркере до того, как диалог не
        закончится.
        """
        done = self.config.params.done
        if self.mode != "dialog":
            return None
        if not done:
            return "mode=dialog требует done — /set done text:<строка> или json:<поле>"
        try:
            formats.parse_done(done)
        except ConfigError as exc:
            return str(exc)
        return done_conflicts_with_stop(done, self.config.params.stop)

    def _done_condition(self) -> tuple[str, str] | None:
        done = self.config.params.done
        if self.mode != "dialog" or not done:
            return None
        try:
            return formats.parse_done(done)
        except ConfigError:
            return None

    def system_prompt(self) -> str | None:
        """system агента: персона, поверх неё — пресет режима.

        Слой формата (`format=json/schema/yaml/md`) сюда НЕ дописывается: его
        добавляет chat._payload() в единственном месте перед отправкой, и
        дублировать его здесь значило бы получить инструкцию формата дважды
        (SPEC-w01d02.md §4).
        """
        persona = self.persona if self.persona is not None else self.config.system_prompt()
        parts = [persona] if persona else []

        condition = self._done_condition()
        if condition is not None:
            kind, needle = condition
            instruction = marker_instruction(kind, needle)
            if self.dialog_preset:
                parts.append(self.dialog_preset.replace("%%MARKER%%", instruction))
            else:
                # Пресет не передали — работаем на одной инструкции про
                # маркер. Диалог от этого не ломается, но ведёт себя беднее,
                # и молчать об этом нельзя.
                self._warn(
                    "пресет режима dialog не задан — в system уходит только "
                    "инструкция про маркер завершения",
                    once=True,
                )
                parts.append(instruction)

        return "\n\n".join(part for part in parts if part) or None

    # --- обрезка контекста -------------------------------------------------

    def _token_budget(self) -> int | None:
        """Сколько токенов можно занять запросом. None — лимит неизвестен."""
        if not self.context_limit:
            return None
        reserve = self.config.params.max_tokens or RESPONSE_RESERVE_TOKENS
        budget = self.context_limit - reserve
        # Запас больше окна — сама по себе странная конфигурация, но падать
        # из-за неё нельзя: оставляем неотрицательный порог, обрезка честно
        # выбросит всё, что сможет, и вызывающий код это увидит по dropped.
        return max(budget, 0)

    def _trim(
        self,
        history: Sequence[Message],
        system: str | None,
        user_input: str,
        *,
        head: Sequence[Message] = (),
    ) -> _Trimmed:
        """Обрезает историю под окно модели. Пары user+assistant — целиком.

        `head` — messages that go into the request but must not be dropped:
        the summary pseudo-pair. It's the compressed form of everything
        already dropped, and dropping it would lose the whole conversation
        instead of just its tail. `head` is not returned in `_Trimmed.history`
        — only real turns, which the caller uses to build the next turn's
        history.

        Порог в токенах — то, ради чего в дне 06 появился счётчик: у недели 01
        он был в символах, потому что считать было нечем. Оба пути живут
        рядом намеренно: счётчика может не быть (тесты, отказ токенизатора), а
        `max_context_length` может не прийти в карточке модели — и тогда
        символьный бюджет остаётся единственным, что не даст истории дорасти
        до ошибки context length.
        """
        budget = self._token_budget()
        if self.counter is not None and budget is not None:
            trimmed = list(history)
            dropped = 0
            # Размер запроса ДО первой выброшенной пары: разница с итоговым и
            # есть объём выброшенного, который SPEC §8 требует назвать вслух.
            before: int | None = None
            while True:
                messages = chat_core.build_messages(
                    user_input, system=system, history=[*head, *trimmed]
                )
                tokens = self.counter.count(messages)
                if tokens is None:
                    # Посчитать не удалось — не притворяемся, что влезло:
                    # уходим на символьный путь, он хотя бы что-то гарантирует.
                    break
                if before is None:
                    before = tokens
                if tokens <= budget or not trimmed:
                    # `not trimmed`: истории больше нет, а запрос всё равно не
                    # влезает — выбрасывать нечего, вопрос уйдёт как есть и
                    # сервер скажет своё. Молча крутиться в цикле нельзя.
                    return _Trimmed(trimmed, dropped, tokens, dropped_tokens=before - tokens)
                count = min(2, len(trimmed))
                del trimmed[:count]
                dropped += count

        # `not self.context_limit`, а не `is None`: нулевое окно — это то же
        # «неизвестно», и _token_budget() трактует его так же.
        if not self.context_limit:
            self._warn(
                "max_context_length модели неизвестен — обрезка идёт по символьному "
                "бюджету, заполненность окна показать не с чем",
                once=True,
            )
        else:
            # Окно известно, а порог всё равно символьный — сказать об этом
            # обязаны: иначе история режется по бюджету в 24 000 символов
            # (~8 000 токенов) при окне на 128 000, а пользователю сообщают
            # только «выброшено сообщений N», не сказав, по какому правилу.
            reason = (
                "счётчика токенов нет"
                if self.counter is None
                else "токенизатор отказался считать этот запрос"
            )
            self._warn(
                f"{reason} — обрезка идёт по символьному бюджету, а не по окну модели "
                f"({self.context_limit}): история может резаться раньше, чем нужно",
                once=True,
            )
        trimmed_chars, dropped_chars = chat_core.trim_history(list(history))
        messages = chat_core.build_messages(
            user_input, system=system, history=[*head, *trimmed_chars]
        )
        tokens = self.counter.count(messages) if self.counter is not None else None
        return _Trimmed(trimmed_chars, dropped_chars, tokens)

    # --- context strategy (day 10) -----------------------------------------

    @property
    def context_strategy(self) -> str:
        """Effective strategy for request assembly this turn. Defaults to summary.

        Reads only `context_strategy` — the `compact` alias (SPEC-w02d10.md
        §3) is a `/set compact off|on` translation the CLI performs by
        writing an explicit `context_strategy` into params; this property
        must not re-derive it, or a bare `compact=False` (day 09's own
        "compaction disabled" spelling, still used by every day-06..09 test)
        would silently reroute into "window" and cut history nobody asked
        to cut.
        """
        return self.config.params.context_strategy or DEFAULT_CONTEXT_STRATEGY

    # --- history compaction --------------------------------------------------

    def compact_enabled(self) -> bool:
        """Whether compaction is on. Nothing to compact with counts as off.

        The summarizer prompt comes from outside (a file under
        advent_core/prompts, read by the CLI). Without it compaction can't
        run, and pretending it does is worse: the user would see "compact on"
        while plain trimming runs underneath. Also off outright when an
        explicit non-summary strategy is in effect — `compact` staying at its
        old default must not fire the summarizer under "window"/"facts"/"branch".
        """
        if self.context_strategy != "summary":
            return False
        if not self.config.params.compact:
            return False
        if not self.summary_prompt:
            self._warn(
                "сжатие истории включено, но промпт суммаризатора не задан — "
                "работает обычная обрезка",
                once=True,
            )
            return False
        return True

    @property
    def keep_last(self) -> int:
        """How many recent messages stay untouched. None — default."""
        value = self.config.params.keep_last
        return DEFAULT_KEEP_LAST if value is None else value

    @property
    def compact_every(self) -> int:
        """Scheduled-compaction threshold, in messages. None — default."""
        value = self.config.params.compact_every
        return DEFAULT_COMPACT_EVERY if value is None else value

    @property
    def facts_max_tokens(self) -> int:
        """Soft cap on the facts block, in tokens. None — default."""
        value = self.config.params.facts_max_tokens
        return DEFAULT_FACTS_MAX_TOKENS if value is None else value

    @property
    def working_max_tokens(self) -> int:
        value = self.config.params.working_max_tokens
        return DEFAULT_WORKING_MAX_TOKENS if value is None else value

    @property
    def long_term_max_tokens(self) -> int:
        value = self.config.params.long_term_max_tokens
        return DEFAULT_LONG_TERM_MAX_TOKENS if value is None else value

    @property
    def memory_max_tokens(self) -> int:
        value = self.config.params.memory_max_tokens
        return MEMORY_RESPONSE_TOKENS if value is None else value

    def _count_request(
        self,
        summary: str | None,
        history: Sequence[Message],
        system: str | None,
        user_input: str,
    ) -> int | None:
        """Request size with the summary substituted in. None — nothing to count."""
        if self.counter is None:
            return None
        messages = chat_core.build_messages(
            user_input, system=system, history=[*summary_messages(summary), *history]
        )
        return self.counter.count(messages)

    def _summarize(self, previous: str | None, older: Sequence[Message]) -> CallResult | None:
        """A separate model call: summarize the old part of the conversation.

        A call error does NOT kill the turn — the only place in the agent
        where AdventError is downgraded to a warning. Compaction is an
        optimization, not a precondition: on failure plain trimming still
        works, and there's no reason to lose the user's already-typed
        question over it. ask()'s own error still propagates up.
        """
        messages: list[Message] = [
            {"role": "system", "content": self.summary_prompt or ""},
            {"role": "user", "content": fold_summary(previous, older)},
        ]
        # Session params don't fit this side call: format=json would force the
        # summarizer to answer with an object, stop would cut the summary
        # mid-word, and the user's max_tokens isn't this task's budget. A copy
        # of the config, not an in-place edit: config is shared with the CLI,
        # and an in-place edit would leak into the next ordinary turn.
        params = replace(
            self.config.params,
            max_tokens=SUMMARY_MAX_TOKENS,
            format=None,
            schema_file=None,
            stop=None,
        )
        try:
            return self._complete(replace(self.config, params=params), messages, self.capabilities)
        except AdventError as error:
            self._warn(f"сжатие истории не удалось ({error.message}) — история будет обрезана")
            return None

    def _compact(
        self,
        previous: str | None,
        older: Sequence[Message],
        tail: Sequence[Message],
        before: int | None,
        system: str | None,
        user_input: str,
    ) -> Compaction | None:
        """The compaction itself: call the summarizer, assemble a Compaction.

        Shared body for scheduled compaction inside ask() and for `/compact`:
        they differ only in the trigger condition, not in what happens. None —
        compaction didn't happen (call failed or the summary was empty), and
        the caller must carry on with the previous summary and history.
        """
        result = self._summarize(previous, older)
        if result is None:
            return None
        text = (result.text or "").strip()
        if not text:
            # An empty response is not a summary. Substituting it would replace
            # the old part of the conversation with nothing — exactly what
            # compaction is supposed to avoid.
            self._warn("суммаризатор вернул пустой пересказ — история будет обрезана")
            return None
        return Compaction(
            summary=text,
            covered=len(older),
            result=result,
            tail=list(tail),
            tokens_before=before,
            tokens_after=self._count_request(text, tail, system, user_input),
        )

    def take_pending_compaction(self) -> Compaction | None:
        """A paid compaction whose turn then failed. Hands it over exactly once."""
        pending = self.pending_compaction
        self.pending_compaction = None
        return pending

    def compact_now(
        self,
        summary: str | None,
        history: Sequence[Message],
        *,
        user_input: str = "",
    ) -> Compaction | None:
        """Compact the history right now, bypassing triggers. None — failed.

        Same path as in ask(), minus the should_compact() condition: `/compact`
        is an explicit request, and "compact because asked" vs. "compact
        because it piled up" are different events — folding them into one
        condition would give the user a button that sometimes silently does
        nothing.

        `user_input` is empty: compaction outside a turn doesn't know the next
        question, so the before/after numbers are counted against the request
        without it — doesn't affect the comparison, it's the same on both sides.
        """
        if not self.compact_enabled():
            # Three different reasons, three different remedies.
            # compact_enabled() returns False when: an EXPLICIT
            # context_strategy overrides summary (checked first — it wins
            # over the compact alias per SPEC §3, so naming the alias's
            # remedy here would be wrong); compact is off; or the summarizer
            # prompt failed to load (`/set compact on` re-toggles a setting
            # that is already on and fixes nothing).
            explicit_strategy = self.config.params.context_strategy
            if explicit_strategy is not None and explicit_strategy != "summary":
                self._warn(
                    f"context_strategy={explicit_strategy} — сжатие решает только "
                    "summary, переключись на него: /set context_strategy summary"
                )
            elif not self.config.params.compact:
                self._warn("сжатие выключено — /set compact on включит его")
            else:
                self._warn("сжимать нечем: промпт суммаризатора не загрузился")
            return None
        system = self.system_prompt()
        older, tail = split_history(history, self.keep_last)
        if not older:
            self._warn(
                f"сжимать нечего: в истории {len(history)} сообщений, "
                f"а хвост keep_last={self.keep_last} остаётся как есть"
            )
            return None
        before = self._count_request(summary, history, system, user_input)
        return self._compact(summary, older, tail, before, system, user_input)

    # --- sticky facts (day 10) ----------------------------------------------

    def _probe_tokens(self, messages: Sequence[Message]) -> int | None:
        """Marginal cost of a message list. None — nothing to count with.

        Probe technique (CLAUDE.md, days 08-09): a bare `count(messages)`
        would include request-level overhead (chat-template framing) as if it
        were part of the content's own weight, and the exact tokenizer refuses
        a list that doesn't end in role=user. Appending an empty user message
        and subtracting its own cost fixes both: `count([*messages,
        empty]) - count([empty])` leaves only the content's own weight.
        """
        if self.counter is None:
            return None
        empty = [{"role": "user", "content": ""}]
        with_content = self.counter.count([*messages, *empty])
        overhead = self.counter.count(empty)
        if with_content is None or overhead is None:
            return None
        return max(0, with_content - overhead)

    def _facts_block_tokens(self, facts: dict[str, str], pinned: Iterable[str]) -> int | None:
        """Size of the current facts block. None — nothing to count with."""
        block = format_facts(facts, pinned)
        if not block:
            return 0
        return self._probe_tokens([{"role": "user", "content": block}])

    def _facts_budget(self) -> int | None:
        """Token budget for the extractor's OWN request. None — window unknown.

        Mirrors `_token_budget()` but reserves FACTS_RESPONSE_TOKENS, the
        extractor's own response ceiling, not the main turn's.
        """
        if not self.context_limit:
            return None
        return max(self.context_limit - FACTS_RESPONSE_TOKENS, 0)

    def _fit_catchup_segment(self, segment: list[Message]) -> tuple[list[Message], bool]:
        """Longest tail of an uncapped backfill segment that fits the extractor's
        own budget (SPEC-w02d10.md §5.6) — message-count caps stop mattering
        once the model's own window does. None counter/window — nothing to
        compare against, so the whole segment goes through untouched.
        """
        budget = self._facts_budget()
        if budget is None:
            return segment, False
        size = self._probe_tokens(segment)
        if size is None or size <= budget:
            return segment, False
        trimmed = list(segment)
        while len(trimmed) > 1:
            del trimmed[0]
            size = self._probe_tokens(trimmed)
            if size is not None and size <= budget:
                break
        return trimmed, True

    def _facts_call(
        self, facts: dict[str, str], pinned: Iterable[str], segment: Sequence[Message]
    ) -> CallResult | None:
        """The extractor request itself — mirrors _summarize().

        A call error does NOT kill the turn: warn, and the caller keeps the
        old facts. Config is a COPY (`replace`), same reasoning as the
        summarizer: format=json would force an object of the wrong shape,
        stop would cut the delta mid-word, and the user's max_tokens isn't
        this call's budget.
        """
        size = self._facts_block_tokens(facts, pinned)
        squeeze = size is not None and size > self.facts_max_tokens
        block = format_facts(facts, pinned) or "(пока пусто)"
        parts = [f"Текущие facts:\n{block}", "Новый обмен:"]
        for message in segment:
            label = ROLE_LABELS.get(message["role"], message["role"])
            parts.append(f"{label}: {message['content']}")
        if squeeze:
            # SPEC-w02d10.md §5.5: code never deletes a fact to make room —
            # only the extractor (by rewording) or a human (/fact del) does.
            parts.append(
                f"Блок facts больше {self.facts_max_tokens} токенов — "
                "уплотняй формулировки в новых значениях."
            )
        messages: list[Message] = [
            {"role": "system", "content": self.facts_prompt or ""},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        params = replace(
            self.config.params,
            max_tokens=FACTS_RESPONSE_TOKENS,
            format="schema",
            schema_file=str(FACTS_SCHEMA_PATH),
            stop=None,
            # Extraction has one right answer; sampling at the session's own
            # temperature is the likeliest cause of the degenerate loop that
            # hit FACTS_RESPONSE_TOKENS twice on a live run — a hypothesis
            # (t=0 probes, 2026-09-12, never looped), not a proven cause.
            temperature=0,
        )
        try:
            return self._complete(replace(self.config, params=params), messages, self.capabilities)
        except AdventError as error:
            self._warn(f"извлечение фактов не удалось ({error.message}) — факты прежние")
            return None

    def _run_facts(
        self,
        facts: dict[str, str],
        pinned: Iterable[str],
        history: Sequence[Message],
        user_input: str,
        facts_upto: int,
        *,
        catchup_max: int | None = FACTS_CATCHUP_MAX,
    ) -> tuple[dict[str, str], int, FactsUpdate | None]:
        """Extractor side call: covers every exchange since `facts_upto`.

        Both `facts` and `facts_upto` are returned UNCHANGED on any failure
        (no prompt, call error, unparsable/malformed delta) — SPEC-w02d10.md
        §5.3: advancing the cursor over a turn that was never actually seen
        would let that exchange fall out of the window later, silently and
        for good. The returned cursor is in the coordinate space of `history`
        PLUS one virtual trailing message for `user_input` — the caller
        (ask()) remaps it into the next turn's coordinate space once it knows
        how much of `history` the strategy's own window cut drops.

        `catchup_max=None` is the backfill case (CLI's `/facts backfill`,
        SPEC §5.6): no message-count cap — the WHOLE session is in scope, not
        just the per-turn catch-up window.
        """
        upto = max(0, min(facts_upto, len(history)))
        if not self.facts_prompt:
            self._warn(
                "стратегия facts включена, но промпт экстрактора не задан — "
                "блок facts не обновляется",
                once=True,
            )
            return facts, upto, None

        segment: list[Message] = [*history[upto:], {"role": "user", "content": user_input}]
        truncated = False
        if catchup_max is not None and len(segment) > catchup_max:
            segment = segment[-catchup_max:]
            truncated = True
            self._warn(
                f"извлечение фактов отстало больше чем на {catchup_max} сообщений — "
                "берётся только хвост, часть обмена в facts не попадёт"
            )
        elif catchup_max is None:
            # Uncapped ≠ unlimited: a long session could still blow the
            # extractor's own window. Size it and fall back to the longest
            # fitting tail rather than inventing a message-count limit.
            fitted, budget_truncated = self._fit_catchup_segment(segment)
            if budget_truncated:
                truncated = True
                self._warn(
                    f"бюджет экстрактора не тянет весь бэкафилл — покрыты последние "
                    f"{len(fitted)} сообщений из {len(segment)}"
                )
                segment = fitted

        result = self._facts_call(facts, pinned, segment)
        if result is None:
            return facts, upto, None

        try:
            raw = json.loads(result.text or "")
            if not isinstance(raw, dict):
                raise ValueError("верхний уровень не объект")
            outcome = apply_delta(facts, pinned, raw)
        except (ValueError, TypeError) as error:
            # Truncation is OUR ceiling, not the model misbehaving. Saying
            # "вернул невалидную дельту" there blames the wrong party and sends
            # the reader looking at the prompt instead of at max_tokens.
            # complete() never sets `truncated` — that flag belongs to the
            # stream path (an interrupted answer already on screen). A
            # non-stream call says it hit the ceiling through finish_reason.
            if result.finish_reason == "length" or result.truncated:
                self._warn(
                    f"ответ экстрактора обрезан лимитом в {FACTS_RESPONSE_TOKENS} токенов — "
                    "факты прежние, обмен догонится на следующем ходу"
                )
                reason = "truncated"
            else:
                self._warn(f"извлечение фактов вернуло невалидную дельту ({error}) — факты прежние")
                reason = "invalid"
            # Paid call, nothing usable — must still be billed (SPEC §9), so
            # it's parked exactly like pending_facts is on the success path.
            self.pending_facts_failed = FactsFailure(result=result, reason=reason)
            return facts, upto, None

        update = FactsUpdate(
            facts=outcome.facts,
            delta=outcome,
            covered=len(history) - upto,
            result=result,
            truncated=truncated,
        )
        # The cursor advances over the whole segment even when only its tail was
        # sent (`truncated`). Deliberate, and the losing alternative is the
        # obvious one: holding the cursor back makes the next turn's segment
        # longer still, truncated to the same last N — the middle is never seen
        # anyway, the backlog never clears, and every later turn pays the cap.
        # So the choice is between a one-off loss said out loud (the warning
        # above) and a permanent one paid for every turn. `truncated` travels on
        # the update so the caller can say it, which is what SPEC-w02d10.md §5.3
        # requires: not silent.
        return outcome.facts, len(history) + 1, update

    def take_pending_facts(self) -> FactsUpdate | None:
        """A paid facts update whose turn then failed. Hands it over exactly once."""
        pending = self.pending_facts
        self.pending_facts = None
        return pending

    def take_pending_facts_failed(self) -> FactsFailure | None:
        """A paid facts failure whose turn then failed. Hands it over exactly once."""
        pending = self.pending_facts_failed
        self.pending_facts_failed = None
        return pending

    def _memory_call(
        self,
        snapshot: MemorySnapshot,
        segment: Sequence[Message],
    ) -> CallResult | None:
        """Run the side extractor with an isolated, deterministic config."""
        if not self.memory_prompt:
            self._warn(
                "стратегия memory включена, но промпт экстрактора не задан — memory не обновляется",
                once=True,
            )
            return None
        blocks = memory_messages(snapshot)
        body = [
            "Текущая working/long-term memory:",
            *(m["content"] for m in blocks if m.get("role") == "user"),
            "Новые сообщения для routing:",
        ]
        body.extend(f"{m['role']}: {m['content']}" for m in segment)
        params = replace(
            self.config.params,
            max_tokens=self.memory_max_tokens,
            format="schema",
            schema_file=str(MEMORY_SCHEMA_PATH),
            stop=None,
            temperature=0,
        )
        try:
            return self._complete(
                replace(self.config, params=params),
                [
                    {"role": "system", "content": self.memory_prompt},
                    {"role": "user", "content": "\n\n".join(body)},
                ],
                self.capabilities,
            )
        except AdventError as error:
            self._warn(f"извлечение memory не удалось ({error.message}) — memory прежняя")
            # The request reached the provider but did not produce a response.
            # Keep a result-shaped paid-call seam so the CLI can account and
            # journal it just like invalid/truncated extractor output.
            return CallResult(
                model_requested=self.config.model,
                finish_reason="error",
            )

    def _memory_head_parts(
        self, snapshot: MemorySnapshot, summary: str | None
    ) -> dict[str, int | None]:
        """Return marginal token costs for protected memory layers.

        A memory block is measured as a real pseudo-pair, with request-level
        chat-template overhead removed by ``_probe_tokens``.  ``None`` means
        the configured counter cannot measure this request.
        """
        parts: dict[str, int | None] = {}
        for name, value in (
            ("long-term", snapshot.long_term),
            ("working", snapshot.working),
        ):
            block = memory_messages(MemorySnapshot(ShortTermMemory(), value, StructuredMemory()))
            parts[name] = self._probe_tokens(block) if block else 0
        summary_messages = memory_messages(MemorySnapshot(), summary=summary)
        parts["summary"] = self._probe_tokens(summary_messages) if summary else 0
        return parts

    def _check_memory_head(
        self,
        snapshot: MemorySnapshot,
        summary: str | None,
        system: str | None,
        user_input: str,
    ) -> None:
        """Warn on soft layer caps and fail before a main request can overflow.

        Structured layers are protected: the generic history trim may remove
        only raw short-term messages.  If the protected head plus system and
        current input cannot fit in the context budget, sending it anyway
        would turn a deterministic local condition into a paid 400.
        """
        parts = self._memory_head_parts(snapshot, summary)
        for name, limit in (
            ("working", self.working_max_tokens),
            ("long-term", self.long_term_max_tokens),
        ):
            size = parts[name]
            if size is not None and size > limit:
                self._warn(
                    f"{name} memory превышает local cap {limit} токенов ({size}); "
                    "значения не удаляются автоматически"
                )

        budget = self._token_budget()
        if self.counter is None or budget is None:
            return
        head = memory_messages(
            MemorySnapshot(ShortTermMemory(), snapshot.working, snapshot.long_term),
            summary=summary,
        )
        request = chat_core.build_messages(user_input, system=system, history=head)
        tokens = self.counter.count(request)
        if tokens is None or tokens <= budget:
            return
        breakdown = ", ".join(
            f"{name}={value if value is not None else '—'}" for name, value in parts.items()
        )
        raise AdventError(
            "защищённая часть memory не помещается в context window: "
            f"{breakdown}, request={tokens}, budget={budget}"
        )

    @staticmethod
    def _memory_delta(raw: object) -> MemoryDelta:
        if not isinstance(raw, dict) or set(raw) != {"working", "long_term"}:
            raise ValueError("memory delta должен содержать только working и long_term")
        sections: dict[str, tuple[MemoryOperation, ...]] = {}
        for destination in ("working", "long_term"):
            section = raw.get(destination)
            if not isinstance(section, dict) or set(section) != {"set", "delete"}:
                raise ValueError(f"секция {destination} должна содержать set и delete")
            operations: list[MemoryOperation] = []
            for item in section["set"]:
                if not isinstance(item, dict) or set(item) != {"field", "key", "value", "evidence"}:
                    raise ValueError("memory set operation имеет неверную форму")
                operations.append(
                    MemoryOperation(
                        "set", item["field"], item["key"], item["value"], item["evidence"]
                    )
                )
            for item in section["delete"]:
                if not isinstance(item, dict) or set(item) != {"field", "key", "evidence"}:
                    raise ValueError("memory delete operation имеет неверную форму")
                operations.append(
                    MemoryOperation("delete", item["field"], item["key"], None, item["evidence"])
                )
            sections[destination] = tuple(operations)
        return MemoryDelta(working=sections["working"], long_term=sections["long_term"])

    def _run_memory(
        self,
        snapshot: MemorySnapshot,
        history: Sequence[Message],
        user_input: str,
        memory_upto: int,
        *,
        catchup_max: int | None = MEMORY_CATCHUP_MAX,
    ) -> tuple[MemorySnapshot, int, MemoryUpdate | None, MemoryFailure | None]:
        upto = max(0, min(memory_upto, len(history)))
        segment: list[Message] = [*history[upto:], {"role": "user", "content": user_input}]
        if catchup_max is not None and len(segment) > catchup_max:
            segment = segment[-catchup_max:]
            self._warn(
                f"извлечение memory отстало больше чем на {catchup_max} сообщений — "
                "берётся только хвост"
            )
        # The extractor needs structured layers plus the uncovered segment;
        # passing the runtime short-term view here duplicated every covered
        # message in its prompt and defeated ``memory_upto``.
        structured = MemorySnapshot(ShortTermMemory(), snapshot.working, snapshot.long_term)
        result = self._memory_call(structured, segment)
        if result is None:
            return snapshot, upto, None, None
        try:
            if result.finish_reason == "error":
                raise ValueError("memory extractor request failed")
            raw = json.loads(result.text or "")
            delta = self._memory_delta(raw)
            update = apply_memory_delta(
                snapshot,
                delta,
                user_messages=tuple(segment),
                memory_upto=len(history),
                call_result=result,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            reason = (
                "request_failed"
                if result.finish_reason == "error"
                else (
                    "truncated"
                    if result.finish_reason == "length" or result.truncated
                    else "invalid"
                )
            )
            failure = MemoryFailure(result, reason, str(error))
            self.pending_memory_failed = failure
            if reason != "request_failed":
                self._warn(
                    f"извлечение memory вернуло невалидную дельту ({error}) — memory прежняя"
                )
            return snapshot, upto, None, failure
        self.pending_memory = update
        # The extractor saw a virtual new user message, but it is not persisted
        # yet.  Keep pending update's cursor inside the current transcript;
        # ask() promotes it after the main call succeeds.
        return update.snapshot, len(history), update, None

    def take_pending_memory(self) -> MemoryUpdate | None:
        pending = self.pending_memory
        self.pending_memory = None
        return pending

    def take_pending_memory_failed(self) -> MemoryFailure | None:
        pending = self.pending_memory_failed
        self.pending_memory_failed = None
        return pending

    # --- сам ход -----------------------------------------------------------

    def ask(
        self,
        user_input: str,
        history: list[Message],
        *,
        summary: str | None = None,
        facts: dict[str, str] | None = None,
        facts_pinned: Iterable[str] = (),
        facts_upto: int = 0,
        memory: MemorySnapshot | None = None,
        memory_upto: int = 0,
        memory_history: Sequence[Message] | None = None,
        profile: dict[str, str] | None = None,
        task: TaskState | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AgentReply:
        """Один ход: собрать, обрезать, спросить, разобрать.

        `history` не мутируется — возвращается новый список (см. AgentReply).
        Ошибки вызова не глотаются: AdventError уходит наверх, где CLI
        печатает её и продолжает сессию, как REPL недели 01.

        Which of the four assemblies runs is `self.context_strategy` (SPEC-
        w02d10.md §3 table): "summary" keeps day 09's own compaction logic
        untouched; "window"/"facts" cut `history` to `keep_last` themselves
        and differ only in what goes in `head`; "branch" sends `history`
        whole. The per-request budget trim (`_trim`) runs underneath ALL
        FOUR — it's the 400-error safety net, not a strategy.
        """
        self._warn(self.check_done(), once=True)

        system = self.system_prompt()
        strategy = self.context_strategy
        summary_now = summary
        compaction: Compaction | None = None
        facts_now = dict(facts) if facts else {}
        facts_update: FactsUpdate | None = None
        facts_upto_now = facts_upto
        memory_now = memory or MemorySnapshot()
        memory_update: MemoryUpdate | None = None
        memory_failed: MemoryFailure | None = None
        memory_upto_now = memory_upto
        window_dropped = 0
        working: list[Message] = list(history)
        head: list[Message] = []

        if strategy == "summary":
            if self.compact_enabled():
                older, tail = split_history(working, self.keep_last)
                before = self._count_request(summary_now, working, system, user_input)
                budget = self._token_budget()
                # Either side can be unknown (no counter, no window) — then the
                # budget trigger simply doesn't fire, and the scheduled
                # message-count trigger still works. A made-up "doesn't fit"
                # would be worse than none at all.
                over_budget = before is not None and budget is not None and before > budget
                if should_compact(older, over_budget=over_budget, compact_every=self.compact_every):
                    compaction = self._compact(summary_now, older, tail, before, system, user_input)
                    if compaction is not None:
                        summary_now = compaction.summary
                        working = list(compaction.tail)
                        # The summarizer call is already made and already paid
                        # for. If the turn's own call below raises, ask() never
                        # returns and this Compaction would vanish with it: no
                        # journal row, no line in /tokens, and — since the
                        # history it was built from is unchanged — a second
                        # summarizer call on the next attempt. Park it where
                        # the caller can still collect it.
                        self.pending_compaction = compaction
            head = summary_messages(summary_now)

        elif strategy == "window":
            older, tail = split_history(working, self.keep_last)
            working = tail
            window_dropped = len(older)

        elif strategy == "facts":
            older, tail = split_history(working, self.keep_last)
            cut = len(older)
            facts_now, upto_after, facts_update = self._run_facts(
                facts_now, facts_pinned, working, user_input, facts_upto_now
            )
            if facts_update is not None:
                # Same reasoning as pending_compaction: the extractor call is
                # already paid for, and must survive the main call below
                # raising — see take_pending_facts().
                self.pending_facts = facts_update
            if self.facts_prompt and facts_update is None and cut > upto_after:
                # The extractor did NOT cover [upto_after:cut] this turn —
                # cutting to `tail` would drop it for good, exactly the loss
                # facts_upto exists to prevent (SPEC-w02d10.md §5.3). Hold the
                # window back to the uncovered boundary instead: a superset of
                # `tail`, so this turn costs more tokens (the budget trim
                # below still runs underneath and caps the request) — never
                # loses history.
                # Gated on a prompt existing: without one the extractor never
                # runs at all, so holding would not mean "wait for the next
                # attempt" — it would silently turn `facts` into a strategy
                # with no window, paying for the whole history every turn for
                # good. A permanent misconfiguration is not a transient
                # failure: there this degrades to plain `window`, and the
                # once-only warning above is what says so.
                working = working[upto_after:]
                window_dropped = upto_after
                facts_upto_now = 0
                self._warn(
                    "окно придержано: экстрактор ещё не отразил часть истории — "
                    "запрос обойдётся дороже токенами, но ничего не потеряно"
                )
            else:
                working = tail
                window_dropped = cut
                # `upto_after` is in the coordinate space of `working` BEFORE
                # this cut; remap into `tail`'s own space (what the caller
                # will pass back in as `history` next turn) by subtracting
                # what the cut drops.
                facts_upto_now = max(0, upto_after - cut)
            head = facts_messages(facts_now, facts_pinned)

        elif strategy == "memory":
            # Fail locally before paying for an extractor when the current
            # protected head already cannot fit.  A second check below covers
            # a delta that grows either structured layer.
            self._check_memory_head(
                memory_now,
                summary_now,
                system,
                user_input,
            )
            memory_now, memory_upto_now, memory_update, memory_failed = self._run_memory(
                memory_now,
                memory_history if memory_history is not None else working,
                user_input,
                memory_upto_now,
            )
            self._check_memory_head(memory_now, summary_now, system, user_input)
            # Summary and structured blocks are protected; _trim can only
            # remove raw short-term messages from ``working``.
            head = memory_messages(
                MemorySnapshot(ShortTermMemory(), memory_now.working, memory_now.long_term),
                summary=summary_now,
            )

        # "branch": working stays the full history, head stays empty — the
        # budget trim below is its only limiter (SPEC §3, §7.3).

        profile_head = profile_messages(profile) if profile else []
        task_head = task_messages(task)
        strategy_head = head
        protected_head = [*profile_head, *task_head, *strategy_head]
        trimmed = self._trim(working, system, user_input, head=protected_head)
        if task_head and self.counter is not None:
            budget = self._token_budget()
            if budget is not None and trimmed.tokens is not None and trimmed.tokens > budget:
                baseline_final = chat_core.build_messages(
                    user_input,
                    system=system,
                    history=[*profile_head, *strategy_head, *trimmed.history],
                )
                baseline_final_tokens = self.counter.count(baseline_final)
                if baseline_final_tokens is not None and baseline_final_tokens <= budget:
                    raise ConfigurationError(
                        "Task context не помещается в request budget",
                        hint=(
                            f"baseline={baseline_final_tokens}, с task={trimmed.tokens}, "
                            f"budget={budget}; используй /task update или /task clear, "
                            "либо увеличь context_limit"
                        ),
                    )
        # The safety-net trim cuts from the FRONT of `working` — exactly the
        # messages the cursor counts as already extracted. Left uncorrected,
        # facts_upto claims coverage of history that no longer exists, and next
        # turn `min(facts_upto, len(history))` hides the drift by skipping the
        # NEWEST exchange instead: silent permanent loss, which is the one thing
        # facts_upto exists to prevent (SPEC-w02d10.md §5.3). Measured on a
        # 400-token budget: cursor 5 against a 4-message history.
        dropped_by_trim = len(working) - len(trimmed.history)
        if dropped_by_trim > 0:
            facts_upto_now = max(0, facts_upto_now - dropped_by_trim)
        if strategy == "memory":
            structured = memory_messages(
                MemorySnapshot(ShortTermMemory(), memory_now.working, memory_now.long_term),
                summary=summary_now,
            )
            messages = chat_core.build_messages(
                user_input,
                system=system,
                history=[*profile_head, *task_head, *structured, *trimmed.history],
            )
        else:
            messages = chat_core.build_messages(
                user_input, system=system, history=[*protected_head, *trimmed.history]
            )

        if on_chunk is not None and chat_core.should_stream(self.config):
            result = self._stream(self.config, messages, on_chunk, self.capabilities)
        else:
            result = self._complete(self.config, messages, self.capabilities)

        if strategy == "memory" and memory_update is not None:
            # ``_run_memory`` extracted the virtual user message before the
            # main call.  Once that call succeeds, the exchange is persisted
            # as user plus assistant (or user only for an empty response), so
            # return the cursor in the same coordinate space as Session.turns.
            transcript_len = len(memory_history) if memory_history is not None else len(history)
            persisted_len = transcript_len + 1 + bool(result.text)
            memory_update = replace(memory_update, memory_upto=persisted_len)
            memory_upto_now = persisted_len

        # The turn survived: the compaction/facts update now travel in the
        # reply, so the parked copies are nobody's responsibility any more.
        self.pending_compaction = None
        self.pending_facts = None
        self.pending_memory = None
        facts_failed = self.take_pending_facts_failed()
        memory_failed = self.take_pending_memory_failed() or memory_failed

        # Сверять надо с тем, что РЕАЛЬНО ушло в API: слой формата дописывает
        # инструкцию к system внутри chat._payload(), и сверка «до слоя» дала
        # бы стабильную дельту на ровном месте — то есть ложный сигнал «таблица
        # токенизаторов разъехалась». По той же причине context_tokens
        # (посчитанный ДО отправки, когда sent_messages ещё нет) при
        # format != text немного занижен — это цена показа заполненности
        # заранее, и она честнее, чем показ после отправки.
        sent = result.sent_messages or messages
        if self.counter is not None:
            check = reconcile(self.counter, sent, result.usage.prompt_tokens)
            self._warn(check.warning())
            self.counter.calibrate(sent, result.usage.prompt_tokens)

        new_history = [*trimmed.history, {"role": "user", "content": user_input}]
        assistant_text = result.text
        if result.truncated:
            # Сохранённая часть остаётся, но едет с пометкой: следующий ход
            # модели должен видеть обрыв (SPEC-w02d06.md §12).
            assistant_text = f"{assistant_text}\n\n{INTERRUPT_NOTE}".strip()
        if assistant_text:
            new_history.append({"role": "assistant", "content": assistant_text})

        return AgentReply(
            text=result.text,
            history=new_history,
            result=result,
            dropped=trimmed.dropped,
            done=self._detect_done(result.text),
            context_tokens=trimmed.tokens,
            context_exact=bool(self.counter and self.counter.exact and trimmed.tokens is not None),
            dropped_tokens=trimmed.dropped_tokens,
            summary=summary_now,
            compaction=compaction,
            facts=facts_now,
            facts_upto=facts_upto_now,
            facts_update=facts_update,
            facts_failed=facts_failed,
            window_dropped=window_dropped,
            memory=memory_now if strategy == "memory" else None,
            memory_upto=memory_upto_now if strategy == "memory" else 0,
            memory_update=memory_update if strategy == "memory" else None,
            memory_failed=memory_failed if strategy == "memory" else None,
        )

    def _detect_done(self, text: str) -> bool:
        """Сработал ли маркер завершения — по ЦЕЛОМУ ответу.

        Именно по целому, а не по дельтам стрима: «ход завершён» — событие
        другой гранулярности, чем «пришёл кусок текста», и маркер, разорванный
        между двумя чанками, в дельтах не найдётся вовсе. Разведка отдельно
        рекомендует держать эти две гранулярности раздельно
        (openai-agents-python).
        """
        condition = self._done_condition()
        if condition is None:
            return False
        kind, needle = condition
        return formats.is_done(text, kind, needle)


# Экспортируется явно: набор публичных имён модуля — часть контракта с CLI и с
# неделей 03, которая продолжит того же агента.
__all__ = [
    "DEFAULT_COMPACT_EVERY",
    "DEFAULT_CONTEXT_STRATEGY",
    "DEFAULT_FACTS_MAX_TOKENS",
    "DEFAULT_KEEP_LAST",
    "DEFAULT_MAX_TURNS",
    "FACTS_CATCHUP_MAX",
    "FACTS_SCHEMA_PATH",
    "MEMORY_CATCHUP_MAX",
    "MEMORY_RESPONSE_TOKENS",
    "MEMORY_SCHEMA_PATH",
    "INTERRUPT_NOTE",
    "RESPONSE_RESERVE_TOKENS",
    "SUMMARY_MAX_TOKENS",
    "Agent",
    "AgentReply",
    "Compaction",
    "CompleteFn",
    "FactsFailure",
    "FactsUpdate",
    "MemoryFailureResult",
    "StreamFn",
    "done_conflicts_with_stop",
    "marker_instruction",
]
