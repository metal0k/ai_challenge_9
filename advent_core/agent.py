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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from advent_core import chat as chat_core
from advent_core import formats
from advent_core.chat import Message
from advent_core.compact import (
    fold_summary,
    should_compact,
    split_history,
    summary_messages,
)
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError
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
        # A compaction that has been paid for but whose turn hasn't finished
        # yet. ask() parks it here so a failing model call can't take it down
        # with it — see the comment at the assignment.
        self.pending_compaction: Compaction | None = None
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

    # --- history compaction --------------------------------------------------

    def compact_enabled(self) -> bool:
        """Whether compaction is on. Nothing to compact with counts as off.

        The summarizer prompt comes from outside (a file under
        advent_core/prompts, read by the CLI). Without it compaction can't
        run, and pretending it does is worse: the user would see "compact on"
        while plain trimming runs underneath.
        """
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
            # Two different reasons, two different remedies. compact_enabled()
            # also returns False when compaction is ON but the summarizer
            # prompt failed to load — telling that user to `/set compact on`
            # sends them to re-toggle a setting that is already on.
            if not self.config.params.compact:
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

    # --- сам ход -----------------------------------------------------------

    def ask(
        self,
        user_input: str,
        history: list[Message],
        *,
        summary: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AgentReply:
        """Один ход: собрать, обрезать, спросить, разобрать.

        `history` не мутируется — возвращается новый список (см. AgentReply).
        Ошибки вызова не глотаются: AdventError уходит наверх, где CLI
        печатает её и продолжает сессию, как REPL недели 01.
        """
        self._warn(self.check_done(), once=True)

        system = self.system_prompt()
        summary_now = summary
        compaction: Compaction | None = None
        working: list[Message] = list(history)

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
                    # The summarizer call is already made and already paid for.
                    # If the turn's own call below raises, ask() never returns
                    # and this Compaction would vanish with it: no journal row,
                    # no line in /tokens, and — since the history it was built
                    # from is unchanged — a second summarizer call on the next
                    # attempt. Park it where the caller can still collect it.
                    self.pending_compaction = compaction

        head = summary_messages(summary_now)
        trimmed = self._trim(working, system, user_input, head=head)
        messages = chat_core.build_messages(
            user_input, system=system, history=[*head, *trimmed.history]
        )

        if on_chunk is not None and chat_core.should_stream(self.config):
            result = self._stream(self.config, messages, on_chunk, self.capabilities)
        else:
            result = self._complete(self.config, messages, self.capabilities)

        # The turn survived: the compaction now travels in the reply, so the
        # parked copy is nobody's responsibility any more.
        self.pending_compaction = None

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
    "DEFAULT_KEEP_LAST",
    "DEFAULT_MAX_TURNS",
    "INTERRUPT_NOTE",
    "RESPONSE_RESERVE_TOKENS",
    "SUMMARY_MAX_TOKENS",
    "Agent",
    "AgentReply",
    "Compaction",
    "CompleteFn",
    "StreamFn",
    "done_conflicts_with_stop",
    "marker_instruction",
]
