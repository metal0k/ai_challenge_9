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
from dataclasses import dataclass

from advent_core import chat as chat_core
from advent_core import formats
from advent_core.chat import Message
from advent_core.config import Config, ConfigError
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

    def _trim(self, history: Sequence[Message], system: str | None, user_input: str) -> _Trimmed:
        """Обрезает историю под окно модели. Пары user+assistant — целиком.

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
                messages = chat_core.build_messages(user_input, system=system, history=trimmed)
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
        messages = chat_core.build_messages(user_input, system=system, history=trimmed_chars)
        tokens = self.counter.count(messages) if self.counter is not None else None
        return _Trimmed(trimmed_chars, dropped_chars, tokens)

    # --- сам ход -----------------------------------------------------------

    def ask(
        self,
        user_input: str,
        history: list[Message],
        *,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AgentReply:
        """Один ход: собрать, обрезать, спросить, разобрать.

        `history` не мутируется — возвращается новый список (см. AgentReply).
        Ошибки вызова не глотаются: AdventError уходит наверх, где CLI
        печатает её и продолжает сессию, как REPL недели 01.
        """
        self._warn(self.check_done(), once=True)

        system = self.system_prompt()
        trimmed = self._trim(history, system, user_input)
        messages = chat_core.build_messages(user_input, system=system, history=trimmed.history)

        if on_chunk is not None and chat_core.should_stream(self.config):
            result = self._stream(self.config, messages, on_chunk, self.capabilities)
        else:
            result = self._complete(self.config, messages, self.capabilities)

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
    "DEFAULT_MAX_TURNS",
    "INTERRUPT_NOTE",
    "RESPONSE_RESERVE_TOKENS",
    "Agent",
    "AgentReply",
    "CompleteFn",
    "StreamFn",
    "done_conflicts_with_stop",
    "marker_instruction",
]
