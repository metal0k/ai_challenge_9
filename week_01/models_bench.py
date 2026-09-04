"""Day 05: один и тот же запрос на разных версиях модели — точность, latency,
токены, цена.

Модуль знает всё про день: развёртку модель × задача × прогон, метрики на
ячейку, прайс-лист и печать. CLI (week_01/cli.py) только разбирает флаги и
зовёт `run_bench()` — та же архитектура, что у week_01/temperature.py день
раньше (SPEC-w01d05.md §15), и по той же причине: одна механика на команду, а
не размазанная между CLI и модулем.

Исполнение строго последовательное, без потоков и asyncio — тот же довод, что
у Day 03/04: параллельный запуск смешал бы порядок вывода на видео и упёрся бы
в rate limit (у ministral-14b-latest это 30 запросов/мин, SPEC §2) ровно в
момент записи.

Гигиена замера (снятие персоны, обнуление top_p, чистка stop от маркера)
переиспользует week_01/temperature.heat_hygiene() — она была сделана публичной
именно для этого переиспользования (CLAUDE.md: «переиспользуй существующее»).
День отличается от Day 04 одним: там варьировалась температура при
фиксированной модели, здесь — наоборот, модель при temperature=0
зафиксированной на всю развёртку (SPEC §12), поэтому этот модуль дополнительно
прибивает temperature=0 поверх общей гигиены (см. _bench_hygiene).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.chat import Message
from advent_core.client import capabilities_of
from advent_core.config import Config, ConfigError
from advent_core.journal import log_call
from advent_core.telemetry import CallResult
from week_01 import strategies, temperature
from week_01.strategies import (
    Check,
    CompleteFn,
    Problem,
    build_strategy_system,
    check_answer,
    normalize,
)

WEEK_DIR = Path(__file__).resolve().parent
PRICES_PATH = WEEK_DIR / "prices.json"

# Две задачи дня (SPEC-w01d05.md §11): children — закрытая, даёт колонку
# точности; sky — открытая, объяснительный вопрос без эталона, по ней человек
# сравнивает связность (print_human_review). Порядок содержательный: сначала
# задача с ключом (главный замер §5), затем открытая.
BENCH_PROBLEMS: tuple[str, ...] = ("children", "sky")


def load_bench_problems(
    problem_id: str | None = None, directory: Path | None = None
) -> list[Problem]:
    """Задачи развёртки: обе BENCH_PROBLEMS (None/"all") или одна по id.

    Тонкая обёртка над strategies.load_problem_set() — то же правило разбора
    флага, что и у week_01.temperature.load_temp_problems (Day 04), общий код
    вынесен туда, чтобы не заводить вторую копию (CLAUDE.md).
    """
    return strategies.load_problem_set(BENCH_PROBLEMS, problem_id, directory)


# --------------------------------------------------------------------------
# Цены (SPEC-w01d05.md §10)
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ModelPrice:
    """Цена одной модели, $/1M токенов: обычный input, кэшированный input, output."""

    input: float
    cached_input: float
    output: float
    doc: str = ""


@dataclass(slots=True, frozen=True)
class PriceTable:
    """Прайс-лист week_01/prices.json целиком — с датой проверки и источником.

    checked_on печатается рядом с колонкой цены (§10): API цену не отдаёт,
    значит это второй источник истины — ровно тот класс, на котором проект уже
    обжёгся с maximum=2.0 для температуры (CLAUDE.md), и дату проверки нельзя
    прятать в файл.
    """

    checked_on: date
    source: str
    currency: str
    unit: str
    models: dict[str, ModelPrice]

    def price_of(self, model: str) -> ModelPrice | None:
        """Цена модели или None. None — модель не в прайс-листе (§10: «—», не 0)."""
        return self.models.get(model)


def load_prices(path: Path | None = None) -> PriceTable:
    """Читает week_01/prices.json. ConfigError на любой поломке файла.

    Та же философия, что у strategies._read_problem(): прайс-лист правится
    руками, и опечатка в нём должна читаться как «почини вот этот файл», а не
    как traceback посреди развёртки.
    """
    file = path or PRICES_PATH
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Не читается файл цен {file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Файл цен {file} — не валидный JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Файл цен {file} должен быть JSON-объектом")

    try:
        checked_on = date.fromisoformat(str(data["checked_on"]))
    except (KeyError, ValueError) as exc:
        raise ConfigError(f'{file}: поле "checked_on" отсутствует или не ISO-дата') from exc

    models: dict[str, ModelPrice] = {}
    for name, entry in (data.get("models") or {}).items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{file}: цена модели {name!r} должна быть объектом")
        try:
            models[name] = ModelPrice(
                input=float(entry["input"]),
                cached_input=float(entry["cached_input"]),
                output=float(entry["output"]),
                doc=str(entry.get("doc") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{file}: цена модели {name!r} собрана неверно: {exc}") from exc

    return PriceTable(
        checked_on=checked_on,
        source=str(data.get("source") or ""),
        currency=str(data.get("currency") or ""),
        unit=str(data.get("unit") or ""),
        models=models,
    )


# Порог устаревания — тот же довод, что у самого прайс-листа (§10): цена, за
# которой перестали следить, хуже отсутствующей. 180, а не произвольное иное
# число — совпадает с порогом, зафиксированным в SPEC-w01d05.md §10 и §16.
PRICE_STALE_DAYS = 180


def warn_if_prices_stale(prices: PriceTable, *, today: date | None = None) -> None:
    """Предупреждает в stderr, если checked_on старше PRICE_STALE_DAYS дней.

    `today` — параметр, а не date.today() внутри тела: без инъекции тест на
    «предупреждает при устаревшей цене» был бы либо flaky (правкой даты в
    прайс-листе раз в полгода), либо вообще непроверяемым.
    """
    age = (today or date.today()) - prices.checked_on
    if age.days > PRICE_STALE_DAYS:
        console.warn(
            f"{PRICES_PATH.name}: цены проверялись {prices.checked_on.isoformat()} "
            f"({age.days} дн. назад, порог {PRICE_STALE_DAYS}) — возможно, устарели"
        )


def _call_cost(
    usage_prompt: int | None, usage_completion: int | None, cached: int | None, price: ModelPrice
) -> float | None:
    """Стоимость одного ответа в $ по формуле SPEC §10. None — usage не пришёл.

    cached учитывается (SPEC §10): повторный одинаковый промпт биллится по
    input в 10 раз дешевле, и день как раз крутит один и тот же промпт N раз.
    """
    if usage_prompt is None or usage_completion is None:
        return None
    cached_tokens = cached or 0
    billed_input = max(usage_prompt - cached_tokens, 0)
    cost = (
        billed_input * price.input
        + cached_tokens * price.cached_input
        + usage_completion * price.output
    )
    return cost / 1_000_000


# --------------------------------------------------------------------------
# Один вызов и одна ячейка (модель × задача)
# --------------------------------------------------------------------------

BenchStepHook = Callable[["BenchStep"], None]


@dataclass(slots=True)
class BenchStep:
    """Один состоявшийся вызов развёртки — то, что печать и журнал видят сразу."""

    model: str
    problem: Problem
    run: int
    runs: int
    result: CallResult
    messages: list[Message]
    text: str
    normalized: str
    check: Check | None
    first_in_cell: bool


@dataclass(slots=True, frozen=True)
class BenchCall:
    """Один вызов внутри готовой ячейки — то, из чего Cell считает метрики."""

    index: int
    result: CallResult
    text: str
    normalized: str
    check: Check | None


@dataclass(slots=True, frozen=True)
class Cell:
    """Метрики одной ячейки таблицы — модель × задача, по N прогонам (SPEC §9).

    Медиана, а не среднее — везде, по прямому требованию SPEC §9: одна
    аномально долгая попытка (429 + retry внутри SDK) не должна сдвигать
    итог. Мс/токен — медиана ОТНОШЕНИЙ по прогонам, а не отношение медиан
    (SPEC §9, отдельным пунктом) — считается по каждому вызову свой ms/token
    и только потом берётся медиана списка.

    frozen: ячейка считается один раз по готовому списку вызовов и дальше
    только читается, как Cell в week_01/temperature.py и Totals в
    advent_core/telemetry.py.
    """

    model: str
    problem: Problem
    calls: tuple[BenchCall, ...]

    @property
    def total(self) -> int:
        return len(self.calls)

    @property
    def accuracy(self) -> tuple[int, int] | None:
        """k/N по эталону банка. None — задача open, сверки не было (§9).

        None, а не (0, N): ноль читался бы как «ни разу не угадал», хотя на
        деле сверки не было вовсе — эталона не существует (та же поправка,
        что у Cell.accuracy в week_01/temperature.py).
        """
        if self.problem.open:
            return None
        correct = sum(1 for call in self.calls if call.check is not None and call.check.ok)
        return correct, self.total

    @property
    def latency_ms(self) -> float:
        """Медиана total ms по прогонам — латентность клиента, есть всегда."""
        if not self.calls:
            return 0.0
        return median(call.result.latency_ms for call in self.calls)

    @property
    def output_tokens(self) -> float | None:
        """Медиана completion_tokens. None — usage не пришёл ни разу."""
        values = [
            call.result.usage.completion_tokens
            for call in self.calls
            if call.result.usage.completion_tokens is not None
        ]
        return median(values) if values else None

    @property
    def input_tokens(self) -> float | None:
        """Медиана prompt_tokens. Токен по SPEC §7 постоянен между прогонами —
        медиана и константа совпадают, если usage пришёл на всех вызовах, и
        честно усредняет, если пришёл не на всех."""
        values = [
            call.result.usage.prompt_tokens
            for call in self.calls
            if call.result.usage.prompt_tokens is not None
        ]
        return median(values) if values else None

    @property
    def ms_per_token(self) -> float | None:
        """Медиана ОТНОШЕНИЙ latency_ms/completion_tokens по каждому вызову.

        Не отношение медиан — SPEC §9 требует это явно: отношение медиан дало
        бы другое число и не соответствовало бы ни одному реальному вызову.
        Вызовы с completion_tokens=0 (или без usage) в отношение не входят —
        делить на ноль нечем, а «мс/токен» на нулевой знаменатель не метрика.
        """
        ratios = [
            call.result.latency_ms / call.result.usage.completion_tokens
            for call in self.calls
            if call.result.usage.completion_tokens
        ]
        return median(ratios) if ratios else None

    def median_cost(self, prices: PriceTable) -> float | None:
        """Медиана цены ОДНОГО ответа в $ по прогонам. None — нет цены модели
        или usage не пришёл ни на одном вызове."""
        price = prices.price_of(self.model)
        if price is None:
            return None
        costs = [
            cost
            for call in self.calls
            if (
                cost := _call_cost(
                    call.result.usage.prompt_tokens,
                    call.result.usage.completion_tokens,
                    call.result.usage.cached_tokens,
                    price,
                )
            )
            is not None
        ]
        return median(costs) if costs else None

    def cost_per_1000(self, prices: PriceTable) -> float | None:
        """$ за 1000 ответов (SPEC §10) — иначе колонка состоит из нулей."""
        cost = self.median_cost(prices)
        return None if cost is None else cost * 1000


@dataclass(slots=True)
class ProblemSweep:
    """Итог по одной задаче: ячейка на каждую модель, в порядке --models."""

    problem: Problem
    cells: tuple[Cell, ...]


# --------------------------------------------------------------------------
# Гигиена замера и пауза для моделей с низким лимитом (SPEC §12, §13)
# --------------------------------------------------------------------------


def _bench_hygiene(config: Config) -> Config:
    """Гигиена замера дня 05 — персона/top_p/stop через Day 04, плюс t=0 всегда.

    temperature.heat_hygiene() уже умеет снимать персону (она укорачивает
    ответ вдвое и искажает колонку токенов/цены — CLAUDE.md), обнулять top_p с
    предупреждением и чистить stop от маркера ОТВЕТ: — переиспользуется, а не
    копируется (см. её докстринг, специально сделана публичной под это).

    Отличие от Day 04: там варьировалась температура, здесь она зафиксирована
    в 0 на всю развёртку (SPEC §12) — день сравнивает МОДЕЛИ, а не
    температуру, и t=0 единственное значение, которое спека для bench
    разрешает. Если пользователь задал temperature явно (флаг/--env) —
    предупреждаем и всё равно прибиваем к 0, а не молчим об этом.
    """
    config = temperature.heat_hygiene(config)
    if config.params.temperature not in (None, 0.0):
        console.warn(
            f"temperature={config.params.temperature:g} задан явно, но развёртка bench "
            "всегда идёт при t=0 (SPEC-w01d05.md §12) — день сравнивает модели, "
            "а не температуру"
        )
    return replace(config, params=replace(config.params, temperature=0.0))


# Затравка req/min по моделям аккаунта — измерено 2026-09-04 (SPEC-w01d05.md §2).
#
# Это НЕ источник истины, а значение на ПЕРВЫЙ вызов к модели: заголовок
# x-ratelimit-limit-req-minute приходит только вместе с ответом, поэтому до
# первого ответа паузу считать не по чему. Со второго вызова используется
# живой заголовок, и он всегда перебивает эту таблицу — числа тут устареют,
# как устарел maximum=2.0 для температуры, и на них нельзя опираться дольше
# одного запроса.
#
# Только модели, реально отвечающие 200: у 429/403-моделей вызов и так не
# пройдёт, и записывать им «лимит» значило бы делать вид, что их можно звать.
_SEED_RPM: dict[str, int] = {
    "ministral-3b-latest": 750,
    "ministral-8b-latest": 188,
    "ministral-14b-latest": 30,
    "codestral-latest": 125,
    "voxtral-small-latest": 60,
}
_SEED_RPM_CHECKED_ON = date(2026, 9, 4)


@dataclass(slots=True)
class RateLimits:
    """Что развёртка знает о лимите каждой модели и когда её звали в прошлый раз.

    Лимит у Mistral ПОМОДЕЛЬНЫЙ, а не общий на ключ: 2026-09-04 на одном и том
    же ключе ministral-3b отдавал 750 запросов в минуту, ministral-14b — 30, а
    mistral-small — 0 и 429 в те же секунды. Поэтому и словарь по моделям, а
    не одно число на развёртку.
    """

    limits: dict[str, int] = field(default_factory=lambda: dict(_SEED_RPM))
    live: set[str] = field(default_factory=set)
    last_call_at: dict[str, float] = field(default_factory=dict)

    def observe(self, model: str, rpm: int | None) -> None:
        """Запоминает лимит, прочитанный из заголовка ответа.

        None означает «заголовка не было» — тогда затравка не затирается:
        устаревшее измерение всё же лучше, чем отсутствие паузы совсем.
        Ноль — законное значение и записывается как есть: именно ноль Mistral
        отдаёт для модели, недоступной на тарифе.
        """
        if rpm is None:
            return
        self.limits[model] = rpm
        self.live.add(model)

    def pause_before(self, model: str) -> None:
        """Держит интервал между стартами запросов к ОДНОЙ модели.

        Не короче 60/лимит секунд, и спит ровно недостающее, а не
        фиксированную паузу. Молчит и не спит, если лимит неизвестен вовсе —
        честнее не тормозить, чем тормозить по выдуманному числу.

        Лимит <= 0 паузой не лечится: это «модель недоступна на тарифе», и
        любой сон тут был бы просто потерянным временем перед неизбежным 429.

        `time.sleep` зовётся через модульный атрибут, а не захватывается в
        аргумент по умолчанию: тесты подменяют его
        `monkeypatch.setattr(models_bench.time, "sleep", fake)` и не тратят
        реальные секунды на развёртку с ministral-14b-latest.
        """
        limit = self.limits.get(model)
        last = self.last_call_at.get(model)
        if limit is not None and limit > 0 and last is not None:
            remaining = 60.0 / limit - (time.monotonic() - last)
            if remaining > 0.05:
                source = (
                    "заголовок ответа"
                    if model in self.live
                    else f"замер {_SEED_RPM_CHECKED_ON.isoformat()}, заголовка ещё не было"
                )
                console.note(
                    f"пауза {remaining:.1f} с — {model}: лимит {limit} запрос/мин ({source})"
                )
                time.sleep(remaining)
        self.last_call_at[model] = time.monotonic()


# --------------------------------------------------------------------------
# Развёртка целиком
# --------------------------------------------------------------------------


def _run_cell(
    model: str,
    problem: Problem,
    config: Config,
    runs: int,
    *,
    capabilities: dict[str, Any] | None,
    complete: CompleteFn,
    on_step: BenchStepHook | None,
    limits: RateLimits,
) -> Cell:
    """Прогоняет один и тот же запрос N раз на одной модели — одна ячейка.

    Персона снимается на уровне всей развёртки (_bench_hygiene), здесь только
    собирается system с добавкой маркера, если задача её требует
    (problem.format_check) — та же машинерия Day 03/04
    (strategies.build_strategy_system).
    """
    cell_config = replace(config, model=model)
    marker_needed = problem.format_check == strategies.FORMAT_CHECK_MARKER
    system = build_strategy_system(cell_config, marker=marker_needed)

    calls: list[BenchCall] = []
    for index in range(1, runs + 1):
        limits.pause_before(model)
        messages = chat_core.build_messages(problem.statement, system=system)
        result = complete(cell_config, messages, capabilities)
        # Живой заголовок перебивает затравку начиная со второго вызова.
        limits.observe(model, result.rate_limit_rpm)
        text = result.text
        normalized = normalize(text)
        check = None if problem.open else check_answer(text, problem)

        calls.append(
            BenchCall(index=index, result=result, text=text, normalized=normalized, check=check)
        )

        if on_step is not None:
            on_step(
                BenchStep(
                    model=model,
                    problem=problem,
                    run=index,
                    runs=runs,
                    result=result,
                    # sent_messages — то же обоснование, что в strategies.py и
                    # temperature.py: это то, что реально ушло в API после
                    # дописывания инструкции формата в chat._payload().
                    messages=result.sent_messages or messages,
                    text=text,
                    normalized=normalized,
                    check=check,
                    first_in_cell=index == 1,
                )
            )

    return Cell(model=model, problem=problem, calls=tuple(calls))


def run_bench(
    config: Config,
    problems: Sequence[Problem],
    model_names: Sequence[str],
    runs: int,
    *,
    account_models: list[dict] | None = None,
    complete: CompleteFn = chat_core.complete,
    on_step: BenchStepHook | None = None,
) -> list[ProblemSweep]:
    """Развёртка модель × задача × прогон — ядро дня (SPEC-w01d05.md §1, §8, §9).

    `complete` принимается параметром и передаётся вниз явно, а не как значение
    по умолчанию сигнатуры: аргумент по умолчанию связывается на импорте
    модуля, и monkeypatch (`monkeypatch.setattr(cli.chat_core, "complete",
    fake)`) до него не достаёт — та же ловушка Day 03, записанная в CLAUDE.md.
    CLI обязан звать эту функцию с complete=chat_core.complete явно.

    `account_models` — сырой список моделей аккаунта (advent_core.client.
    list_models()) для отсева параметров по capabilities КАЖДОЙ модели
    отдельно: capabilities судьи на Day 03 уже показали, что чужая карточка
    отправляет параметр не той модели. None/пустой список — capabilities не
    известны, параметры уходят без отсева (тот же fallback, что у
    Session.capabilities в week_01/cli.py).

    Порядок циклов — задача снаружи, модель внутри: ячейки одной задачи идут
    подряд, поэтому и в журнале, и в потоке on_step замеры одной задачи не
    перемешаны с другой. Таблицы (print_bench_table) при этом печатает CLI
    ПОСЛЕ полного возврата развёртки — по мере поступления идут только сами
    ответы, через on_step.
    """
    if runs < 1:
        raise ConfigError(f"runs должен быть не меньше 1, получено {runs}")
    if not problems:
        raise ConfigError("список задач пуст")
    if not model_names:
        raise ConfigError("список моделей пуст")

    config = _bench_hygiene(config)
    limits = RateLimits()

    sweeps: list[ProblemSweep] = []
    for problem in problems:
        cells = [
            _run_cell(
                model,
                problem,
                config,
                runs,
                capabilities=capabilities_of(account_models, model) if account_models else None,
                complete=complete,
                on_step=on_step,
                limits=limits,
            )
            for model in model_names
        ]
        sweeps.append(ProblemSweep(problem=problem, cells=tuple(cells)))

    return sweeps


# --------------------------------------------------------------------------
# Журнал
# --------------------------------------------------------------------------


def log_bench_step(step: BenchStep, *, week: int, day: int) -> None:
    """Пишет ОДИН состоявшийся вызов развёртки в журнал — по факту вызова.

    Единица записи — вызов, а не ячейка: 429 на последнем прогоне ячейки не
    должен унести уже оплаченные (тот же принцип, что у Day 03/04). Зовётся из
    on_step, то есть сразу после успешного complete().
    """
    log_call(
        step.result,
        step.messages,
        week=week,
        day=day,
        extra={
            "model": step.model,
            "problem": step.problem.id,
            "run": step.run,
        },
    )


# --------------------------------------------------------------------------
# Печать
# --------------------------------------------------------------------------


# Потолок на печать одного ответа. Не ограничение генерации: модель отвечает
# сколько хочет, метрики и журнал видят ответ целиком — обрезается ТОЛЬКО
# печать.
#
# Заведено по dry-run перед записью Day 05: ministral-8b на children выдал
# 4205 выходных токенов за 55 секунд — зациклился, раз за разом писал «Ошибка
# в понимании задачи», передоказывал и закончил неверным ответом. finish=stop,
# то есть это не обрыв по max_tokens, а нормальное поведение модели. В выводе
# это заняло 442 строки из 741 — одна ячейка съела бы больше половины ролика.
#
# Ограничивать саму генерацию через max_tokens было бы неверно: обрезанный
# ответ изменил бы и число токенов, и цену, то есть испортил бы ровно те
# метрики, которые день измеряет. Медиана по прогонам такой выброс переживает
# (в замере §19 медиана 8b — 300 токенов), а экран — нет.
ANSWER_PRINT_LINES = 12
# И по символам тоже: splitlines() считает ЛОГИЧЕСКИЕ строки, а ответ может
# прийти одним абзацем без единого перевода строки — тогда лимит по строкам
# не сработает вовсе, а терминал развернёт его на десятки строк переносами.
# 1200 символов — это около 15 строк при ширине 80.
ANSWER_PRINT_CHARS = 1200
# Ниже этого хвоста обрезка не стоит своей пометки: строка «показано начало
# ответа» из-за одного лишнего символа — шум, который ещё и заставляет читателя
# гадать, что он не увидел. Найдено на третьем dry-run: ответ переваливал
# лимит ровно на 1 символ.
ANSWER_PRINT_SLACK = 200


def _shorten_for_screen(text: str) -> tuple[str, int]:
    """Начало ответа для экрана и число скрытых символов (0 — влез целиком).

    Режет по тому пределу, который наступил раньше. Скрытое считается в
    символах, а не в строках: при обрезке по длине «строк» может не быть
    вовсе, и число строк было бы неинформативным.
    """
    shown = text
    lines = shown.splitlines()
    if len(lines) > ANSWER_PRINT_LINES:
        shown = "\n".join(lines[:ANSWER_PRINT_LINES])
    if len(shown) > ANSWER_PRINT_CHARS:
        shown = shown[:ANSWER_PRINT_CHARS].rstrip()
    hidden = len(text) - len(shown)
    return (text, 0) if hidden <= ANSWER_PRINT_SLACK else (shown, hidden)


def print_bench_step(step: BenchStep) -> None:
    """Один вызов развёртки — печатается по мере поступления (SPEC §8).

    Тот же приём, что у temperature.print_heat_step(): первый прогон ячейки
    печатается целиком (ответ — в stdout, контракт проекта), остальные —
    короткой строкой в stderr, иначе N прогонов × M моделей × K задач заливают
    экран сырым текстом.

    «Целиком» — с точностью до ANSWER_PRINT_LINES: см. комментарий там о том,
    почему хвост очень длинного ответа не печатается и почему при этом ничего
    не теряется.
    """
    if step.first_in_cell:
        suffix = f" · {temperature.runs_label(step.runs)}" if step.runs > 1 else ""
        header = f"── {step.model} · {step.problem.id}{suffix} ──"
        console.note(header)
        shown, hidden = _shorten_for_screen(step.text)
        console.write_chunk(shown)
        console.finish_answer()
        if hidden:
            # В stderr: это служебная пометка о печати, а не часть ответа.
            # «симв.» несклоняемым сокращением намеренно: числительное перед
            # полным словом требует согласования («1 символ», «2 символа»,
            # «5 символов»), а число здесь любое.
            console.note(
                f"…показано начало ответа: {len(shown)} из {len(step.text)} симв.; "
                "ответ целиком учтён в метриках и записан в журнал"
            )
        console.footer(step.result)
    else:
        header = f"── {step.model} · {step.problem.id} · прогон {step.run}/{step.runs} ──"
        console.note(f"{header} {rich_escape(step.normalized)}")


def _ms_label(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.1f} s"


def _tokens_label(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _cost_label(value: float | None, prices: PriceTable) -> str:
    return "—" if value is None else f"{value:.3f} {prices.currency}"


def print_bench_table(problem: Problem, cells: Sequence[Cell], prices: PriceTable) -> None:
    """Таблица на задачу — колонки ровно по SPEC-w01d05.md §9.

    «точность» отсутствует у open-задачи вовсе — не 0/N, а пропущенная
    колонка, та же поправка, что у print_cell_table в week_01/temperature.py.
    Дата прайс-листа — в заголовке колонки цены (SPEC §10: «дата проверки
    печатается рядом с колонкой»).
    """
    runs = cells[0].total if cells else 0
    reference = "эталона нет" if problem.open else f"эталон {problem.answer}"
    title = f"{problem.title} · {reference} · {temperature.runs_label(runs)}"

    table = Table(title=title)
    table.add_column("модель", style="cyan", no_wrap=True)
    if not problem.open:
        table.add_column("точность", justify="center")
    table.add_column("latency", justify="right")
    table.add_column("вых. токены", justify="right")
    table.add_column("мс/токен", justify="right")
    table.add_column("вх. токены", justify="right")
    table.add_column(f"$/1000 отв. (цены на {prices.checked_on.isoformat()})", justify="right")

    for cell in cells:
        row = [cell.model]
        if not problem.open:
            correct, total = cell.accuracy or (0, cell.total)
            row.append(f"{correct}/{total}")
        row.append(_ms_label(cell.latency_ms))
        row.append(_tokens_label(cell.output_tokens))
        row.append(_ms_label(cell.ms_per_token))
        row.append(_tokens_label(cell.input_tokens))
        row.append(_cost_label(cell.cost_per_1000(prices), prices))
        table.add_row(*row)

    console.out.print(table)

    if problem.open:
        print_human_review(problem, cells)


def print_human_review(problem: Problem, cells: Sequence[Cell]) -> None:
    """По ОДНОМУ ответу каждой модели рядом — связность сравнивает человек.

    Та же идея, что print_human_review в week_01/temperature.py (Day 04,
    пользовательское решение «оценку — человеку, не судье»): судьи здесь тоже
    нет, и по другой причине (SPEC-w01d05.md §3) — единственная модель,
    годная в судьи (14b, самая сильная в аккаунте), сама участвует в
    сравнении, то есть судила бы саму себя.

    Один ответ на модель, а НЕ все N, — и это отличие от Day 04 не косметика.
    Там ось сравнения — разброс ОДНОЙ модели между прогонами, поэтому все
    прогоны и нужны в кадре. Здесь ось — разные модели, а прогоны существуют
    только чтобы взять медиану метрик; три почти одинаковых ответа на модель
    втрое удлиняют таблицу и хоронят само сравнение. Найдено на dry-run перед
    записью Day 05: блок занимал 255 строк вывода из 537, половину ролика.

    Остальные прогоны не теряются: каждый уже напечатан по мере поступления
    (print_bench_step) и записан в журнал.

    stdout: содержимое — ответы модели, тот же контракт, что у
    print_bench_step.
    """
    if not problem.open or not cells:
        return
    shown = [(cell.model, cell.calls[0]) for cell in cells if cell.calls]
    if not shown:
        return

    runs = max((cell.total for cell in cells), default=0)
    suffix = " (первый прогон из каждой ячейки)" if runs > 1 else ""
    console.out.print(
        f"\n[bold]{problem.id}: ответы моделей рядом{suffix} — оценка связности за человеком[/bold]"
    )

    table = Table()
    for model, _ in shown:
        table.add_column(model, overflow="fold")
    table.add_row(*(rich_escape(call.text.strip() or "(пустой ответ)") for _, call in shown))

    console.out.print(table)


def _names(models: Sequence[str]) -> str:
    if len(models) == 1:
        return models[0]
    return "ничья между " + ", ".join(models)


def _cheapest_by(pairs: Sequence[tuple[str, float]]) -> list[str]:
    """Модели с минимальным значением — с явной ничьёй, а не первой по порядку.

    Тот же приём, что _accuracy_conclusion/_diversity_conclusion в
    week_01/temperature.py (CLAUDE.md, ловушка «max()/min() возвращает первый
    максимум/минимум и на плоской колонке молча называет случайного
    победителя»).
    """
    best = min(value for _, value in pairs)
    return [name for name, value in pairs if value == best]


def _best_by(pairs: Sequence[tuple[str, float]]) -> list[str]:
    """Модели с МАКСИМАЛЬНЫМ значением — зеркало _cheapest_by, с той же ничьёй."""
    best = max(value for _, value in pairs)
    return [name for name, value in pairs if value == best]


def _accuracy_conclusion(cells: Sequence[Cell]) -> list[str]:
    """Точность — первая ось задания, и она обязана прозвучать словами.

    Урок Day 04: метрика, посчитанная в таблице, но не названная в выводе,
    задание не закрывает — зритель не обязан истолковывать колонку сам.

    Открытая задача сюда не попадает вовсе (accuracy is None): у неё нет
    эталона, и «0 из N» читалось бы как «ни разу не угадала» вместо «мерить
    нечем» — та же поправка, что в Cell.accuracy.

    Плоская колонка называется ничьёй, а не победителем по порядку --models.
    Отдельно разобран случай, когда НИ ОДНА модель не ответила верно: там
    «лучшие» формально есть (все с нулём), но называть их лучшими — враньё.
    """
    pairs = [
        (cell.model, correct / total)
        for cell in cells
        if (accuracy := cell.accuracy) is not None and (correct := accuracy[0]) is not None
        for total in [accuracy[1]]
        if total
    ]
    if len(pairs) < 2:
        return []

    best_share = max(share for _, share in pairs)
    if best_share == 0:
        return ["  верно не ответила ни одна модель — по точности задача не разделяет"]

    leaders = _best_by(pairs)
    if len(leaders) == len(pairs):
        return [f"  точность одинакова у всех ({best_share:.0%}) — ничья, победителя нет"]
    return [f"  точнее всего — {_names(leaders)} ({best_share:.0%})"]


def _price_rank_conclusion(cells: Sequence[Cell], prices: PriceTable) -> list[str]:
    """§5 SPEC: цена за токен и цена за 1000 ответов ранжируют модели по-разному.

    Не утверждается как вечная истина — каждый раз проверяется на ЭТОМ
    прогоне: если модель пишет одинаково многословно, разница может и не
    случиться, и тогда так и написано. Без id задачи в тексте строки — его
    печатает заголовок в print_conclusions(), повторять было бы избыточно.
    """
    token_pairs = [
        (cell.model, price.output)
        for cell in cells
        if (price := prices.price_of(cell.model)) is not None
    ]
    response_pairs = [
        (cell.model, cost) for cell in cells if (cost := cell.cost_per_1000(prices)) is not None
    ]
    if len(token_pairs) < 2 or len(response_pairs) < 2:
        return []

    by_token = _cheapest_by(token_pairs)
    by_response = _cheapest_by(response_pairs)
    if set(by_token) == set(by_response):
        return [f"  дешевле всего и за токен, и за 1000 ответов — {_names(by_token)}"]
    return [
        f"  дешевле за токен — {_names(by_token)}, дешевле за 1000 ответов — "
        f"{_names(by_response)} — цена за токен и цена за ответ ранжируют модели по-разному "
        "(модель может писать длиннее или короче)"
    ]


def _latency_rank_conclusion(cells: Sequence[Cell]) -> list[str]:
    """§5 SPEC: latency (весь ответ) и мс/токен ранжируют модели по-разному.

    Latency — не свойство модели саму по себе: многословная модель может быть
    самой медленной целиком и при этом самой быстрой на токен (SPEC §5,
    ministral-8b на живом замере).
    """
    latency_pairs = [(cell.model, cell.latency_ms) for cell in cells]
    per_token_pairs = [
        (cell.model, ratio) for cell in cells if (ratio := cell.ms_per_token) is not None
    ]
    if len(latency_pairs) < 2 or len(per_token_pairs) < 2:
        return []

    by_latency = _cheapest_by(latency_pairs)
    by_per_token = _cheapest_by(per_token_pairs)
    if set(by_latency) == set(by_per_token):
        return [f"  быстрее всего и целиком, и на токен — {_names(by_latency)}"]
    return [
        f"  быстрее целиком — {_names(by_latency)}, быстрее на токен — "
        f"{_names(by_per_token)} — latency не свойство модели, а функция длины ответа"
    ]


def print_conclusions(sweeps: Sequence[ProblemSweep], prices: PriceTable) -> None:
    """Блок «Вывод» — механические факты этого прогона (SPEC §5), без обобщений.

    Ничья называется ничьёй словами, а не победителем по порядку --models —
    тот же принцип, что print_conclusions в week_01/temperature.py.
    """
    console.out.print("\n[bold]Вывод[/bold]")
    for sweep in sweeps:
        lines = _accuracy_conclusion(sweep.cells)
        lines += _price_rank_conclusion(sweep.cells, prices)
        lines += _latency_rank_conclusion(sweep.cells)
        if lines:
            console.out.print(f"{sweep.problem.id}:")
            console.out.print("\n".join(lines))


def print_links(model_names: Sequence[str], prices: PriceTable) -> None:
    """Ссылки — часть результата дня, печатаются в конце (SPEC §14).

    Источник — prices.json: карточка каждой модели (поле doc) лежит рядом с
    ценой, которую она подтверждает, плюс общая страница цен (source).
    """
    console.out.print("\n[bold]Ссылки[/bold]")
    if prices.source:
        console.out.print(f"  цены Mistral: {prices.source}")
    for name in model_names:
        price = prices.price_of(name)
        doc = price.doc if price and price.doc else "—"
        console.out.print(f"  {name}: {doc}")
