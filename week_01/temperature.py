"""Day 04: один и тот же запрос при разных температурах — точность, разнообразие,
креативность.

Модуль знает всё про день: развёртку задача × температура × прогон, метрики
на ячейку и печать. CLI (week_01/cli.py) только разбирает флаги и зовёт
`run_temperature_sweep()` — вся механика здесь, по той же причине, что и в
week_01/strategies.py день раньше (SPEC-w01d04.md §11).

Исполнение строго последовательное, без потоков и asyncio (SPEC-w01d04.md §3):
параллельный запуск смешал бы порядок вывода на видео и упёрся бы в rate limit
ровно в момент записи — тот же довод, что и у Day 03.

Функции подсчёта (Cell.*) и функции печати (print_*) разделены сознательно:
print_* только форматируют уже готовые значения, ничего не вычисляют, а Cell
ничего не печатает. Это и держит обещание модуля — тесты дёргают метрики без
rich и без stdout/stderr.

Креативность у open-задач оценивает человек, а не LLM-судья: решение
пользователя. Раньше здесь был отдельный судья дня 04 (своя схема, свой
промпт) — он убран целиком; вместо ранжирования и балла день печатает сами
ответы всех N прогонов рядом (print_human_review), чтобы сравнить их одним
взглядом (см. докстринг print_human_review). Судья Day 03 (week_01/strategies.py,
команда `solve`) этим не затронут — это отдельная, самостоятельная машинерия.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.chat import Message
from advent_core.config import Config, ConfigError
from advent_core.journal import log_call
from advent_core.telemetry import CallResult, Totals
from week_01 import strategies
from week_01.strategies import (
    Check,
    CompleteFn,
    Notify,
    Problem,
    build_strategy_system,
    check_answer,
    normalize,
)

# Тройка задач дня — сценарий дня, а не свойство банка (SPEC-w01d04.md §6):
# alice и digits5 принадлежат и Day 03, привязка тут — просто "какие задачи
# показывает demo `temp`". Задать это флагом в файле задачи значило бы
# перепутать свойство задачи со сценарием конкретного дня.
#
# Порядок содержательный, не алфавитный и не случайный — офлайн-sweep 10
# прогонов на mistral-small-latest (SPEC-w01d04.md §2, проверено 2026-09-03):
#
#   digits5 (эталон 225, модель обычно права):
#     t=0.0 верно 10/10   t=0.7 верно 8/10   t=1.2 верно 5/10
#   alice   (эталон 3, модель обычно ошибается):
#     t=0.0 верно  0/10   t=0.7 верно 2/10   t=1.2 верно 5/10
#
# Сначала digits5, где рост температуры портит точность — это и есть
# обычное, "учебниковое" ожидание. Затем alice, где рост температуры её
# улучшает — результат ровно обратный. Обе сходятся к 5/10 при t=1.2, но
# идут туда с разных концов. Порознь каждая задача показывает НЕВЕРНОЕ
# обобщение ("температура снижает точность" или "температура повышает
# точность") — только пара вместе доказывает вывод дня: температура не
# делает ответ точнее или менее точным сама по себе, она увеличивает разброс
# вокруг наиболее вероятного ответа модели, и есть ли от этого польза,
# зависит от того, верен ли этот самый вероятный ответ. Убирать одну задачу
# из пары нельзя — это ломает единственное, что доказывает день.
#
# coffee — третья и последняя: creativity/разнообразие, где эталона нет
# вовсе, поэтому её нельзя перепутать с парой выше.
TEMP_PROBLEMS: tuple[str, ...] = ("digits5", "alice", "coffee")

# Колбэк «вызов состоялся» — тот же контракт, что StepHook на Day 03: зовётся
# сразу после успешного complete(), до следующего вызова развёртки. На нём
# держится и печать по мере поступления, и запись в журнал по факту вызова
# (SPEC-w01d04.md §6, §9, §12) — обрыв на середине ячейки не должен унести уже
# оплаченные прогоны.
HeatStepHook = Callable[["HeatStep"], None]


# --------------------------------------------------------------------------
# Один вызов и одна ячейка (задача × температура)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HeatStep:
    """Один состоявшийся вызов развёртки — то, что печать и журнал видят сразу.

    duplicate/first_in_cell нужны печати «по мере поступления» (SPEC §10):
    первый прогон ячейки печатается целиком, остальные — одной строкой с
    отметкой «повтор», если normalized уже встречался в этой же ячейке.
    Признак вычисляется здесь, в момент вызова, а не в print_heat_step() —
    print_* функции не считают, они форматируют готовые значения.
    """

    problem: Problem
    temperature: float
    run: int
    runs: int
    result: CallResult
    messages: list[Message]
    text: str
    normalized: str
    format_ok: bool
    check: Check | None
    first_in_cell: bool
    duplicate: bool


@dataclass(slots=True, frozen=True)
class HeatCall:
    """Один вызов внутри готовой ячейки — то, из чего Cell считает метрики.

    check=None для open-задач: check_answer() сверяет с эталоном, а у
    open-задачи эталона нет по определению (Problem.open, SPEC §6) — звать
    сверку на пустом answer значило бы получить случайный, а не «неприменимый»
    вердикт.
    """

    index: int
    result: CallResult
    text: str
    normalized: str
    format_ok: bool
    check: Check | None


@dataclass(slots=True, frozen=True)
class Cell:
    """Метрики одной ячейки таблицы — задача × температура, по N прогонам
    (SPEC-w01d04.md §7).

    frozen: ячейка считается один раз по готовому списку вызовов и дальше
    только читается, как и Totals в advent_core/telemetry.py.
    """

    problem: Problem
    temperature: float
    calls: tuple[HeatCall, ...]

    @property
    def total(self) -> int:
        return len(self.calls)

    @property
    def accuracy(self) -> tuple[int, int] | None:
        """k/N по эталону банка. None — задача open, сверки не было (§6, §7).

        None, а не (0, N): ноль читался бы как «ни разу не угадал», хотя на
        деле сверка вообще не проводилась — эталона не существует.
        """
        if self.problem.open:
            return None
        correct = sum(1 for call in self.calls if call.check is not None and call.check.ok)
        return correct, self.total

    @property
    def format_score(self) -> tuple[int, int]:
        """k/N по соблюдению формата — считается для обеих задач (§7)."""
        ok = sum(1 for call in self.calls if call.format_ok)
        return ok, self.total

    @property
    def distinct(self) -> int:
        """Число различных normalize(ответ) из N — не сырых строк (§7).

        В живом прогоне выпали "Морская волна", " Морская волна" и
        "Морская Волна" — формально три строки, фактически один ответ. Без
        normalize() разнообразие при t=0 завышалось бы ровно на артефакты
        форматирования, то есть на то, что метрика измерять не должна.
        """
        return len({call.normalized for call in self.calls})

    @property
    def avg_length(self) -> float:
        """Средняя длина ответа в символах."""
        if not self.calls:
            return 0.0
        return sum(len(call.text) for call in self.calls) / len(self.calls)

    @property
    def totals(self) -> Totals:
        return Totals.of(call.result for call in self.calls)

    def duplicate_flags(self) -> tuple[bool, ...]:
        """Для каждого вызова — встречался ли уже такой же normalize(ответ) раньше в ЭТОЙ ячейке.

        Тот же признак, что и HeatStep.duplicate печатал раньше по мере
        поступления (_run_cell, SPEC §10) — здесь пересчитан по уже готовой
        ячейке, для сводного блока ручной оценки креативности
        (print_human_review): при temperature=0 повторов много, при
        temperature=1.2 почти нет, и число повторов само по себе данные,
        которые пользователь просил показать вместо оценки судьи.
        """
        seen: set[str] = set()
        flags: list[bool] = []
        for call in self.calls:
            flags.append(call.normalized in seen)
            seen.add(call.normalized)
        return tuple(flags)


@dataclass(slots=True)
class ProblemSweep:
    """Итог по одной задаче: ячейка на каждую температуру.

    Метода поиска ячейки по temperature здесь нет намеренно (review metrics
    #2): --temps допускает дубликаты значений, поиск по float был бы тем же
    footgun'ом, что и был у прежнего судьи (снят целиком — креативность
    оценивает человек, а не LLM, пользовательское решение) — ячейки везде
    адресуются по индексу в cells.
    """

    problem: Problem
    cells: tuple[Cell, ...]


def runs_label(n: int) -> str:
    """Согласует числительное с «прогон»: 1 прогон, 2 прогона, 5 прогонов.

    «1 прогонов» на экране (заголовок таблицы, заголовок ячейки) читается на
    записи как ошибка речи, а не как мелочь (review М2) — используется везде,
    где число прогонов идёт в текст, а не только в отладочный вывод.

    Публичная (без подчёркивания): day 05 (week_01/models_bench.py) заводит
    ту же развёртку N-прогонов-на-ячейку и переиспользует эту функцию вместо
    второй копии согласования числительных (CLAUDE.md — «переиспользуй
    существующее»).
    """
    tail, tens = n % 10, n % 100
    if tail == 1 and tens != 11:
        word = "прогон"
    elif 2 <= tail <= 4 and not 11 <= tens <= 14:
        word = "прогона"
    else:
        word = "прогонов"
    return f"{n} {word}"


def _format_ok(text: str, problem: Problem, check: Check | None) -> bool:
    """Соблюдение формата — одна метрика с двумя реализациями (SPEC §7).

    "single_line" (coffee): ровно одна непустая строка после strip(). "marker"
    (alice): присутствие ОТВЕТ: — Check.has_marker, когда сверка уже
    посчитана (не-open задачи); extract_answer() напрямую иначе, чтобы формат
    можно было проверить и без эталона. "none" (день 05, вне TEMP_PROBLEMS
    этого дня) формата не требует вовсе — соблюдён тривиально, любым текстом.
    """
    if problem.format_check == strategies.FORMAT_CHECK_NONE:
        return True
    if problem.format_check == strategies.FORMAT_CHECK_SINGLE_LINE:
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == 1
    if check is not None:
        return check.has_marker
    return strategies.extract_answer(text) is not None


def load_temp_problems(
    problem_id: str | None = None, directory: Path | None = None
) -> list[Problem]:
    """Задачи для развёртки: обе TEMP_PROBLEMS (None/"all") или одна по id.

    Тонкая обёртка над strategies.load_problem_set() — правило разбора
    None/"all" общее с Day 05 (week_01/models_bench.py), набор задач свой.
    """
    return strategies.load_problem_set(TEMP_PROBLEMS, problem_id, directory)


def heat_hygiene(config: Config) -> Config:
    """Гигиена замера (SPEC-w01d04.md §9) — обе правки видны пользователю в stderr.

    top_p обнуляется с предупреждением: Mistral рекомендует менять либо
    temperature, либо top_p, не оба разом — иначе день молча меряет их
    суперпозицию, и колонки таблицы сравнивают не то, что заявлено.

    random_seed НЕ трогается. При temperature=0 декодирование жадное и seed
    применять некуда (§2 — на пяти прогонах остаточных 2 разных ответа из 5,
    seed этого не лечит); при temperature>0 он сузил бы измеряемое
    разнообразие. Если пользователь всё же его задал — это его выбор, но
    молчать об эффекте нельзя.

    stop, пересекающийся с маркером ОТВЕТ:, снимается тем же кодом, что и на
    Day 03 (strategies._solving_config) — переписать эту логику здесь значило
    бы завести второй источник истины, который разойдётся при первой правке.

    Публичная (без подчёркивания): day 05 (week_01/models_bench.py) меряет
    модели, а не температуру, но нуждается в той же гигиене (снятие персоны
    через strategies._solving_config, top_p, предупреждение про random_seed)
    — переиспользуется отсюда вместо второй копии (CLAUDE.md).
    """
    config = strategies._solving_config(config)

    if config.params.top_p is not None:
        console.warn(
            f"top_p={config.params.top_p} обнуляется на время развёртки по температуре — "
            "Mistral рекомендует менять либо temperature, либо top_p; иначе день молча "
            "измеряет их суперпозицию"
        )
        config = replace(config, params=replace(config.params, top_p=None))

    if config.params.random_seed is not None:
        console.warn(
            f"random_seed={config.params.random_seed} задан явно и не трогается — "
            "при temperature=0 он ничего не даёт (декодирование жадное), при "
            "temperature>0 сузил бы измеряемое разнообразие"
        )

    return config


def _run_cell(
    problem: Problem,
    temperature: float,
    config: Config,
    runs: int,
    *,
    capabilities: dict[str, Any] | None,
    complete: CompleteFn,
    notify: Notify,
    on_step: HeatStepHook | None,
) -> Cell:
    """Прогоняет один и тот же запрос N раз при одной температуре — одна ячейка.

    Персона снимается ровно так же, как в стратегиях Day 03
    (build_strategy_system): пока system не задан явно, штатный
    «лаконичный ассистент» в вызов не подмешивается — он прижимает длину и
    разнообразие, то есть две метрики из четырёх (SPEC §9). Добавка про
    маркер ОТВЕТ: идёт только для format_check="marker" (alice) — на coffee
    она бы прямо противоречила условию «без пояснений и без вариантов».
    """
    cell_config = replace(config, params=replace(config.params, temperature=temperature))
    marker_needed = problem.format_check == strategies.FORMAT_CHECK_MARKER
    system = build_strategy_system(cell_config, marker=marker_needed)

    # Заголовка ячейки здесь намеренно нет (review М1): раньше notify()
    # печатал «coffee · t=0» между заголовком задачи (print_problem_header)
    # и заголовком первого прогона («── coffee · t=0 · прогон 1/1 ──») — три
    # строки об одном и том же событии подряд. Единственный заголовок ячейки
    # теперь печатает print_heat_step() на первом прогоне (first_in_cell).
    calls: list[HeatCall] = []
    seen: set[str] = set()
    for index in range(1, runs + 1):
        messages = chat_core.build_messages(problem.statement, system=system)
        result = complete(cell_config, messages, capabilities)
        text = result.text
        normalized = normalize(text)
        check = None if problem.open else check_answer(text, problem)
        format_ok = _format_ok(text, problem, check)
        duplicate = normalized in seen
        seen.add(normalized)

        calls.append(
            HeatCall(
                index=index,
                result=result,
                text=text,
                normalized=normalized,
                format_ok=format_ok,
                check=check,
            )
        )

        if on_step is not None:
            on_step(
                HeatStep(
                    problem=problem,
                    temperature=temperature,
                    run=index,
                    runs=runs,
                    result=result,
                    # sent_messages — то, что реально ушло в API после
                    # дописывания инструкции формата в chat._payload(); журнал
                    # обязан видеть именно его (тот же довод, что у
                    # strategies._Caller.ask()).
                    messages=result.sent_messages or messages,
                    text=text,
                    normalized=normalized,
                    format_ok=format_ok,
                    check=check,
                    first_in_cell=index == 1,
                    duplicate=duplicate,
                )
            )

    return Cell(problem=problem, temperature=temperature, calls=tuple(calls))


# --------------------------------------------------------------------------
# Развёртка целиком
# --------------------------------------------------------------------------


def log_heat_step(step: HeatStep, *, week: int, day: int) -> None:
    """Пишет ОДИН состоявшийся вызов развёртки в журнал — по факту вызова.

    Единица записи — вызов, а не ячейка: 429 на последнем прогоне ячейки не
    должен унести уже оплаченные (SPEC §6, §9, §12) — та же находка, что на
    Day 03 (strategies.log_step). Зовётся из on_step, то есть сразу после
    успешного complete().
    """
    log_call(
        step.result,
        step.messages,
        week=week,
        day=day,
        extra={
            "problem": step.problem.id,
            "temperature": step.temperature,
            "run": step.run,
        },
    )


def run_temperature_sweep(
    config: Config,
    problems: Sequence[Problem],
    temperatures: Sequence[float],
    runs: int,
    *,
    capabilities: dict[str, Any] | None = None,
    complete: CompleteFn = chat_core.complete,
    notify: Notify = console.note,
    on_step: HeatStepHook | None = None,
) -> list[ProblemSweep]:
    """Развёртка задача × температура × прогон — ядро дня (SPEC-w01d04.md §1, §7).

    `complete` принимается параметром и передаётся вниз явно на каждый вызов,
    а не как значение по умолчанию сигнатуры: аргумент по умолчанию
    связывается на импорте модуля, и monkeypatch (`monkeypatch.setattr(...,
    "complete", fake)`) до него не достаёт — на этом уже обжигались на Day 03
    (CLAUDE.md, «A default argument binds at import time»). CLI обязан звать
    эту функцию с complete=chat_core.complete явно.

    Бюджет вызовов — ровно `len(temperatures) × len(problems) × runs`, без
    добавки на судью: LLM-судья дня 04 убран целиком по решению пользователя
    («отдай оценку креативности человеку, просто выведи данные») — креативность
    open-задач оценивает человек по print_human_review(), а не отдельный
    вызов API. Судья Day 03 (week_01/strategies.py, команда `solve`) этим не
    затронут — он вообще не читает этот модуль.
    """
    if runs < 1:
        raise ConfigError(f"runs должен быть не меньше 1, получено {runs}")
    if not temperatures:
        raise ConfigError("список температур пуст")
    if not problems:
        raise ConfigError("список задач пуст")

    config = heat_hygiene(config)

    sweeps: list[ProblemSweep] = []
    for problem in problems:
        # Заголовка задачи здесь намеренно нет: его печатает
        # print_problem_header() один раз до начала развёртки (week_01/cli.py),
        # а первый прогон каждой ячейки печатает свой заголовок сам
        # (print_heat_step(), first_in_cell). Notify тут раньше дублировал
        # print_problem_header почти дословно ("задача coffee" против "задача
        # coffee — Название кофейни у моря") — три строки об одном и том же
        # событии подряд на экране, отсюда и review М1 у _run_cell().
        cells = [
            _run_cell(
                problem,
                temperature,
                config,
                runs,
                capabilities=capabilities,
                complete=complete,
                notify=notify,
                on_step=on_step,
            )
            for temperature in temperatures
        ]

        sweeps.append(ProblemSweep(problem=problem, cells=tuple(cells)))

    return sweeps


# --------------------------------------------------------------------------
# Печать
# --------------------------------------------------------------------------


def print_problem_header(problem: Problem) -> None:
    """Условие задачи — в stderr: это обстановка, а не ответ модели.

    Явная пометка «эталона нет» для open-задач вместо строки "эталон: " с
    пустым значением — печатать пустой problem.answer как эталон было бы
    враньём: open значит эталона нет в принципе, а не «эталон — пустая
    строка» (SPEC §6).

    problem.note НЕ печатается здесь (review method #4): это внутренняя
    пометка банка задач, а не факт, подтверждённый ЭТИМ прогоном. У coffee
    note прямо предсказывает результат таблицы («при t≥1.2 модель начинает
    выдавать по три варианта»), и печатается это ДО развёртки — то есть день,
    который должен измерять и показывать факт прогона (SPEC §10, §13),
    подсказывает зрителю ответ до измерения. Хуже: если конкретный прогон
    формат не сломает, напечатанный заранее прогноз разойдётся с таблицей на
    том же экране. День меряет — значит не имеет права печатать вывод раньше
    измерения. (strategies.print_problem() на Day 03 не трогаем — там note
    цитирует уже подтверждённый прошлый замер, а не прогноз текущего.)
    """
    console.note(f"задача {problem.id} — {problem.title}")
    console.err.print(rich_escape(problem.statement))
    if problem.open:
        console.note("эталона нет — задача open, сверка с ключом не проводится")
    else:
        console.note(f"эталон: {problem.answer}")
    if problem.placeholder:
        console.warn(f"задача {problem.id} помечена placeholder — эталон не выверен живым прогоном")


def print_heat_step(step: HeatStep) -> None:
    """Один вызов развёртки — печатается по мере поступления (SPEC §10).

    Первый прогон ячейки печатается целиком, ответ — в stdout (контракт: в
    stdout идёт только ответ модели). Остальные прогоны — одной строкой в
    stderr: нормализованный ответ плюс отметка «повтор», если он совпал с уже
    виденным в этой ячейке. Иначе N прогонов × M температур × K задач заливают
    экран сырым текстом — 18 молчащих вызовов подряд плохой кадр для записи.

    Ровно один заголовок на ячейку (review М1): «прогон N/M» печатается,
    начиная со ВТОРОГО прогона — на первом (first_in_cell) счётчик избыточен,
    это и так начало новой ячейки, а число прогонов в ней указывается один
    раз, в самом заголовке, только если их больше одного.
    """
    if step.first_in_cell:
        suffix = f" · {runs_label(step.runs)}" if step.runs > 1 else ""
        header = f"── {step.problem.id} · t={step.temperature:g}{suffix} ──"
        console.note(header)
        console.write_chunk(step.text)
        console.finish_answer()
        console.footer(step.result)
    else:
        header = f"── {step.problem.id} · t={step.temperature:g} · прогон {step.run}/{step.runs} ──"
        mark = " · повтор" if step.duplicate else ""
        console.note(f"{header} {rich_escape(step.normalized)}{mark}")


def print_cell_table(problem: Problem, cells: Sequence[Cell]) -> None:
    """Таблица на задачу — колонки ровно как в SPEC-w01d04.md §10, минус судья.

    Набор колонок зависит от Problem.open: «верно»/«время» есть только у
    задач с эталоном. Колонки «судья» больше нет вовсе (пользовательское
    решение — креативность оценивает человек, LLM-судья дня 04 убран целиком);
    для open-задач print_cell_table следом зовёт print_human_review() — там
    печатаются сами ответы, а не чужой балл за них.
    """
    runs = cells[0].total if cells else 0
    reference = "эталона нет" if problem.open else f"эталон {problem.answer}"
    title = f"{problem.title} · {reference} · {runs_label(runs)}"

    table = Table(title=title)
    table.add_column("t°", justify="right")
    if not problem.open:
        table.add_column("верно", justify="center")
    table.add_column("формат", justify="center")
    table.add_column("различных", justify="center")
    table.add_column("длина", justify="right")
    table.add_column("токены", justify="right")
    if not problem.open:
        table.add_column("время", justify="right")

    for cell in cells:
        row = [f"{cell.temperature:g}"]
        if not problem.open:
            correct, total = cell.accuracy or (0, cell.total)
            row.append(f"{correct}/{total}")
        format_ok, format_total = cell.format_score
        row.append(f"{format_ok}/{format_total}")
        row.append(f"{cell.distinct} из {cell.total}")
        row.append(f"{cell.avg_length:.0f}")
        row.append(cell.totals.tokens_label())
        if not problem.open:
            row.append(cell.totals.time_label())
        table.add_row(*row)

    console.out.print(table)

    if problem.open:
        print_human_review(problem, cells)


def print_human_review(problem: Problem, cells: Sequence[Cell]) -> None:
    """Все N ответов каждой температуры рядом — креативность оценивает человек.

    Пользовательское решение: LLM-судья дня 04 убран целиком («отдай оценку
    креативности человеку, просто выведи данные»). Место, где раньше стоял
    вызов судьи, теперь занимает эта функция — она ничего не оценивает и не
    ранжирует, только сводит ответы так, чтобы их было удобно сравнить одним
    взглядом (SPEC-w01d04.md §10 требовал ровно это для остальных прогонов
    ячейки, "10" описывал печать по мере поступления — здесь тот же принцип
    применён к финальной сводке).

    Печатается только для open-задач (Problem.open) — там нет эталона, и
    судить может только человек, глядящий на сами ответы. Для задач с
    эталоном (не-open) вердикт уже даёт сверка с ключом (колонка «верно»
    print_cell_table), второй блок был бы лишним — оба места, где решается,
    печатать ли блок (здесь и в print_cell_table), одинаково проверяют
    problem.open, а не наличие судьи, которого больше нет.

    stdout, не stderr: содержимое таблицы — ответы модели, а по контракту
    дня 01 (CLAUDE.md, «stdout/stderr contract») в stdout идёт только ответ
    модели, всё остальное — в stderr. print_heat_step() уже печатал первый
    прогон каждой ячейки в stdout по той же причине; этот блок — те же
    ответы, сведённые в одну таблицу для сравнения, а не новый вид контента.

    Одна строка таблицы — один номер прогона, один столбец — одна температура:
    так рядом оказываются все N ответов ОДНОЙ ячейки, а не только первый, как
    было видно построчно на экране раньше (SPEC §10 отмечал это неудобство
    как повод для правки). Повтор (Cell.duplicate_flags — тот же normalize(),
    что и Cell.distinct) помечается явно: при temperature=0 повторов много,
    при temperature=1.2 почти нет, и это само по себе данные, которые нужно
    просто показать, а не превращать в ещё один автоматический балл.
    """
    if not problem.open or not cells:
        return

    total = max((cell.total for cell in cells), default=0)
    if total == 0:
        return

    console.out.print(f"\n[bold]{problem.id}: все ответы — оценка креативности за человеком[/bold]")

    table = Table()
    table.add_column("№", justify="right")
    for cell in cells:
        table.add_column(f"t={cell.temperature:g}", overflow="fold")

    flags_by_cell = [cell.duplicate_flags() for cell in cells]
    for run in range(total):
        row = [str(run + 1)]
        for cell, flags in zip(cells, flags_by_cell, strict=True):
            if run < len(cell.calls):
                text = cell.calls[run].text.strip() or "(пустой ответ)"
                mark = " · повтор" if flags[run] else ""
                row.append(f"{rich_escape(text)}{mark}")
            else:
                row.append("—")
        table.add_row(*row)

    console.out.print(table)


def _format_temps(cells: Sequence[Cell]) -> str:
    return ", ".join(f"t={cell.temperature:g}" for cell in cells)


def _accuracy_conclusion(cells: Sequence[Cell]) -> str:
    """Строка «точнее всего» с явным обнаружением ничьей (review metrics #1).

    Старый код брал max() по cell.accuracy[0] без проверки на ничью — при
    равенстве max() тихо возвращает ПЕРВУЮ ячейку по порядку, то есть самую
    низкую температуру (cells идут в порядке --temps). При accuracy 0/N у
    всех ячеек это печатало «точнее всего — t=0.0 (0/N)», заявляя победителя
    там, где не выиграл никто — не гипотетика: на Day 03 прямой ответ на
    alice верен ~1 раз из 10, то есть плоская колонка нулей на 3 прогонах —
    вероятный исход демо, а не редкий край.
    """
    total = cells[0].total
    best = max(cell.accuracy[0] if cell.accuracy else 0 for cell in cells)
    winners = [cell for cell in cells if (cell.accuracy[0] if cell.accuracy else 0) == best]
    if len(winners) > 1:
        if best == 0:
            return (
                f"  точность нулевая у всех температур — ни одна не дала верного "
                f"ответа за {runs_label(total)}"
            )
        return f"  точность одинакова у всех температур: {best}/{total}"
    winner = winners[0]
    correct, cell_total = winner.accuracy or (0, winner.total)
    return f"  точнее всего — t={winner.temperature:g} ({correct}/{cell_total})"


def _diversity_conclusion(cells: Sequence[Cell]) -> list[str]:
    """Строки про разнообразие, с той же проверкой на ничью (review metrics #1,
    tests #13).

    При ПОЛНОЙ ничье (все ячейки дают одно и то же число различных ответов —
    гарантированно при runs=1, и возможно при runs>1) старый код через
    min()/max() тихо брал первую ячейку и максимумом, и минимумом сразу и
    печатал «минимум разнообразия дала t=X, а не самая низкая температура —
    обычное ожидание тут не подтвердилось», хотя расхождения нет вовсе: все
    ячейки идентичны, сравнивать нечего. При полной ничье — одна строка про
    равенство, без ложной оговорки. Оговорку про минимум печатаем, только
    когда он однозначен (единственная «проигравшая» ячейка) — иначе неясно,
    про какую именно температуру говорить.
    """
    total = cells[0].total
    best = max(cell.distinct for cell in cells)
    worst = min(cell.distinct for cell in cells)
    if best == worst:
        return [f"  разнообразие одинаково у всех температур: {best} из {total}"]

    winners = [cell for cell in cells if cell.distinct == best]
    if len(winners) > 1:
        lines = [f"  разнообразнее всего — ничья между {_format_temps(winners)}: {best} из {total}"]
    else:
        lines = [f"  разнообразнее всего — t={winners[0].temperature:g} ({best} из {total})"]

    losers = [cell for cell in cells if cell.distinct == worst]
    lowest_temperature = min(cell.temperature for cell in cells)
    if len(losers) == 1 and losers[0].temperature != lowest_temperature:
        lines.append(
            f"  минимум разнообразия дала t={losers[0].temperature:g}, а не самая "
            "низкая температура — обычное ожидание тут не подтвердилось"
        )
    return lines


def _format_conclusion(cells: Sequence[Cell]) -> str:
    """Строка про соблюдение формата — та же проверка на ничью, что у accuracy
    (review metrics #1)."""
    total = cells[0].total
    best = max(cell.format_score[0] for cell in cells)
    winners = [cell for cell in cells if cell.format_score[0] == best]
    if len(winners) > 1:
        return f"  формат соблюдён одинаково у всех температур: {best}/{total}"
    winner = winners[0]
    format_ok, format_total = winner.format_score
    return (
        f"  формат лучше всего соблюдён при t={winner.temperature:g} ({format_ok}/{format_total})"
    )


def print_conclusions(sweeps: Sequence[ProblemSweep]) -> None:
    """Блок «Вывод»: механические факты ЭТОГО прогона, без обобщений (SPEC §10).

    Для каждой задачи и метрики называется температура, давшая максимум в
    посчитанных цифрах — не «температура X лучше вообще», а «в этом прогоне у
    X максимум по Y». Ничья (максимум делят несколько температур) называется
    ничьей словами, а не победителем по порядку --temps — тот же принцип для
    всех трёх метрик кодом (точность/разнообразие/формат), см. _accuracy_
    conclusion/_diversity_conclusion/_format_conclusion (review metrics #1,
    tests #13). Расхождение с обычным ожиданием (минимум разнообразия не у
    самой низкой температуры) называется прямо, а не сглаживается — риск §14.

    Про креативность здесь намеренно ни слова: LLM-судья дня 04 убран целиком
    (пользовательское решение), а «который вариант интереснее» — не
    механический факт, который можно вывести из посчитанных чисел, это и есть
    то, что отдано человеку (print_human_review).
    """
    console.out.print("\n[bold]Вывод[/bold]")
    for sweep in sweeps:
        problem, cells = sweep.problem, sweep.cells
        if not cells:
            continue
        lines = [f"{problem.id}:"]

        if not problem.open:
            lines.append(_accuracy_conclusion(cells))

        lines.extend(_diversity_conclusion(cells))
        lines.append(_format_conclusion(cells))

        console.out.print("\n".join(lines))
