"""Day 04: развёртка задача × температура × прогон — метрики, гигиена, ручная оценка.

Клиент не мокается вовсе — `run_temperature_sweep()` принимает `complete`
параметром (та же грабля с default-аргументом, что и в week_01/strategies.py:
подмена `chat_core.complete` через monkeypatch до значения по умолчанию не
достаёт, поэтому тесты подают свою `_Complete` явно). Сеть не трогается,
кредиты не тратятся.

Каждый блок ниже привязан к конкретному пункту SPEC-w01d04.md §12 — в
docstring теста указано, какую регрессию он ловит.

LLM-судья дня 04 убран целиком (пользовательское решение: креативность
open-задач оценивает человек, а не API) — с ним ушли и тесты судьи. Судья
Day 03 (week_01/strategies.py, команда `solve`) не затронут и живёт своими
тестами в tests/test_strategies.py / tests/test_cli_solve.py.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass

import pytest
from rich.console import Console

from advent_core import console
from advent_core.config import DEFAULT_SYSTEM_PROMPT, Config
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, Totals, Usage
from week_01 import strategies, temperature

# --------------------------------------------------------------------------
# Инструменты
# --------------------------------------------------------------------------


def _config(**param_kwargs) -> Config:
    """Конфиг без system prompt по умолчанию — персона добавляется явно там, где нужна."""
    return Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        system_prompt_path=None,
        params=GenerationParams.build(**param_kwargs),
    )


# Реальные задачи банка, а не переписанные в тесте строки (SPEC §12, п.14):
# статья условия coffee идёт из JSON-файла, стык промпт↔код проверяется на
# фактическом тексте, а не на копии, которая может незаметно разойтись с ним.
ALICE = strategies.load_problem("alice")
COFFEE = strategies.load_problem("coffee")
DIGITS5 = strategies.load_problem("digits5")


@dataclass(slots=True)
class _Call:
    config: Config
    messages: list[dict[str, str]]
    capabilities: dict | None

    @property
    def system(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "system"), "")

    @property
    def user(self) -> str:
        return self.messages[-1]["content"]


class _Complete:
    """Замена chat.complete: отдаёт заготовленные тексты по очереди, копит вызовы.

    `watch`, если задан, — список журнала: перед КАЖДЫМ вызовом фиксируется
    его текущая длина (`before_call`), что и позволяет проверить «запись по
    факту вызова» (SPEC §12, п.11) — тот же приём, что в tests/test_cli_solve.py.
    """

    def __init__(self, *texts: str, default: str = "ОТВЕТ: 3", latency_ms: int = 100):
        self.texts = list(texts)
        self.default = default
        self.latency_ms = latency_ms
        self.calls: list[_Call] = []
        self.watch: list | None = None
        self.before_call: list[int] = []

    def __call__(self, config, messages, capabilities=None) -> CallResult:
        if self.watch is not None:
            self.before_call.append(len(self.watch))
        self.calls.append(_Call(config, messages, capabilities))
        text = self.texts.pop(0) if self.texts else self.default
        return CallResult(
            text=text,
            model_requested=config.model,
            usage=Usage(10, 5, 15),
            latency_ms=self.latency_ms,
            stream=False,
            finish_reason="stop",
            sent_messages=messages,
        )


def _silent(_message: str) -> None:
    pass


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


@pytest.fixture
def stdout_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


def _sweep(problems, complete, *, config=None, **kwargs):
    return temperature.run_temperature_sweep(
        config or _config(),
        problems,
        kwargs.pop("temperatures", [0.0, 0.7]),
        kwargs.pop("runs", 1),
        complete=complete,
        notify=_silent,
        **kwargs,
    )


# --------------------------------------------------------------------------
# п.1 — temperature=0.0 доезжает до payload, temperature=None не попадает
# --------------------------------------------------------------------------


def test_zero_temperature_reaches_the_payload_not_dropped_as_unset():
    """Регрессия на `if not value`: ноль превратился бы в «не задана» — серверные 0.3."""
    payload, _ = GenerationParams.build(temperature=0.0).as_payload({"completion_chat": True})
    assert payload["temperature"] == 0.0


def test_unset_temperature_is_absent_from_the_payload():
    payload, _ = GenerationParams.build().as_payload({"completion_chat": True})
    assert "temperature" not in payload


def test_the_sweep_actually_sends_zero_temperature_on_the_wire():
    """Не только хелпер — сам прогон обязан передать 0.0 в config, ушедший в complete()."""
    complete = _Complete()
    _sweep([ALICE], complete, temperatures=[0.0], runs=1)

    assert complete.calls[0].config.params.temperature == 0.0


# --------------------------------------------------------------------------
# п.2 — разбор `temps`
# --------------------------------------------------------------------------


def test_temps_parses_a_comma_separated_string():
    assert GenerationParams.build(temps="0,0.7,1.2").temps == [0.0, 0.7, 1.2]


def test_temps_rejects_an_element_above_the_api_ceiling():
    from advent_core.params import ParamError

    with pytest.raises(ParamError):
        GenerationParams.build(temps="0,1.51")


def test_temps_rejects_an_element_below_the_floor():
    from advent_core.params import ParamError

    with pytest.raises(ParamError):
        GenerationParams.build(temps="-0.1,0.5")


def test_temps_rejects_a_non_numeric_element():
    from advent_core.params import ParamError

    with pytest.raises(ParamError):
        GenerationParams.build(temps="0,тепло")


def test_temps_keeps_duplicates_in_input_order():
    assert GenerationParams.build(temps="0.7,0,0.7").temps == [0.7, 0.0, 0.7]


def test_empty_temps_string_gives_none():
    assert GenerationParams.build(temps="").temps is None


def test_temperature_ceiling_is_1_5_not_2():
    """SPEC §2: живой замер дал 422 уже на 1.51 — потолок API 1.5, не 2.0."""
    from advent_core.params import ParamError

    assert GenerationParams.build(temperature=1.5).temperature == 1.5
    with pytest.raises(ParamError):
        GenerationParams.build(temperature=2)


# --------------------------------------------------------------------------
# п.4 — apply_defaults не делит один list-объект temps между инстансами
# --------------------------------------------------------------------------


def test_apply_defaults_does_not_share_the_temps_list_between_instances():
    """Мутация temps одного инстанса не должна быть видна другому.

    Spec.defaults[TEMP_COMMAND] для temps — один list-объект на весь модуль
    (SPECS строится один раз при импорте). apply_defaults() обязан копировать
    его, а не присваивать as is.
    """
    first = GenerationParams()
    second = GenerationParams()
    first.apply_defaults("temp")
    second.apply_defaults("temp")

    assert first.temps == [0.0, 0.7, 1.2]
    first.temps.append(1.5)

    assert second.temps == [0.0, 0.7, 1.2], "второй инстанс не должен увидеть чужую мутацию"


# --------------------------------------------------------------------------
# п.5 — разнообразие считается по normalize(), а не по сырой строке
# --------------------------------------------------------------------------


def _cell_of(problem, texts: list[str]) -> temperature.Cell:
    calls = tuple(
        temperature.HeatCall(
            index=index,
            result=CallResult(text=text, usage=Usage(10, 5, 15), latency_ms=100),
            text=text,
            normalized=strategies.normalize(text),
            format_ok=True,
            check=None if problem.open else strategies.check_answer(text, problem),
        )
        for index, text in enumerate(texts, start=1)
    )
    return temperature.Cell(problem=problem, temperature=0.7, calls=calls)


def test_distinct_counts_normalized_answers_not_raw_strings():
    """Живой прогон: три строки формально разные, ответ фактически один (SPEC §7)."""
    cell = _cell_of(COFFEE, ["Морская волна", " Морская волна", "Морская Волна"])
    assert cell.distinct == 1


def test_distinct_still_tells_genuinely_different_answers_apart():
    cell = _cell_of(COFFEE, ["Морская волна", "Ракушка", "Ракушка"])
    assert cell.distinct == 2


# --------------------------------------------------------------------------
# п.6 — format_check: marker на alice, single_line на coffee
# --------------------------------------------------------------------------


def test_marker_format_check_follows_the_computed_verdict_on_alice():
    check_ok = strategies.check_answer("ОТВЕТ: 3", ALICE)
    check_no_marker = strategies.check_answer("три сестры, но без маркера", ALICE)

    assert temperature._format_ok("ОТВЕТ: 3", ALICE, check_ok) is True
    assert temperature._format_ok("три сестры, но без маркера", ALICE, check_no_marker) is False


def test_single_line_format_check_flags_two_content_lines_as_a_violation():
    """coffee просит одну строку без вариантов — второй содержательной строки быть не должно."""
    text = "Морской бриз\nа ещё вариант: Волна"
    assert temperature._format_ok(text, COFFEE, None) is False


def test_single_line_format_check_ignores_a_trailing_blank_line():
    """Хвостовой перевод строки — не нарушение, это не вторая строка контента."""
    text = "Морской бриз\n"
    assert temperature._format_ok(text, COFFEE, None) is True


def test_single_line_format_check_accepts_exactly_one_line():
    assert temperature._format_ok("Морской бриз", COFFEE, None) is True


# --------------------------------------------------------------------------
# п.7 — open-задача не сверяется с эталоном; не-open к судье не идёт
# --------------------------------------------------------------------------


def test_open_problem_cell_has_no_accuracy_column():
    """None, а не (0, N): сверки не было вовсе, а не «ни разу не угадал»."""
    cell = _cell_of(COFFEE, ["Морской бриз"])
    assert cell.accuracy is None


def test_non_open_problem_cell_reports_accuracy():
    cell = _cell_of(ALICE, ["ОТВЕТ: 3", "ОТВЕТ: 2"])
    assert cell.accuracy == (1, 2)


def test_open_problem_cell_still_has_no_accuracy_and_no_extra_calls():
    """coffee (open) не сверяется с эталоном и не порождает лишних вызовов —
    LLM-судья дня 04 убран целиком (пользовательское решение), поэтому один
    прогон open-задачи — ровно один вызов, без добавки на судью."""
    complete = _Complete(default="Морской бриз")
    sweeps = _sweep([COFFEE], complete, temperatures=[0.0], runs=1)

    assert sweeps[0].cells[0].accuracy is None
    assert len(complete.calls) == 1


# --------------------------------------------------------------------------
# п.8 — персона снимается по умолчанию, сохраняется при явном --system
# --------------------------------------------------------------------------


def test_persona_is_stripped_on_a_real_run_with_the_default_system_path():
    """Проверяется на прогоне, не только на хелпере (урок Day 03 — CLAUDE.md).

    Иначе вызов build_strategy_system() можно удалить из temperature.py, не
    покрасив ни одного теста.
    """
    config = Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        system_prompt_path=DEFAULT_SYSTEM_PROMPT,
        params=GenerationParams.build(),
    )
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep([ALICE], complete, config=config, temperatures=[0.0], runs=1)

    persona = DEFAULT_SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    assert persona not in complete.calls[0].system


def test_explicit_system_prompt_survives_into_the_real_run(tmp_path):
    path = tmp_path / "system.md"
    path.write_text("ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ", encoding="utf-8")
    config = Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        system_prompt_path=path,
        params=GenerationParams.build(),
    )
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep([ALICE], complete, config=config, temperatures=[0.0], runs=1)

    assert "ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ" in complete.calls[0].system


# --------------------------------------------------------------------------
# п.9 — top_p обнуляется и предупреждает; random_seed предупреждает, но остаётся
# --------------------------------------------------------------------------


def test_top_p_is_zeroed_for_the_sweep_and_warns(capsys):
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep([ALICE], complete, config=_config(top_p=0.9), temperatures=[0.0], runs=1)

    assert complete.calls[0].config.params.top_p is None
    assert "top_p" in _flat(capsys.readouterr().err)


def test_random_seed_is_kept_but_warns(capsys):
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep(
        [ALICE],
        complete,
        config=_config(random_seed=42),
        temperatures=[0.0],
        runs=1,
    )

    assert complete.calls[0].config.params.random_seed == 42, "выбор пользователя не трогаем"
    assert "random_seed" in _flat(capsys.readouterr().err)


def test_hygiene_is_silent_when_nothing_needs_a_warning(capsys):
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep([ALICE], complete, temperatures=[0.0], runs=1)

    stderr = _flat(capsys.readouterr().err)
    assert "top_p" not in stderr
    assert "random_seed" not in stderr


# --------------------------------------------------------------------------
# п.10 — бюджет вызовов: ровно len(temps) × задач × runs, без судьи
# --------------------------------------------------------------------------


def test_call_budget_is_exactly_temps_times_problems_times_runs():
    """LLM-судья дня 04 убран целиком (пользовательское решение) — бюджет
    больше не получает добавку "+1 на судью": ровно len(temps) × задач × runs,
    даже когда среди задач есть open (coffee)."""
    complete = _Complete(default="ОТВЕТ: 3")
    _sweep([ALICE, COFFEE], complete, temperatures=[0.0, 0.7, 1.2], runs=3)

    # 2 задачи × 3 температуры × 3 прогона = 18, без добавки — единственная
    # часть бюджета вообще.
    assert len(complete.calls) == 18


# --------------------------------------------------------------------------
# п.11 — журнал пишется по факту каждого вызова
# --------------------------------------------------------------------------


def test_journal_has_exactly_n_minus_1_lines_before_the_nth_call():
    """Перед N-м вызовом в журнале лежит ровно N−1 строк, а не пачка в конце."""
    journal: list[dict] = []
    complete = _Complete(default="ОТВЕТ: 3")
    complete.watch = journal

    def on_step(step: temperature.HeatStep) -> None:
        journal.append({"temperature": step.temperature, "run": step.run})

    _sweep(
        [ALICE],
        complete,
        temperatures=[0.0, 0.7],
        runs=2,
        on_step=on_step,
    )

    assert complete.before_call == [0, 1, 2, 3]
    assert len(journal) == 4


def test_a_mid_sweep_failure_keeps_the_already_paid_calls_logged():
    """429 на третьем вызове ячейки не должен унести два уже оплаченных."""
    journal: list[dict] = []

    def on_step(step: temperature.HeatStep) -> None:
        journal.append({"temperature": step.temperature, "run": step.run})

    calls_made = []

    def failing(config, messages, capabilities=None):
        calls_made.append(1)
        if len(calls_made) == 3:
            raise AdventError("rate limit")
        return CallResult(
            text="ОТВЕТ: 3", usage=Usage(10, 5, 15), latency_ms=100, sent_messages=messages
        )

    with pytest.raises(AdventError):
        _sweep(
            [ALICE],
            failing,
            temperatures=[0.0, 0.7, 1.2],
            runs=1,
            on_step=on_step,
        )

    assert len(journal) == 2, "два уже оплаченных вызова обязаны остаться в журнале"


# --------------------------------------------------------------------------
# п.12 — агрегация k/N и суммирование Totals по ячейке
# --------------------------------------------------------------------------


def test_cell_accuracy_and_format_aggregate_k_of_n():
    cell = _cell_of(ALICE, ["ОТВЕТ: 3", "ОТВЕТ: 2", "ОТВЕТ: 3"])
    assert cell.accuracy == (2, 3)
    assert cell.format_score == (3, 3), "все три с маркером и с format_ok=True в фикстуре"


def test_cell_totals_sum_tokens_and_latency_across_calls():
    calls = tuple(
        temperature.HeatCall(
            index=i,
            result=CallResult(usage=Usage(10, 5, 15), latency_ms=200),
            text="ОТВЕТ: 3",
            normalized="3",
            format_ok=True,
            check=strategies.check_answer("ОТВЕТ: 3", ALICE),
        )
        for i in range(1, 4)
    )
    cell = temperature.Cell(problem=ALICE, temperature=0.7, calls=calls)
    totals: Totals = cell.totals

    assert totals.calls == 3
    assert totals.prompt_tokens == 30
    assert totals.completion_tokens == 15
    assert totals.total_tokens == 45
    assert totals.latency_ms == 600


def test_cell_average_length_is_computed_over_the_calls():
    cell = _cell_of(COFFEE, ["абв", "абвгд"])
    assert cell.avg_length == 4.0


# --------------------------------------------------------------------------
# Cell.duplicate_flags() — тот же normalize(), что и Cell.distinct, но
# по каждому вызову отдельно: печатается ли «повтор» у print_human_review.
# --------------------------------------------------------------------------


def test_duplicate_flags_marks_repeats_after_the_first_occurrence():
    cell = _cell_of(COFFEE, ["Морской бриз", "Ракушка", "морской бриз", "Ракушка"])
    # normalize() приводит к нижнему регистру и схлопывает пробелы — третий
    # ответ повторяет первый (с точностью до регистра), четвёртый — второй.
    assert cell.duplicate_flags() == (False, False, True, True)


def test_duplicate_flags_all_false_when_every_answer_is_distinct():
    cell = _cell_of(COFFEE, ["Морской бриз", "Ракушка", "Волна"])
    assert cell.duplicate_flags() == (False, False, False)


# --------------------------------------------------------------------------
# п.14 — стык промпта и кода: текст coffee берётся из банка, не переписывается
# --------------------------------------------------------------------------


def test_coffee_statement_sent_on_the_wire_matches_the_bank_file_verbatim():
    complete = _Complete(default="Морской бриз")
    _sweep([COFFEE], complete, temperatures=[0.0], runs=1)

    assert complete.calls[0].user == COFFEE.statement
    # И вырожденная проверка на «не переписано вручную»: файл банка меняется
    # независимо от теста, тест должен читать его же, а не литерал.
    from week_01.strategies import PROBLEMS_DIR

    raw = json.loads((PROBLEMS_DIR / "coffee.json").read_text(encoding="utf-8"))
    assert complete.calls[0].user == raw["statement"]


def test_coffee_problem_is_marked_open_with_single_line_format_check():
    assert COFFEE.open is True
    assert COFFEE.format_check == strategies.FORMAT_CHECK_SINGLE_LINE


def test_alice_problem_is_not_open_with_marker_format_check():
    assert ALICE.open is False
    assert ALICE.format_check == strategies.FORMAT_CHECK_MARKER


# --------------------------------------------------------------------------
# load_temp_problems: обе задачи по умолчанию/"all", одна по id
# --------------------------------------------------------------------------


def test_load_temp_problems_none_returns_all_three_problems_of_the_day():
    # Литерал ["digits5", "alice", "coffee"], а не list(temperature.TEMP_PROBLEMS):
    # сверка с той же константой, которую читает сам load_temp_problems(),
    # тавтологична и не покраснела бы при любой правке TEMP_PROBLEMS — тот же
    # класс дефекта, что и DAY в tests/test_cli_temp.py (находка tests #12/#1
    # задания).
    problems = temperature.load_temp_problems(None)
    assert [p.id for p in problems] == ["digits5", "alice", "coffee"]


def test_load_temp_problems_all_returns_all_three_problems_of_the_day():
    problems = temperature.load_temp_problems("all")
    assert [p.id for p in problems] == ["digits5", "alice", "coffee"]


def test_load_temp_problems_by_id_returns_one():
    problems = temperature.load_temp_problems("alice")
    assert [p.id for p in problems] == ["alice"]


def test_digits5_and_alice_are_both_part_of_the_day_in_that_order():
    """Пара digits5/alice — сам вывод дня (temperature.py, комментарий у
    TEMP_PROBLEMS): порознь каждая задача показывает неверное обобщение про
    температуру и точность, только пара вместе — правильное. Потеря любой из
    двух ломает то, что день доказывает, поэтому проверяется явно, отдельно
    от общего порядка задач."""
    ids = list(temperature.TEMP_PROBLEMS)
    assert "digits5" in ids
    assert "alice" in ids
    assert ids.index("digits5") < ids.index("alice"), (
        "digits5 (точность падает с температурой — обычное ожидание) обязана идти "
        "перед alice (точность растёт — обратный результат), см. комментарий у "
        "TEMP_PROBLEMS"
    )


def test_temp_problems_order_is_digits5_alice_coffee():
    assert temperature.TEMP_PROBLEMS == ("digits5", "alice", "coffee")


# --------------------------------------------------------------------------
# Печать: базовая проверка, что таблица не роняется на open/non-open задачах
# --------------------------------------------------------------------------


def test_print_cell_table_smoke_for_both_kinds_of_problem(stdout_capture):
    complete = _Complete(default="ОТВЕТ: 3")
    sweeps = _sweep([ALICE, COFFEE], complete, temperatures=[0.0], runs=1)

    for sweep in sweeps:
        temperature.print_cell_table(sweep.problem, sweep.cells)

    printed = stdout_capture.getvalue()
    assert "различных" in printed
    assert "верно" in printed  # колонка alice
    # Колонки «судья» больше нет вовсе (LLM-судья дня 04 убран целиком) — ни
    # у одной из задач, включая open (coffee).
    assert "судья" not in printed


# --------------------------------------------------------------------------
# print_human_review(): судья убран целиком (пользовательское решение) —
# вместо ранжирования день печатает сами ответы, чтобы человек сравнил их
# одним взглядом. Печатается только для open-задач, показывает ВСЕ N ответов
# каждой ячейки (не только первый), помечает повторы.
# --------------------------------------------------------------------------


def _sweep_cells(
    problem: strategies.Problem, per_temperature: list[list[str]]
) -> list[temperature.Cell]:
    """cells для (0.0, 0.7, 1.2) с заданными текстами ответов на температуру.

    Использует _conclusions_cell (определена ниже, в блоке print_conclusions):
    те же готовые ячейки, обёрнутые по одной на температуру — порядок
    определений в модуле не важен, обе функции читаются только во время
    выполнения тестов, когда файл уже загружен целиком.
    """
    return [
        _conclusions_cell(problem, temp, texts)
        for temp, texts in zip((0.0, 0.7, 1.2), per_temperature, strict=True)
    ]


def test_print_human_review_is_printed_for_an_open_problem(stdout_capture):
    cells = _sweep_cells(
        COFFEE,
        [["Морской бриз", "Ракушка"], ["Волна", "Штиль"], ["Прибой", "Маяк"]],
    )
    temperature.print_human_review(COFFEE, cells)

    printed = stdout_capture.getvalue()
    assert "coffee" in printed
    assert "оценка креативности за человеком" in printed


def test_print_human_review_is_not_printed_for_a_non_open_problem(stdout_capture):
    """Задача с эталоном (alice) не нуждается в ручной оценке — вердикт даёт
    сверка с ключом (колонка «верно»), второй блок был бы лишним."""
    cells = _sweep_cells(
        ALICE,
        [["ОТВЕТ: 3", "ОТВЕТ: 2"], ["ОТВЕТ: 3", "ОТВЕТ: 3"], ["ОТВЕТ: 2", "ОТВЕТ: 3"]],
    )
    temperature.print_human_review(ALICE, cells)

    assert stdout_capture.getvalue() == ""


def test_print_cell_table_prints_human_review_after_the_table_for_open_problems(stdout_capture):
    """Стык: print_cell_table сама решает звать print_human_review — та же
    точка входа, где раньше стоял вызов судьи."""
    complete = _Complete(default="Морской бриз")
    sweeps = _sweep([COFFEE], complete, temperatures=[0.0, 0.7, 1.2], runs=2)

    temperature.print_cell_table(sweeps[0].problem, sweeps[0].cells)

    assert "оценка креативности за человеком" in stdout_capture.getvalue()


def test_print_cell_table_does_not_print_human_review_for_a_non_open_problem(stdout_capture):
    complete = _Complete(default="ОТВЕТ: 3")
    sweeps = _sweep([ALICE], complete, temperatures=[0.0, 0.7, 1.2], runs=2)

    temperature.print_cell_table(sweeps[0].problem, sweeps[0].cells)

    assert "оценка креативности" not in stdout_capture.getvalue()


def test_print_human_review_shows_all_n_answers_not_only_the_first(stdout_capture):
    """SPEC-w01d04.md §10 отмечал ровно эту проблему у прежней печати: раньше
    целиком печатался только первый прогон ячейки, остальные — свёрнуты в
    одну строку. print_human_review обязана показать ВСЕ прогоны рядом."""
    cells = _sweep_cells(
        COFFEE,
        [
            ["Морской бриз", "Ракушка", "Волна"],
            ["Штиль", "Прибой", "Маяк"],
            ["Пена", "Соль", "Горизонт"],
        ],
    )
    temperature.print_human_review(COFFEE, cells)

    printed = _flat(stdout_capture.getvalue())
    all_texts = (
        "Морской бриз",
        "Ракушка",
        "Волна",
        "Штиль",
        "Прибой",
        "Маяк",
        "Пена",
        "Соль",
        "Горизонт",
    )
    for text in all_texts:
        assert text in printed, f"{text!r} обязан быть виден — не только первый прогон ячейки"


def test_print_human_review_marks_a_repeated_answer(stdout_capture):
    """При t=0 повторов много — это данные, которые пользователь просил
    показать явно, а не превращать в ещё один автоматический балл."""
    cells = _sweep_cells(
        COFFEE,
        [["Морской бриз", "Морской бриз"], ["Волна", "Штиль"], ["Прибой", "Маяк"]],
    )
    temperature.print_human_review(COFFEE, cells)

    printed = _flat(stdout_capture.getvalue())
    assert "повтор" in printed


def test_print_human_review_does_not_claim_a_repeat_when_all_answers_differ(stdout_capture):
    cells = _sweep_cells(
        COFFEE,
        [["Морской бриз", "Ракушка"], ["Волна", "Штиль"], ["Прибой", "Маяк"]],
    )
    temperature.print_human_review(COFFEE, cells)

    assert "повтор" not in _flat(stdout_capture.getvalue())


# --------------------------------------------------------------------------
# print_conclusions() — до этого дня не покрыт ни одним тестом (находки
# metrics #1, tests #13). max()/min() без обнаружения ничьей молча возвращают
# ПЕРВУЮ ячейку по порядку --temps и печатают её как единственного победителя,
# даже когда все ячейки равны, — риск §14 SPEC (alice: прямой ответ верен 1
# раз из 10, на 3 прогонах 0/3, 0/3, 0/3 вполне реальны).
# --------------------------------------------------------------------------


def _conclusions_cell(
    problem: strategies.Problem,
    temperature_value: float,
    texts: list[str],
    *,
    format_ok: list[bool] | None = None,
) -> temperature.Cell:
    """Тот же принцип, что _cell_of() выше, но с управляемой температурой и
    format_ok: print_conclusions сравнивает ячейки МЕЖДУ СОБОЙ, поэтому нужны
    несколько ячеек с намеренно равными или намеренно разными метриками, а не
    одна фиксированная t=0.7.
    """
    calls = tuple(
        temperature.HeatCall(
            index=index,
            result=CallResult(text=text, usage=Usage(10, 5, 15), latency_ms=100),
            text=text,
            normalized=strategies.normalize(text),
            format_ok=True if format_ok is None else format_ok[index - 1],
            check=None if problem.open else strategies.check_answer(text, problem),
        )
        for index, text in enumerate(texts, start=1)
    )
    return temperature.Cell(problem=problem, temperature=temperature_value, calls=calls)


def _conclusions_of(problem: strategies.Problem, cells: list[temperature.Cell]) -> None:
    temperature.print_conclusions([temperature.ProblemSweep(problem=problem, cells=tuple(cells))])


def test_print_conclusions_does_not_claim_a_winner_on_a_full_accuracy_tie(stdout_capture):
    """0/3, 0/3, 0/3 у alice: ни одна температура не дала верный ответ, но
    max(key=accuracy) без обнаружения ничьей молча берёт первую ячейку (t=0) и
    печатает её как «точнее всего» — ровно риск §14 SPEC."""
    cells = [
        _conclusions_cell(ALICE, temp, ["ОТВЕТ: 2", "ОТВЕТ: 2", "ОТВЕТ: 2"])
        for temp in (0.0, 0.7, 1.2)
    ]
    _conclusions_of(ALICE, cells)

    assert "точнее всего — t=0 (0/3)" not in _flat(stdout_capture.getvalue())


def test_print_conclusions_does_not_claim_a_diversity_mismatch_on_a_full_tie(stdout_capture):
    """Все три ячейки дают distinct=1 (одна не-open, здесь coffee/open — это не
    важно для diversity), а --temps поданы НЕ по возрастанию (1.2, 0.7, 0.0).
    min(key=distinct) без обнаружения ничьей берёт первую ячейку СПИСКА
    (t=1.2), сравнивает её с истинным минимумом температур (0.0) и печатает
    ложное «ожидание не подтвердилось», хотя реального расхождения нет —
    находка metrics/tests #13."""
    cells = [
        _conclusions_cell(COFFEE, temp, [text])
        for temp, text in ((1.2, "Морской бриз"), (0.7, "Волна"), (0.0, "Ракушка"))
    ]
    _conclusions_of(COFFEE, cells)

    assert "не подтвердилось" not in _flat(stdout_capture.getvalue())


def test_print_conclusions_does_not_claim_a_winner_on_a_full_format_tie(stdout_capture):
    """Тот же класс дефекта, что и у accuracy (находка metrics #1): формат
    соблюдён одинаково (2 из 3) на всех трёх температурах, первая по порядку
    не должна молча объявляться лучшей."""
    cells = [
        _conclusions_cell(
            ALICE, temp, ["ОТВЕТ: 3", "ОТВЕТ: 3", "ОТВЕТ: 3"], format_ok=[True, False, True]
        )
        for temp in (0.0, 0.7, 1.2)
    ]
    _conclusions_of(ALICE, cells)

    assert "формат лучше всего соблюдён при t=0 (2/3)" not in _flat(stdout_capture.getvalue())


def test_print_conclusions_survives_runs_equal_one(stdout_capture):
    """SPEC §12 требует отдельно покрыть runs=1: одна точка на ячейку — крайний
    случай для агрегатов k/N (0/1 или 1/1). Победитель здесь настоящий (t=0.7
    — единственная верная), а не ничья, поэтому обязан называться и при N=1."""
    cells = [
        _conclusions_cell(ALICE, 0.0, ["ОТВЕТ: 2"]),
        _conclusions_cell(ALICE, 0.7, ["ОТВЕТ: 3"]),
        _conclusions_cell(ALICE, 1.2, ["ОТВЕТ: 2"]),
    ]
    _conclusions_of(ALICE, cells)

    assert "точнее всего — t=0.7 (1/1)" in _flat(stdout_capture.getvalue())


def test_print_conclusions_names_the_winner_when_there_really_is_one(stdout_capture):
    """Контрольный случай: настоящий победитель по accuracy (3/3 против 0/3 у
    двух других температур) обязан называться по имени — починка ничьей не
    имеет права затереть заодно и реальные различия."""
    cells = [
        _conclusions_cell(ALICE, 0.0, ["ОТВЕТ: 2", "ОТВЕТ: 2", "ОТВЕТ: 2"]),
        _conclusions_cell(ALICE, 0.7, ["ОТВЕТ: 3", "ОТВЕТ: 3", "ОТВЕТ: 3"]),
        _conclusions_cell(ALICE, 1.2, ["ОТВЕТ: 2", "ОТВЕТ: 2", "ОТВЕТ: 2"]),
    ]
    _conclusions_of(ALICE, cells)

    assert "точнее всего — t=0.7 (3/3)" in _flat(stdout_capture.getvalue())


# --------------------------------------------------------------------------
# Мутационная проверка (задание, п.3): normalize() и снятие небезопасного stop
# покрыты только на уровне готовой Cell/хелпера, а не на полном пути через
# run_temperature_sweep() — тот же урок Day 03 (CLAUDE.md, «A default argument
# binds at import time» и «проверяется на прогоне, не только на хелпере»).
# Проверено фактической мутацией temperature.py (сломать строку → прогнать
# существующий набор → вернуть обратно): без этих двух тестов ни один из 67
# существующих тестов не краснеет ни при удалении normalize() из _run_cell,
# ни при удалении strategies._solving_config() из _heat_hygiene().
# --------------------------------------------------------------------------


def test_the_sweep_normalizes_answers_through_the_full_pipeline_not_only_the_cell_property():
    """test_distinct_counts_normalized_answers_not_raw_strings (выше) проверяет
    только свойство Cell.distinct на вручную собранных HeatCall — она не
    заметит, если _run_cell() перестанет звать normalize() и станет писать в
    HeatCall.normalized сырой текст. Гоняем тот же случай через настоящий
    run_temperature_sweep()."""
    complete = _Complete("Морская волна", " Морская волна", "Морская Волна")
    sweeps = _sweep([COFFEE], complete, temperatures=[0.7], runs=3)

    assert sweeps[0].cells[0].distinct == 1


def test_the_sweep_strips_an_unsafe_stop_that_would_eat_the_marker():
    """_heat_hygiene() зовёт strategies._solving_config() ровно как на Day 03
    (tests/test_strategies.py, «НАХОДКА 14») — но ни один тест temp не проверял
    это на прогоне. stop, пересекающийся с маркером ОТВЕТ:, обязан не доехать
    до API, иначе API вырежет маркер из ответа и сверка с эталоном молча
    скажет «нет маркера» (CLAUDE.md: «A stop sequence never reaches the
    output»)."""
    complete = _Complete(default="ОТВЕТ: 3")
    config = _config(stop="ОТВЕТ:")
    sweeps = _sweep([ALICE], complete, config=config, temperatures=[0.0], runs=1)

    assert complete.calls[0].config.params.stop is None
    assert sweeps[0].cells[0].calls[0].check.ok, "маркер обязан дойти до ответа и разобраться"
