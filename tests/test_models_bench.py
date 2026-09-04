"""Day 05: развёртка модель × задача × прогон — цена, метрики, лестница, вывод.

Модульный уровень (week_01/models_bench.py) — цена и агрегаты собираются на
руками построенных Cell/BenchCall, тот же приём, что у _cell_of/
_conclusions_cell в tests/test_temperature.py: считающий код и печать
проверяются раздельно от сетевого пути (тот покрыт tests/test_cli_bench.py).

Каждый блок привязан к конкретному пункту SPEC-w01d05.md §16.
"""

from __future__ import annotations

import io
import re
from datetime import date

import pytest
from rich.console import Console

from advent_core import console
from advent_core.config import DEFAULT_MODEL
from advent_core.params import BENCH_COMMAND, BY_NAME
from advent_core.telemetry import CallResult, Usage
from week_01 import models_bench, strategies

CHILDREN = strategies.load_problem("children")
SKY = strategies.load_problem("sky")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


@pytest.fixture
def stdout_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


def _call(
    index: int,
    *,
    latency_ms: float = 100,
    prompt: int | None = 100,
    completion: int | None = 10,
    cached: int | None = None,
    text: str = "ОТВЕТ: 7",
) -> models_bench.BenchCall:
    result = CallResult(
        text=text,
        usage=Usage(prompt, completion, cached_tokens=cached),
        latency_ms=latency_ms,
    )
    return models_bench.BenchCall(
        index=index,
        result=result,
        text=text,
        normalized=strategies.normalize(text),
        check=None if CHILDREN.open else strategies.check_answer(text, CHILDREN),
    )


def _cell(model: str, calls: list[models_bench.BenchCall], problem=CHILDREN) -> models_bench.Cell:
    return models_bench.Cell(model=model, problem=problem, calls=tuple(calls))


def _prices(**models: models_bench.ModelPrice) -> models_bench.PriceTable:
    return models_bench.PriceTable(
        checked_on=date(2026, 9, 4),
        source="https://example.test/pricing",
        currency="USD",
        unit="per 1M tokens",
        models=models,
    )


# --------------------------------------------------------------------------
# Цена (SPEC §10, §16)
# --------------------------------------------------------------------------


def test_call_cost_formula_on_known_numbers():
    """Числа подобраны так, чтобы каждое слагаемое формулы дало отдельный,
    легко проверяемый вклад: input=1, cached_input=0.5, output=2 ($/1M),
    2M входных токенов (без кэша) и 0.5M выходных."""
    price = models_bench.ModelPrice(input=1.0, cached_input=0.5, output=2.0)
    cost = models_bench._call_cost(2_000_000, 500_000, None, price)
    # 2_000_000 * 1.0 + 0 * 0.5 + 500_000 * 2.0 = 3_000_000, /1e6 = 3.0
    assert cost == pytest.approx(3.0)


def test_call_cost_is_none_when_usage_did_not_arrive():
    price = models_bench.ModelPrice(input=1.0, cached_input=0.5, output=2.0)
    assert models_bench._call_cost(None, 100, None, price) is None
    assert models_bench._call_cost(100, None, None, price) is None


def test_cached_tokens_make_the_same_call_cheaper():
    """SPEC §10: повторный промпт биллится по input в 10 раз дешевле — при
    ОДИНАКОВЫХ prompt_tokens/completion_tokens ячейка с cached_tokens=800
    обязана стоить меньше, чем ячейка с cached_tokens=0."""
    price = models_bench.ModelPrice(input=0.2, cached_input=0.02, output=0.2)
    cost_no_cache = models_bench._call_cost(1000, 50, 0, price)
    cost_cached = models_bench._call_cost(1000, 50, 800, price)
    assert cost_cached < cost_no_cache


def test_a_model_without_a_price_gives_none_not_zero():
    """SPEC §10/§16: модель без цены — «—», а не 0 (0 читалось бы как «даром»)."""
    prices = _prices()
    cell = _cell("no-such-model-in-prices", [_call(1)])
    assert cell.median_cost(prices) is None
    assert cell.cost_per_1000(prices) is None
    assert models_bench._cost_label(cell.cost_per_1000(prices), prices) == "—"


def test_cost_per_1000_is_the_median_single_response_cost_times_a_thousand():
    """SPEC §10: в таблице печатается цена за 1000 ответов, иначе колонка
    после четвёртого знака состоит из нулей."""
    price = models_bench.ModelPrice(input=0.2, cached_input=0.02, output=0.2)
    prices = _prices(m=price)
    cell = _cell("m", [_call(1, prompt=1000, completion=10)])
    assert cell.cost_per_1000(prices) == pytest.approx(cell.median_cost(prices) * 1000)
    assert cell.cost_per_1000(prices) > cell.median_cost(prices)


def test_price_older_than_the_threshold_warns(capsys):
    prices = _prices()  # checked_on = 2026-09-04
    models_bench.warn_if_prices_stale(prices, today=date(2027, 6, 1))  # ~270 дней
    assert "устар" in _flat(capsys.readouterr().err)


def test_price_within_the_threshold_is_silent(capsys):
    prices = _prices()
    models_bench.warn_if_prices_stale(prices, today=date(2026, 12, 1))  # < 180 дней
    assert capsys.readouterr().err == ""


def test_prices_json_checked_on_is_not_stale_right_now():
    """SPEC §10/§16: «checked_on не старше 180 дней» — единственный способ
    поймать забытый прайс-лист без правки этого теста: он читает текущую дату
    по-настоящему (не инъекцией), поэтому и обязан покраснеть сам собой через
    полгода после checked_on, если файл никто не обновит."""
    prices = models_bench.load_prices()
    age = date.today() - prices.checked_on
    assert age.days <= models_bench.PRICE_STALE_DAYS


def test_all_three_ladder_models_have_a_price():
    ladder = BY_NAME["models"].defaults[BENCH_COMMAND]
    prices = models_bench.load_prices()
    missing = [name for name in ladder if prices.price_of(name) is None]
    assert missing == [], f"без цены: {missing}"


# --------------------------------------------------------------------------
# Метрики: медиана, а не среднее (SPEC §9, §16)
# --------------------------------------------------------------------------


def test_latency_ms_is_the_median_not_the_mean_across_calls():
    """Выброс 900 при остальных 100: медиана (100) им не двигается, среднее
    (260) — сдвигается сильно. Проверено мутацией: временная замена
    statistics.median на statistics.mean в Cell.latency_ms красит именно этот
    тест (см. отчёт агента)."""
    calls = [_call(i, latency_ms=v) for i, v in enumerate([100, 100, 100, 100, 900], start=1)]
    cell = _cell("m", calls)
    assert cell.latency_ms == 100


def test_ms_per_token_is_the_median_of_per_call_ratios_not_the_ratio_of_medians():
    """SPEC §9 отдельным пунктом: отношение медиан дало бы другое число.
    Три вызова подобраны так, чтобы оба варианта считали по-разному:

        latency  tokens  ratio
          100      10     10
          200      50      4
          900      10     90

    Медиана latency = 200, медиана tokens = 10 → «отношение медиан» = 20.
    Медиана отношений [4, 10, 90] = 10. Значения различаются (20 != 10) —
    ровно то расхождение, которое SPEC требует не допустить.
    """
    calls = [
        _call(1, latency_ms=100, completion=10),
        _call(2, latency_ms=200, completion=50),
        _call(3, latency_ms=900, completion=10),
    ]
    cell = _cell("m", calls)
    assert cell.ms_per_token == pytest.approx(10.0)
    assert cell.ms_per_token != pytest.approx(20.0), "это было бы отношение медиан, не по SPEC"


def test_ms_per_token_ignores_calls_with_zero_or_missing_completion_tokens():
    calls = [
        _call(1, latency_ms=100, completion=0),
        _call(2, latency_ms=200, completion=None),
        _call(3, latency_ms=300, completion=10),
    ]
    cell = _cell("m", calls)
    assert cell.ms_per_token == pytest.approx(30.0)


# --------------------------------------------------------------------------
# Точность открытой задачи — None, не 0 (SPEC §9, §16)
# --------------------------------------------------------------------------


def test_open_problem_accuracy_is_none_not_zero():
    calls = [
        models_bench.BenchCall(
            index=1,
            result=CallResult(text="небо голубое из-за рэлеевского рассеяния", usage=Usage(50, 60)),
            text="небо голубое из-за рэлеевского рассеяния",
            normalized="небо голубое из-за рэлеевского рассеяния",
            check=None,
        )
    ]
    cell = _cell("m", calls, problem=SKY)
    assert cell.accuracy is None


def test_non_open_problem_reports_accuracy_as_a_fraction():
    calls = [_call(1, text="ОТВЕТ: 7"), _call(2, text="ОТВЕТ: 2")]
    cell = _cell("m", calls, problem=CHILDREN)
    assert cell.accuracy == (1, 2)


# --------------------------------------------------------------------------
# Лестница: --models по умолчанию — три модели по возрастанию (SPEC §8, §16)
# --------------------------------------------------------------------------


def test_bench_default_models_are_the_three_ministral_in_ascending_order():
    ladder = BY_NAME["models"].defaults[BENCH_COMMAND]
    assert ladder == ["ministral-3b-latest", "ministral-8b-latest", "ministral-14b-latest"]


def test_bench_problems_are_children_then_sky_in_that_order():
    """§11: children (эталон) идёт первой — главный замер дня, sky (open)
    следует за ней. Порядок содержательный, не алфавитный (c < s случайно
    совпадает, но полагаться на это нельзя)."""
    assert models_bench.BENCH_PROBLEMS == ("children", "sky")


# --------------------------------------------------------------------------
# Вывод: на плоской колонке — ничья, а не выдуманный победитель (CLAUDE.md,
# ловушка Day 04: max()/min() без обнаружения ничьей молча берёт первую
# ячейку по порядку --models).
# --------------------------------------------------------------------------


def _flat_pair(
    price_a: float, price_b: float, latency: float = 500.0
) -> tuple[models_bench.PriceTable, list[models_bench.Cell]]:
    price = models_bench.ModelPrice(input=0.1, cached_input=0.01, output=price_a)
    prices = _prices(**{"model-a": price, "model-b": replace_output(price, price_b)})
    calls_a = [_call(1, latency_ms=latency, prompt=100, completion=10)]
    calls_b = [_call(1, latency_ms=latency, prompt=100, completion=10)]
    cells = [_cell("model-a", calls_a), _cell("model-b", calls_b)]
    return prices, cells


def replace_output(price: models_bench.ModelPrice, output: float) -> models_bench.ModelPrice:
    return models_bench.ModelPrice(
        input=price.input, cached_input=price.cached_input, output=output
    )


def test_print_conclusions_calls_a_flat_price_column_a_tie_not_a_winner(stdout_capture):
    """Обе модели — одинаковая цена за токен и, значит (при равном usage),
    одинаковая цена за 1000 ответов. Строка не имеет права назвать ни одну из
    двух единственным победителем."""
    prices, cells = _flat_pair(price_a=0.2, price_b=0.2)
    sweep = models_bench.ProblemSweep(problem=CHILDREN, cells=tuple(cells))
    models_bench.print_conclusions([sweep], prices)

    printed = _flat(stdout_capture.getvalue())
    assert "ничья" in printed
    assert "model-a" in printed and "model-b" in printed


def test_print_conclusions_names_a_real_price_winner_when_there_is_one(stdout_capture):
    """Контрольный случай: реальная разница по цене за токен обязана назвать
    победителя по имени — починка ничьей не должна стирать настоящий результат."""
    prices, cells = _flat_pair(price_a=0.1, price_b=0.4)
    sweep = models_bench.ProblemSweep(problem=CHILDREN, cells=tuple(cells))
    models_bench.print_conclusions([sweep], prices)

    printed = _flat(stdout_capture.getvalue())
    assert "ничья" not in printed or "быстрее" in printed
    assert "model-a" in printed


def test_print_conclusions_calls_a_flat_latency_column_a_tie_not_a_winner(stdout_capture):
    prices = _prices()
    calls_a = [_call(1, latency_ms=500, completion=10)]
    calls_b = [_call(1, latency_ms=500, completion=10)]
    sweep = models_bench.ProblemSweep(
        problem=CHILDREN, cells=(_cell("model-a", calls_a), _cell("model-b", calls_b))
    )
    models_bench.print_conclusions([sweep], prices)

    printed = _flat(stdout_capture.getvalue())
    assert "ничья" in printed


# --------------------------------------------------------------------------
# DEFAULT_MODEL: верхняя ступень лестницы, присутствует в prices.json (SPEC §16)
# --------------------------------------------------------------------------


def test_default_model_is_the_top_of_the_bench_ladder():
    ladder = BY_NAME["models"].defaults[BENCH_COMMAND]
    assert ladder[-1] == DEFAULT_MODEL


def test_default_model_is_priced():
    prices = models_bench.load_prices()
    assert prices.price_of(DEFAULT_MODEL) is not None


# --------------------------------------------------------------------------
# Лимит частоты: живой заголовок против затравки (SPEC-w01d05.md §13)
# --------------------------------------------------------------------------


def _no_sleep(monkeypatch):
    """Убирает реальный сон и возвращает список запрошенных пауз."""
    slept: list[float] = []
    monkeypatch.setattr(models_bench.time, "sleep", slept.append)
    return slept


def test_live_header_overrides_the_seeded_limit(monkeypatch):
    """Затравка нужна только на первый вызов — дальше правит заголовок ответа.

    Иначе пауза считалась бы по числу, измеренному однажды и зашитому в код:
    ровно тот второй источник истины, на котором проект уже обжёгся с
    maximum=2.0 для температуры.
    """
    slept = _no_sleep(monkeypatch)
    now = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(models_bench.time, "monotonic", lambda: next(now))

    limits = models_bench.RateLimits()
    assert limits.limits["ministral-14b-latest"] == 30  # затравка: 60/30 = 2 с

    limits.pause_before("ministral-14b-latest")  # первый вызов — паузы нет
    limits.observe("ministral-14b-latest", 600)  # заголовок: 60/600 = 0.1 с
    limits.pause_before("ministral-14b-latest")

    assert slept and slept[0] == pytest.approx(0.1)


def test_a_missing_header_keeps_the_seeded_limit(monkeypatch):
    """None — «заголовка не было», а не «лимита нет». Затирать затравку
    отсутствием данных значило бы перестать тормозить именно тогда, когда
    известно меньше всего."""
    slept = _no_sleep(monkeypatch)
    now = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(models_bench.time, "monotonic", lambda: next(now))

    limits = models_bench.RateLimits()
    limits.pause_before("ministral-14b-latest")
    limits.observe("ministral-14b-latest", None)
    limits.pause_before("ministral-14b-latest")

    assert slept and slept[0] == pytest.approx(2.0)


def test_a_zero_limit_does_not_sleep(monkeypatch):
    """Ноль — «модель недоступна на тарифе» (так Mistral отдаёт для
    mistral-small на этом аккаунте), а не «жди бесконечно»: 60/0 — деление на
    ноль, и любая пауза здесь всё равно кончилась бы 429."""
    slept = _no_sleep(monkeypatch)
    now = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(models_bench.time, "monotonic", lambda: next(now))

    limits = models_bench.RateLimits()
    limits.pause_before("mistral-small-latest")
    limits.observe("mistral-small-latest", 0)
    limits.pause_before("mistral-small-latest")

    assert slept == []


def test_an_unknown_model_is_not_paced(monkeypatch):
    """Лимита не знаем — не тормозим и не выдумываем константу."""
    slept = _no_sleep(monkeypatch)
    now = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(models_bench.time, "monotonic", lambda: next(now))

    limits = models_bench.RateLimits()
    limits.pause_before("совсем-новая-модель")
    limits.pause_before("совсем-новая-модель")

    assert slept == []


def test_limits_are_tracked_per_model_not_per_account(monkeypatch):
    """У Mistral лимит помодельный: 2026-09-04 на одном ключе 3b отдавал 750,
    14b — 30. Один общий счётчик смешал бы паузы разных моделей."""
    slept = _no_sleep(monkeypatch)
    monkeypatch.setattr(models_bench.time, "monotonic", lambda: 0.0)

    limits = models_bench.RateLimits()
    limits.pause_before("ministral-3b-latest")
    limits.pause_before("ministral-14b-latest")
    limits.pause_before("ministral-3b-latest")

    # 3b: 60/750 = 0.08 с — ниже порога 0.05? нет, выше, значит одна пауза
    # и ровно 3b-шная, а не 14b-шная (2 с).
    assert slept == [pytest.approx(0.08)]


# --------------------------------------------------------------------------
# Вывод: точность — первая ось задания и обязана звучать словами (SPEC §16)
# --------------------------------------------------------------------------


def _accuracy_sweep(*answers: tuple[str, list[str]]) -> models_bench.ProblemSweep:
    """Ячейки по (модель, список текстов ответов). "ОТВЕТ: 7" — верно."""
    cells = [
        _cell(model, [_call(i, text=text) for i, text in enumerate(texts, 1)])
        for model, texts in answers
    ]
    return models_bench.ProblemSweep(problem=CHILDREN, cells=tuple(cells))


def test_conclusions_name_the_most_accurate_model_aloud(stdout_capture):
    """Ловушка Day 04: метрика, посчитанная в таблице, но не названная в
    выводе, задание не закрывает — зритель не обязан читать колонку сам."""
    sweep = _accuracy_sweep(
        ("model-a", ["ОТВЕТ: 6", "ОТВЕТ: 6"]),
        ("model-b", ["ОТВЕТ: 7", "ОТВЕТ: 7"]),
    )
    models_bench.print_conclusions([sweep], _prices())

    printed = _flat(stdout_capture.getvalue())
    assert "точнее всего" in printed
    assert "model-b" in printed


def test_a_flat_accuracy_column_is_called_a_tie(stdout_capture):
    """Обе модели правы одинаково часто — победителя нет, и `max()` не имеет
    права назначить первую по порядку --models."""
    sweep = _accuracy_sweep(
        ("model-a", ["ОТВЕТ: 7", "ОТВЕТ: 6"]),
        ("model-b", ["ОТВЕТ: 7", "ОТВЕТ: 6"]),
    )
    models_bench.print_conclusions([sweep], _prices())

    printed = _flat(stdout_capture.getvalue())
    assert "ничья" in printed
    assert "точнее всего" not in printed


def test_nobody_correct_is_reported_as_no_separation_not_as_a_winner(stdout_capture):
    """Ноль у всех — «задача не разделяет», а не «лучшие вот эти». Формально
    максимум существует и там, но назвать его лучшим было бы враньём: ровно
    та ошибка, из-за которой Day 04 печатал «точнее всего — t=0 (0/3)»."""
    sweep = _accuracy_sweep(
        ("model-a", ["ОТВЕТ: 6"]),
        ("model-b", ["ОТВЕТ: 5"]),
    )
    models_bench.print_conclusions([sweep], _prices())

    printed = _flat(stdout_capture.getvalue())
    assert "не ответила ни одна" in printed
    assert "точнее всего" not in printed


def test_an_open_problem_gets_no_accuracy_line_at_all(stdout_capture):
    """У sky нет эталона. «0 из N» читалось бы как «ни разу не угадала»
    вместо «мерить нечем» — строки про точность там быть не должно вовсе."""
    cells = [
        _cell("model-a", [_call(1, text="небо голубое из-за рассеяния")], problem=SKY),
        _cell("model-b", [_call(1, text="потому что атмосфера")], problem=SKY),
    ]
    sweep = models_bench.ProblemSweep(problem=SKY, cells=tuple(cells))
    models_bench.print_conclusions([sweep], _prices())

    printed = _flat(stdout_capture.getvalue())
    assert "точнее всего" not in printed
    assert "не ответила ни одна" not in printed
    assert "точность одинакова" not in printed


# --------------------------------------------------------------------------
# Блок ответов для человека: один ответ на модель, а не все N
# --------------------------------------------------------------------------


def test_human_review_shows_one_answer_per_model_not_every_run(stdout_capture):
    """Ось сравнения этого дня — модели, а не прогоны. Все N ответов на модель
    втрое удлиняют таблицу и хоронят сравнение: на dry-run перед записью блок
    занимал 255 строк из 537, половину ролика. Прогоны при этом не теряются —
    каждый печатается по мере поступления и пишется в журнал."""
    cells = [
        _cell(
            "model-a",
            [_call(1, text="ПЕРВЫЙ-A"), _call(2, text="ВТОРОЙ-A"), _call(3, text="ТРЕТИЙ-A")],
            problem=SKY,
        ),
        _cell("model-b", [_call(1, text="ПЕРВЫЙ-B"), _call(2, text="ВТОРОЙ-B")], problem=SKY),
    ]
    models_bench.print_human_review(SKY, cells)

    printed = _flat(stdout_capture.getvalue())
    assert "ПЕРВЫЙ-A" in printed and "ПЕРВЫЙ-B" in printed
    assert "ВТОРОЙ-A" not in printed
    assert "ТРЕТИЙ-A" not in printed
    assert "ВТОРОЙ-B" not in printed
    # Читатель обязан знать, что видит один прогон из нескольких.
    assert "первый прогон" in printed


def test_human_review_does_not_claim_one_of_many_on_a_single_run(stdout_capture):
    """При --runs 1 оговорка «первый прогон из каждой ячейки» была бы шумом:
    других прогонов и нет."""
    cells = [_cell("model-a", [_call(1, text="ЕДИНСТВЕННЫЙ")], problem=SKY)]
    models_bench.print_human_review(SKY, cells)

    printed = _flat(stdout_capture.getvalue())
    assert "ЕДИНСТВЕННЫЙ" in printed
    assert "первый прогон" not in printed


def test_human_review_is_silent_for_a_closed_problem(stdout_capture):
    """У children есть эталон и колонка точности — сравнивать ответы глазами
    там нечего, а «ОТВЕТ: 7» против «ОТВЕТ: 6» в таблице только шум."""
    models_bench.print_human_review(CHILDREN, [_cell("model-a", [_call(1)])])

    assert stdout_capture.getvalue() == ""


# --------------------------------------------------------------------------
# Печать ответа на экран обрезается, измерение — нет
# --------------------------------------------------------------------------


def test_a_short_answer_is_printed_whole():
    text = "ОТВЕТ: 7"

    assert models_bench._shorten_for_screen(text) == (text, 0)


def test_a_tail_shorter_than_the_slack_is_not_cut_at_all():
    """Обрезка ради одного лишнего символа — шум, который ещё и заставляет
    читателя гадать, чего он не увидел. Поймано на третьем dry-run: ответ
    переваливал лимит ровно на 1 символ, и на экране появилось «…ещё 1
    символов»."""
    text = "я" * (models_bench.ANSWER_PRINT_CHARS + 1)

    assert models_bench._shorten_for_screen(text) == (text, 0)


def test_a_long_answer_is_cut_by_lines():
    """Найдено на dry-run: ministral-8b выдал 4205 токенов одной ячейкой
    (зациклился на «Ошибка в понимании задачи») — 442 строки из 741 в выводе,
    больше половины ролика на один вызов."""
    text = "\n".join(f"строка {i}" for i in range(100))

    shown, hidden = models_bench._shorten_for_screen(text)

    assert shown.count("\n") + 1 == models_bench.ANSWER_PRINT_LINES
    assert hidden == len(text) - len(shown)
    assert hidden > 0


def test_a_long_answer_without_newlines_is_still_cut():
    """Лимита по строкам мало: ответ может прийти одним абзацем без единого
    перевода строки, и тогда splitlines() насчитает ровно одну строку, а
    терминал развернёт её на десятки переносами."""
    text = "я" * 9000

    shown, hidden = models_bench._shorten_for_screen(text)

    assert len(shown) <= models_bench.ANSWER_PRINT_CHARS
    assert hidden == len(text) - len(shown)


def test_truncation_never_touches_the_measurement():
    """Обрезается ТОЛЬКО печать. Ограничить саму генерацию через max_tokens
    было бы неверно: обрезанный ответ изменил бы и число токенов, и цену,
    то есть испортил бы ровно те метрики, которые день измеряет."""
    long_text = "\n".join(f"строка {i}" for i in range(500))
    cell = _cell("model-a", [_call(1, completion=4205, text=long_text)])

    assert cell.output_tokens == 4205
    shown, hidden = models_bench._shorten_for_screen(long_text)
    assert hidden > 0
    assert len(shown) < len(long_text)


def test_the_hidden_tail_is_announced_on_stderr_not_silently_dropped(capsys):
    """Молча обрезанный ответ читался бы как ответ модели целиком — то есть
    экран врал бы про то, что произошло."""
    long_text = "\n".join(f"строка {i}" for i in range(100))
    step = models_bench.BenchStep(
        model="model-a",
        problem=CHILDREN,
        run=1,
        runs=1,
        result=CallResult(text=long_text, usage=Usage(100, 900)),
        messages=[],
        text=long_text,
        normalized=strategies.normalize(long_text),
        check=None,
        first_in_cell=True,
    )

    models_bench.print_bench_step(step)
    captured = capsys.readouterr()

    assert "показано начало ответа" in captured.err
    assert "строка 99" not in captured.out
