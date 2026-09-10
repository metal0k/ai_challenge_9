"""console: эхо piped-ввода и footer() с частичным usage.

echo_input появилось в день 08: гигантский ввод (~700 КБ одной строкой) при
полном эхе прокручивал терминал на тысячи строк — в кадре важен факт
огромности ввода, а не его текст (SPEC-w02d08.md §7, шаг 4б).
"""

from advent_core import console
from advent_core.telemetry import CallResult, Usage


def test_echo_short_passes_through(capsys):
    console.echo_input("привет")
    assert capsys.readouterr().err.strip() == "привет"


def test_echo_long_is_truncated_with_length(capsys):
    # Литерал 200, а не console.ECHO_LIMIT: ожидание, построенное из той же
    # константы, что и код, не покраснеет при случайном изменении порога
    # (правило проекта — тест не должен брать эталон из источника под тестом).
    assert console.ECHO_LIMIT == 200
    text = "а" * (console.ECHO_LIMIT + 500)
    console.echo_input(text)
    # rich переносит длинную строку по ширине консоли — сравниваем без \n:
    # перенос дисплея не часть содержимого.
    out = capsys.readouterr().err.replace("\n", "")
    # Обрезанный кусок, многоточие и полная длина ввода — «неизвестно»
    # не становится молчанием даже в эхе.
    assert out == f"{'а' * console.ECHO_LIMIT}… (всего {len(text)} символов)"
    assert len(text) > console.ECHO_LIMIT  # сам тест не на коротком вводе


def _plain(text: str) -> str:
    """Снимает rich-подсветку: FORCE_COLOR в окружении добавляет ANSI-коды."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_footer_partial_usage_puts_estimate_in_the_middle(capsys):
    # {"total_tokens": N} без completion — форма usage от локальных серверов
    # (SPEC-w02d08.md §6): completion_estimate должен встать в среднюю позицию
    # тройки, а не печататься отдельной строкой.
    result = CallResult(
        text="ok",
        model_requested="m",
        model_actual="m",
        latency_ms=1,
        usage=Usage(prompt_tokens=10, total_tokens=15),
    )
    console.footer(result, completion_estimate=7)
    err = _plain(capsys.readouterr().err)
    assert "tokens 10/~7/15" in err


def test_footer_partial_usage_without_estimate_is_a_dash(capsys):
    # Usage(total_tokens=51) без prompt/completion — раньше здесь печаталось
    # буквальное "tokens None/None/51"; правило проекта — «неизвестно» это
    # прочерк, а не выдуманное значение и не слово None.
    result = CallResult(
        text="ok",
        model_requested="m",
        model_actual="m",
        latency_ms=1,
        usage=Usage(total_tokens=51),
    )
    console.footer(result)
    err = _plain(capsys.readouterr().err)
    assert "tokens —/—/51" in err
    assert "None" not in err
