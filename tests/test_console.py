"""console.echo_input: эхо piped-ввода — длинная строка режется с честной длиной.

Появилось в день 08: гигантский ввод (~700 КБ одной строкой) при полном эхе
прокручивал терминал на тысячи строк — в кадре важен факт огромности ввода,
а не его текст (SPEC-w02d08.md §7, шаг 4б).
"""

from advent_core import console


def test_echo_short_passes_through(capsys):
    console.echo_input("привет")
    assert capsys.readouterr().err.strip() == "привет"


def test_echo_long_is_truncated_with_length(capsys):
    text = "а" * (console.ECHO_LIMIT + 500)
    console.echo_input(text)
    # rich переносит длинную строку по ширине консоли — сравниваем без \n:
    # перенос дисплея не часть содержимого.
    out = capsys.readouterr().err.replace("\n", "")
    # Обрезанный кусок, многоточие и реальная длина хвоста — «неизвестно»
    # не становится молчанием даже в эхе.
    assert out == f"{'а' * console.ECHO_LIMIT}… (+500 символов)"
    assert len(text) > console.ECHO_LIMIT  # сам тест не на коротком вводе
