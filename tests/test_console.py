"""console: эхо piped-ввода и footer() с частичным usage.

echo_input появилось в день 08: гигантский ввод (~700 КБ одной строкой) при
полном эхе прокручивал терминал на тысячи строк — в кадре важен факт
огромного ввода, а не его текст (SPEC-w02d08.md §7, шаг 4б).
"""

from advent_core import console
from advent_core.telemetry import CallResult, Usage


def test_echo_short_passes_through(capsys):
    console.echo_input("привет")
    assert capsys.readouterr().err.strip() == "привет"


def test_echo_long_is_truncated_with_length(capsys):
    assert console.ECHO_LIMIT == 200
    text = "а" * (console.ECHO_LIMIT + 500)
    console.echo_input(text)
    out = capsys.readouterr().err.replace("\n", "")
    assert out == f"{'а' * console.ECHO_LIMIT}… (всего {len(text)} символов)"
    assert len(text) > console.ECHO_LIMIT


def _plain(text: str) -> str:
    """Снимает rich-подсветку: FORCE_COLOR в окружении добавляет ANSI-коды."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_footer_partial_usage_puts_estimate_in_the_middle(capsys):
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


def test_record_console_kwargs_are_empty_by_default(monkeypatch):
    monkeypatch.delenv("ADVENT_RECORD_COLOR", raising=False)
    assert console._record_console_kwargs() == {}


def test_record_console_kwargs_force_rich_colour(monkeypatch):
    monkeypatch.setenv("ADVENT_RECORD_COLOR", "1")
    assert console._record_console_kwargs() == {
        "force_terminal": True,
        "color_system": "standard",
        "legacy_windows": False,
        "no_color": False,
    }


def test_record_console_kwargs_overrides_no_color(monkeypatch):
    monkeypatch.setenv("ADVENT_RECORD_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    assert console._record_console_kwargs()["no_color"] is False


def test_enable_record_color_rebuilds_shared_consoles(monkeypatch):
    monkeypatch.delenv("ADVENT_RECORD_COLOR", raising=False)
    old_out, old_err = console.out, console.err
    try:
        console.enable_record_color()
        assert console.out._force_terminal is True
        assert console.out.color_system == "standard"
        assert console.err._force_terminal is True
        assert console.err.color_system == "standard"
    finally:
        console.out, console.err = old_out, old_err
