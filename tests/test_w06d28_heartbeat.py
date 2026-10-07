"""Heartbeat: one rewritten line on a tty, a sparse full line otherwise."""

from __future__ import annotations

import pytest

from week_06 import ragbench

CLEAR = ragbench._CLEAR_LINE  # carriage return + clear-to-end-of-line


@pytest.fixture(autouse=True)
def _reset_tick_state(monkeypatch):
    monkeypatch.setattr(ragbench, "_tick_visible", False)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _beat(tty: bool, clock: FakeClock) -> ragbench.Heartbeat:
    beat = ragbench.Heartbeat("local #9 прогон 1", is_tty=lambda: tty, clock=clock)
    beat._start = beat._last_line = clock()  # what __enter__ does, minus the thread
    return beat


def test_clear_sequence_is_cr_plus_erase_line():
    assert CLEAR.startswith(chr(13))
    assert CLEAR.endswith(chr(27) + "[K")


def test_tty_ticks_rewrite_one_line_without_newlines(capsys):
    clock = FakeClock()
    beat = _beat(True, clock)
    for _ in range(3):
        clock.now += 3
        beat._pulse()
    err = capsys.readouterr().err
    assert err.count(CLEAR) == 3
    assert "\n" not in err
    assert "9 s" in err
    ragbench.say_err("done", markup=False)
    assert capsys.readouterr().err.startswith(CLEAR)


def test_tty_line_is_cleared_before_a_stage_line(capsys):
    clock = FakeClock()
    beat = _beat(True, clock)
    clock.now += 3
    beat._pulse()
    capsys.readouterr()
    ragbench.say_err("  rerank 1.0 s", markup=False, highlight=False)
    err = capsys.readouterr().err
    assert err.startswith(CLEAR)
    assert "rerank 1.0 s" in err
    ragbench.say_err("next", markup=False)
    assert CLEAR not in capsys.readouterr().err  # already clear


def test_tty_exit_clears_the_line(capsys):
    beat = ragbench.Heartbeat("x", is_tty=lambda: True)
    with beat:
        beat._pulse()
    assert capsys.readouterr().err.endswith(CLEAR)


def test_non_tty_prints_a_full_line_at_most_every_10_seconds(capsys):
    clock = FakeClock()
    beat = _beat(False, clock)
    lines = []
    for _ in range(8):  # 3 s steps over 24 s
        clock.now += 3
        beat._pulse()
        lines.extend(capsys.readouterr().err.splitlines())
    assert len(lines) == 2  # at 12 s and 24 s
    assert chr(13) not in "".join(lines)
    assert "12 s" in lines[0]
    assert "24 s" in lines[1]
