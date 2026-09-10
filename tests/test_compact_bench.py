"""Offline harness of day 09 (SPEC-w02d09.md §13): helpers and the dry-run path.

The network path itself is not covered here — it is verified by an actual run
before the recording. What these tests hold is the arithmetic the day's headline
numbers are built from, and the promise that `--dry-run` costs nothing.
"""

from __future__ import annotations

import pytest

from advent_core.config import ConfigError
from advent_core.params import GenerationParams
from tools import compact_bench


def test_cumulative_adds_up_and_reports_the_total():
    rows, total, missing = compact_bench._cumulative([10, 20, 30])

    assert rows == [("10", "10"), ("20", "30"), ("30", "60")]
    assert total == 60
    assert missing == 0


def test_a_gap_is_a_dash_in_both_columns_and_does_not_carry_forward():
    """Missing usage is unknown, not zero: printing the previous cumulative
    value again would invent a number the server never sent."""
    rows, total, missing = compact_bench._cumulative([10, None, 5])

    assert rows == [("10", "10"), ("—", "—"), ("5", "15")]
    assert total == 15
    assert missing == 1


def test_all_unknown_gives_no_total_rather_than_zero():
    rows, total, missing = compact_bench._cumulative([None, None])

    assert rows == [("—", "—"), ("—", "—")]
    assert total is None
    assert missing == 2


def test_net_savings_subtracts_what_the_compactions_cost():
    """The day's headline number. Ignoring the summarizer calls would report a
    saving the run did not make."""
    assert compact_bench._net_savings(1000, 600, 150) == 250


def test_net_savings_is_unknown_when_either_side_is():
    assert compact_bench._net_savings(None, 600, 0) is None
    assert compact_bench._net_savings(1000, None, 0) is None


def test_the_verdict_names_the_memory_bought_when_tokens_were_lost():
    """The measured case: compaction costs more and keeps what trimming dropped.

    A negative number on its own reads as a failed feature; the day's actual
    result is a trade, and the line has to carry both halves.
    """
    line = compact_bench._verdict(-3464, off_remembered=False, on_remembered=True)

    assert "дороже" in line
    assert "3464" in line
    assert "только сжатие" in line
    # The minus sign belongs to the word "дороже", not to the number after it.
    assert "-3464" not in line and "−3464" not in line


def test_the_verdict_says_when_both_runs_remembered():
    """Nothing was bought here — a run where trimming loses nothing must not be
    reported as if compaction had rescued something."""
    line = compact_bench._verdict(-867, off_remembered=True, on_remembered=True)

    assert "помнят оба" in line


def test_the_verdict_reports_a_real_saving_as_one():
    assert "дешевле обрезки на 250" in compact_bench._verdict(
        250, off_remembered=False, on_remembered=True
    )


def test_the_verdict_of_an_unknown_net_is_not_a_number():
    """`None` is "the server didn't say", and it must not turn into a zero
    verdict — the same rule the dash in the table follows."""
    line = compact_bench._verdict(None, off_remembered=True, on_remembered=True)

    assert "usage" in line
    assert "поровну" not in line


def test_mode_params_differ_only_by_compact_and_leave_the_base_alone():
    base = GenerationParams()

    on = compact_bench._mode_params(base, compact=True, keep_last=4, compact_every=6)
    off = compact_bench._mode_params(base, compact=False, keep_last=4, compact_every=6)

    assert on.compact is True
    assert off.compact is False
    assert (on.keep_last, on.compact_every) == (off.keep_last, off.compact_every) == (4, 6)
    assert base.compact is None and base.keep_last is None


def test_mode_params_defers_the_bounds_to_the_registry():
    """`keep_last >= 2` is written down once, in advent_core/params.py; a copy
    of the rule here would drift out of sync with it."""
    with pytest.raises(ConfigError):
        compact_bench._mode_params(
            GenerationParams(), compact=True, keep_last=1, compact_every=None
        )


def test_a_shortened_run_still_ends_with_the_codeword_question():
    """`--turns` exists so the take doesn't play the same dialog for minutes.
    A plain slice would cut the last turn — the day's only quality check — and
    both arms would be compared without ever being asked for the codeword."""
    short = compact_bench.scenario_for(4)

    assert len(short) == 4
    assert short[:3] == compact_bench.SCENARIO[:3]
    assert short[-1] == compact_bench.SCENARIO[-1]
    assert compact_bench.CODEWORD in short[0]


def test_turns_at_or_beyond_the_scenario_length_leaves_it_whole():
    full = compact_bench.SCENARIO

    assert compact_bench.scenario_for(None) == full
    assert compact_bench.scenario_for(len(full)) == full
    assert compact_bench.scenario_for(len(full) + 5) == full


def test_too_few_turns_are_refused_before_anything_is_spent(monkeypatch):
    """Two turns is a codeword and a question about it with no history in
    between: trim has nothing to drop, so the comparison measures nothing."""

    def boom(*args, **kwargs):
        raise AssertionError("проверка --turns пропущена")

    monkeypatch.setattr(compact_bench.Config, "resolve", boom)

    with pytest.raises(SystemExit) as excinfo:
        compact_bench.main(["--turns", "2"])

    assert excinfo.value.code == 2


def test_dry_run_shows_only_the_turns_that_will_actually_run(capsys):
    """The rehearsal has to show the shortened scenario, not the full one —
    otherwise it rehearses a different run than the take will play."""
    assert compact_bench.main(["--dry-run", "--turns", "4"]) == 0

    out = capsys.readouterr().out
    assert compact_bench.SCENARIO[-1][:30] in out
    assert compact_bench.SCENARIO[5][:30] not in out


def test_dry_run_prints_the_scenario_and_never_reaches_the_network(monkeypatch, capsys):
    """`--dry-run` exists to rehearse the harness for free; a run that resolved
    a config or listed models would spend credits to print a table."""

    def boom(*args, **kwargs):
        raise AssertionError("--dry-run ушёл в сеть")

    monkeypatch.setattr(compact_bench, "list_models", boom)
    monkeypatch.setattr(compact_bench.Config, "resolve", boom)

    assert compact_bench.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert compact_bench.CODEWORD in out
    assert str(compact_bench.DEFAULT_LIMIT) in out


def test_a_non_positive_limit_is_refused_before_anything_is_spent(monkeypatch):
    """At limit 0 there is no trim to compare against — and the refusal has to
    come before the two runs, not after them."""

    def boom(*args, **kwargs):
        raise AssertionError("проверка лимита пропущена")

    monkeypatch.setattr(compact_bench.Config, "resolve", boom)

    with pytest.raises(SystemExit) as excinfo:
        compact_bench.main(["--limit", "0"])

    assert excinfo.value.code == 2
