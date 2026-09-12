"""Pure-helper tests for tools/strategy_bench.py (SPEC-w02d10.md §13, §15).

No network anywhere: `--dry-run` tests monkeypatch Config.resolve/list_models
to raise if called, and the scoring/verdict/scenario helpers are pure
functions over plain data, mirroring tests/test_compact_bench.py's own style.
"""

from __future__ import annotations

import argparse

import pytest

from advent_core.agent import AgentReply, Compaction, FactsFailure, FactsUpdate
from advent_core.config import ConfigError
from advent_core.facts import DeltaResult
from advent_core.telemetry import CallResult, Usage
from tools import bench_core, strategy_bench


def _usage(prompt: int | None = None, completion: int | None = None) -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion)


def _call_result(
    text: str = "", prompt: int | None = None, completion: int | None = None
) -> CallResult:
    return CallResult(text=text, usage=_usage(prompt, completion))


def _reply(
    text: str = "",
    *,
    prompt: int | None = None,
    completion: int | None = None,
    compaction: Compaction | None = None,
    facts_update: FactsUpdate | None = None,
    facts_failed: FactsFailure | None = None,
) -> AgentReply:
    return AgentReply(
        text=text,
        history=[],
        result=_call_result(text, prompt, completion),
        compaction=compaction,
        facts_update=facts_update,
        facts_failed=facts_failed,
    )


def _compaction(prompt: int, completion: int) -> Compaction:
    result = _call_result(prompt=prompt, completion=completion)
    return Compaction(summary="сжато", covered=2, result=result)


def _facts_update(prompt: int, completion: int) -> FactsUpdate:
    delta = DeltaResult(facts={}, added=(), updated=(), removed=(), blocked=(), rejected=())
    return FactsUpdate(
        facts={}, delta=delta, covered=1, result=_call_result(prompt=prompt, completion=completion)
    )


def _facts_failure(prompt: int, completion: int, reason: str = "invalid") -> FactsFailure:
    return FactsFailure(result=_call_result(prompt=prompt, completion=completion), reason=reason)


# --------------------------------------------------------------------------
# normalize_numbers / detail_present / score_answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["бюджет 520 000 рублей", "бюджет 520тыс рублей", "бюджет 520 тысяч рублей"],
)
def test_normalize_numbers_collapses_the_specs_own_examples(text):
    """SPEC-w02d10.md §13.2's own three forms all collapse to a bare "520"."""
    assert "520" in strategy_bench.normalize_numbers(text)
    assert "520 000" not in strategy_bench.normalize_numbers(text)


def test_normalize_numbers_leaves_unrelated_numbers_alone():
    """ "1 декабря" and a plain "300" (no thousand marker) must not be touched —
    otherwise the regex would over-match every date and every bare number."""
    assert strategy_bench.normalize_numbers("к 1 декабря") == "к 1 декабря"
    assert strategy_bench.normalize_numbers("бюджет 300") == "бюджет 300"


def test_detail_present_matches_any_form_case_insensitively():
    assert strategy_bench.detail_present("Оплата через СБП, карты не принимаем", ("сбп",))
    assert not strategy_bench.detail_present("Оплата картой", ("сбп",))


def _full_answer() -> str:
    return (
        "Бюджет 520 тысяч рублей. Срок — к 1 декабря. Платформа только Android. "
        "Оплата через СБП. Вход по номеру телефона, без регистрации по email. "
        "Логотип уже готов, дизайн логотипа не нужен."
    )


def test_score_answer_finds_all_six_details_and_no_stale_value():
    score = strategy_bench.score_answer(_full_answer())

    assert len(score.present) == len(strategy_bench.DETAILS) == 6
    assert score.missing == ()
    assert score.stale_returned is False


def test_score_answer_reports_missing_details_by_key():
    text = _full_answer().replace("Оплата через СБП. ", "")

    score = strategy_bench.score_answer(text)

    assert "payment" in score.missing
    assert len(score.present) == 5


def test_score_answer_flags_the_stale_value_through_normalization():
    text = _full_answer() + " (раньше обсуждали 480 000 рублей)"

    score = strategy_bench.score_answer(text)

    assert score.stale_returned is True


# --------------------------------------------------------------------------
# scenario_for
# --------------------------------------------------------------------------


def test_scenario_for_none_or_full_length_returns_the_whole_scenario():
    assert strategy_bench.scenario_for(None) == strategy_bench.SCENARIO
    assert strategy_bench.scenario_for(len(strategy_bench.SCENARIO)) == strategy_bench.SCENARIO
    assert strategy_bench.scenario_for(len(strategy_bench.SCENARIO) + 3) == strategy_bench.SCENARIO


def test_scenario_for_minimum_keeps_head_and_tail_only():
    short = strategy_bench.scenario_for(strategy_bench.MIN_TURNS)

    assert len(short) == strategy_bench.MIN_TURNS
    assert short[:-1] == strategy_bench.SCENARIO[: strategy_bench.HEAD_TURNS]
    assert short[-1] == strategy_bench.SCENARIO[-1]


def test_scenario_for_middle_length_keeps_head_a_filler_slice_and_tail():
    turns = strategy_bench.MIN_TURNS + 1
    short = strategy_bench.scenario_for(turns)

    assert len(short) == turns
    head = strategy_bench.HEAD_TURNS
    assert short[:head] == strategy_bench.SCENARIO[:head]
    assert short[-1] == strategy_bench.SCENARIO[-1]
    # Exactly one filler turn survives, and it's the first one, not a random pick.
    assert short[strategy_bench.HEAD_TURNS] == strategy_bench.SCENARIO[strategy_bench.HEAD_TURNS]


def test_a_shortened_scenario_never_drops_a_planted_detail():
    """Even the minimum-length run must still plant everything the checklist
    looks for — a shortened run that silently dropped a detail would make the
    comparison measure a different, easier scenario."""
    short = " ".join(strategy_bench.scenario_for(strategy_bench.MIN_TURNS))

    assert "480" in short
    assert "520" in short
    assert "1 декабря" in short
    assert "СБП" in short


# --------------------------------------------------------------------------
# parse_strategies
# --------------------------------------------------------------------------


def test_parse_strategies_preserves_order_and_drops_duplicates():
    assert strategy_bench.parse_strategies("window, facts ,window") == ["window", "facts"]


def test_parse_strategies_rejects_an_unknown_name():
    with pytest.raises(ConfigError, match="bogus"):
        strategy_bench.parse_strategies("window,bogus")


def test_parse_strategies_rejects_an_empty_list():
    with pytest.raises(ConfigError):
        strategy_bench.parse_strategies(" , ,")


# --------------------------------------------------------------------------
# _score_run / StrategyResult
# --------------------------------------------------------------------------


def test_score_run_sums_prompt_tokens_and_side_calls_separately():
    replies = [
        _reply("вопрос-ответ 1", prompt=100, completion=10),
        _reply("вопрос-ответ 2", prompt=None, completion=5, compaction=_compaction(50, 20)),
        _reply(_full_answer(), prompt=200, completion=8, facts_update=_facts_update(30, 4)),
    ]

    result = strategy_bench._score_run("facts", replies)

    assert result.prompt_total == 300  # 100 + 200; the None turn is a gap, not a zero
    assert result.missing_usage == 1
    assert result.side_calls == 2
    assert (result.side_prompt, result.side_completion) == (80, 24)
    assert result.side_total == 104
    assert result.grand_total == 404
    assert len(result.score.present) == 6


def test_score_run_stores_the_last_replys_text_as_the_answer():
    """C1: the final "one list" answer must survive into StrategyResult so the
    caller can print it — this is the ONLY thing verify_answer-style scoring
    is checked against, so losing it here would silently drop the printout."""
    replies = [
        _reply("вопрос-ответ 1", prompt=100, completion=10),
        _reply(_full_answer(), prompt=200, completion=8),
    ]

    result = strategy_bench._score_run("window", replies)

    assert result.answer == _full_answer()


def test_score_run_captures_trim_totals_from_replies():
    """A reply carrying dropped/dropped_tokens must move StrategyResult.trim —
    otherwise a strategy gutted by the safety net prints no sign of it at all
    (the gap the 2026-09-12 live run exposed)."""
    replies = [
        _reply("вопрос-ответ 1", prompt=100, completion=10),
        AgentReply(
            text=_full_answer(),
            history=[],
            result=_call_result(_full_answer(), 200, 8),
            dropped=4,
            dropped_tokens=900,
        ),
    ]

    result = strategy_bench._score_run("branch", replies)

    assert result.trim.messages == 4
    assert result.trim.tokens == 900
    assert result.trim.turns_trimmed == 1


def test_score_run_folds_facts_failed_into_side_totals_and_tracks_it_separately():
    replies = [
        _reply("вопрос-ответ 1", prompt=100, completion=10, facts_failed=_facts_failure(35, 12)),
        _reply(_full_answer(), prompt=200, completion=8, facts_update=_facts_update(30, 4)),
    ]

    result = strategy_bench._score_run("facts", replies)

    assert result.side_calls == 2
    assert (result.side_prompt, result.side_completion) == (65, 16)
    assert (result.failed_calls, result.failed_prompt, result.failed_completion) == (1, 35, 12)


def test_grand_total_is_none_when_prompt_total_is_unknown():
    result = strategy_bench.StrategyResult(strategy="window", prompt_total=None)

    assert result.grand_total is None


# --------------------------------------------------------------------------
# verdict_lines
# --------------------------------------------------------------------------


def _result(
    strategy: str,
    *,
    prompt_total: int | None,
    side_prompt: int = 0,
    side_completion: int = 0,
    side_calls: int = 0,
    failed_calls: int = 0,
    failed_prompt: int = 0,
    failed_completion: int = 0,
    all_details: bool = True,
    stale: bool = False,
    answer: str | None = None,
    trim: bench_core.TrimTotals | None = None,
) -> strategy_bench.StrategyResult:
    score = strategy_bench.score_answer(_full_answer() if all_details else "ничего не помню")
    if stale:
        score = strategy_bench.DetailScore(
            present=score.present, missing=score.missing, stale_returned=True
        )
    return strategy_bench.StrategyResult(
        strategy=strategy,
        prompt_total=prompt_total,
        side_prompt=side_prompt,
        side_completion=side_completion,
        side_calls=side_calls,
        failed_calls=failed_calls,
        failed_prompt=failed_prompt,
        failed_completion=failed_completion,
        trim=trim if trim is not None else bench_core.TrimTotals(),
        score=score,
        answer=answer if answer is not None else _full_answer(),
    )


def test_verdict_names_facts_cost_as_a_finding_not_a_failure():
    results = [
        _result("window", prompt_total=100),
        _result("facts", prompt_total=100, side_prompt=800, side_completion=45, side_calls=12),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "800" in lines and "45" in lines
    assert "не поломка" in lines


def test_verdict_approves_a_branch_that_is_lossless_and_priciest():
    results = [
        _result("window", prompt_total=100, all_details=False),
        _result("branch", prompt_total=1000),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "lossless-базлайн" in lines


def test_verdict_reports_a_branch_below_six_of_six_as_model_variation_not_a_broken_harness():
    """2026-09-12: three live runs on identical parameters scored 5/6, 5/6,
    6/6 — the checker matched the model's own wording on the 6/6 run, so a
    branch score below 6/6 is the model choosing what to restate, not a
    broken scenario or checker."""
    results = [
        _result("window", prompt_total=100),
        _result("branch", prompt_total=1000, all_details=False),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "не хватает" in lines
    assert "её решение, а не свойство стратегии" in lines
    assert "сломан сценарий либо проверка" not in lines


def test_verdict_flags_a_branch_that_is_not_the_priciest():
    results = [
        _result("facts", prompt_total=100, side_prompt=5000, side_completion=0),
        _result("branch", prompt_total=1000),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "lossless-базлайн" in lines
    assert "не самый дорогой прогон" in lines
    assert "шлёт всю историю" in lines
    assert "сломан" not in lines


def test_verdict_names_the_trim_when_branchs_own_numbers_show_it_was_cut():
    """The trim is a data-driven ADDENDUM to the details line, appended only
    when THIS run's own trim numbers show branch was actually cut — the
    2026-09-12 measurement refuted trim as a GENERAL explanation for a
    branch miss, but a run whose own numbers show a cut still says so."""
    trimmed = bench_core.TrimTotals(messages=4, tokens=900, turns_trimmed=2)
    results = [
        _result("window", prompt_total=100),
        _result("branch", prompt_total=1000, all_details=False, trim=trimmed),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "safety-net" in lines
    assert "2 ход" in lines
    assert "4 сообщ" in lines
    assert "900 токенов" in lines


def test_verdict_says_nothing_about_trim_when_branch_was_not_cut():
    """The default TrimTotals() (0 turns trimmed) must not fire the addendum —
    same run as
    test_verdict_reports_a_branch_below_six_of_six_as_model_variation_not_a_broken_harness,
    but checked for absence of the trim sentence specifically."""
    results = [
        _result("window", prompt_total=100),
        _result("branch", prompt_total=1000, all_details=False),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "safety-net" not in lines


def test_verdict_reports_a_stale_return_as_its_own_line():
    """stale_returned is a separate fact from a missing detail (line B), and
    must print even alongside a missing-details line (A) rather than being
    folded into it."""
    results = [
        _result("window", prompt_total=100),
        _result("branch", prompt_total=1000, all_details=False, stale=True),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "устаревшее значение 480" in lines
    assert "не хватает" in lines


def test_verdict_does_not_claim_a_price_comparison_with_a_single_strategy():
    """A `--strategies branch` run has nothing to compare price against —
    claiming "не самый дорогой прогон" about it would report a comparison
    that never happened (2026-09-12 finding)."""
    results = [_result("branch", prompt_total=1000, all_details=False)]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "не самый дорогой прогон" not in lines
    assert "сравнивать не с чем" in lines


def test_verdict_approves_a_lossless_single_strategy_branch_run():
    """Same single-strategy case, but lossless: must still print the
    lossless-baseline line, minus the price claim it can't back up."""
    results = [_result("branch", prompt_total=1000)]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "lossless-базлайн" in lines
    assert "не самый дорогой прогон" not in lines


def test_verdict_never_declares_a_winner_on_a_tie():
    """CLAUDE.md: max()/min() over a tie silently names a winner nobody won."""
    results = [_result("window", prompt_total=500), _result("summary", prompt_total=500)]

    lines = strategy_bench.verdict_lines(results)
    combined = " ".join(lines)

    assert "ничья" in combined
    assert combined.count("500") >= 1


def test_verdict_names_a_single_cheapest_strategy():
    results = [_result("window", prompt_total=100), _result("summary", prompt_total=900)]

    combined = " ".join(strategy_bench.verdict_lines(results))

    assert "дешевле всех — window" in combined


def test_verdict_names_a_failed_facts_call_as_paid_for_and_wasted():
    """C3: a facts_failed call must be called out by name, not just folded
    silently into the side-call total shown in the table."""
    results = [
        _result("window", prompt_total=100),
        _result(
            "facts",
            prompt_total=100,
            side_prompt=65,
            side_completion=16,
            side_calls=2,
            failed_calls=1,
            failed_prompt=35,
            failed_completion=12,
        ),
    ]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "facts_failed" in lines
    assert "35" in lines and "12" in lines
    assert "не дало ничего использовать" in lines


def test_verdict_says_nothing_about_failures_when_none_happened():
    results = [_result("window", prompt_total=100), _result("facts", prompt_total=100)]

    lines = " ".join(strategy_bench.verdict_lines(results))

    assert "facts_failed" not in lines


def test_verdict_lines_always_include_a_usability_line():
    """SPEC §1's fourth axis (удобство) has no numeric column at all — it has
    to be said in words on every call, not only when some threshold trips."""
    results = [_result("window", prompt_total=100)]

    lines = strategy_bench.verdict_lines(results)

    assert lines[-1] == strategy_bench.usability_line()


def test_usability_line_names_each_strategys_own_commands():
    line = strategy_bench.usability_line()

    assert "/checkpoint" in line
    assert "/branch" in line
    assert "/switch" in line
    assert "/branches" in line
    assert "/facts backfill" in line
    assert "/fact set|del|unpin" in line


# --------------------------------------------------------------------------
# Answer excerpts
# --------------------------------------------------------------------------


def test_excerpt_returns_short_text_unchanged():
    assert strategy_bench.excerpt("короткий ответ") == "короткий ответ"


def test_excerpt_truncates_long_text_and_says_the_real_length():
    text = "а" * 500

    result = strategy_bench.excerpt(text, limit=50)

    assert len(result) < len(text)
    assert result.startswith("а" * 50)
    assert "обрезано" in result
    assert "500" in result


def test_print_answers_prints_every_strategys_own_answer(capsys):
    results = [
        _result("window", prompt_total=100, answer="ответ window"),
        _result("facts", prompt_total=100, answer="ответ facts"),
    ]

    strategy_bench._print_answers(results)

    out = capsys.readouterr().out
    assert "ответ window" in out
    assert "ответ facts" in out


def test_print_answers_excerpts_a_long_answer_and_says_so(capsys):
    long_answer = "деталь. " * 200
    results = [_result("branch", prompt_total=100, answer=long_answer)]

    strategy_bench._print_answers(results)

    out = capsys.readouterr().out
    assert "обрезано" in out
    assert str(len(long_answer)) in out


def test_print_fork_section_shows_both_branches_own_answers(capsys):
    strategy_bench._print_fork_section("Сейчас бюджет 520 тысяч рублей.", "У нас 300 тысяч рублей.")

    out = capsys.readouterr().out
    assert "520" in out
    assert "300" in out
    assert "изолированы" in out


# --------------------------------------------------------------------------
# Fork isolation
# --------------------------------------------------------------------------


def test_fork_isolation_ok_when_each_branch_holds_only_its_own_value():
    a_ok, b_ok = strategy_bench.fork_isolation_ok(
        "Сейчас бюджет 520 тысяч рублей.", "У нас 300 тысяч рублей."
    )

    assert (a_ok, b_ok) == (True, True)


def test_fork_isolation_catches_a_leak_into_one_branch():
    a_ok, b_ok = strategy_bench.fork_isolation_ok(
        "Бюджет 520, а раньше в другой ветке говорили про 300.", "У нас 300 тысяч рублей."
    )

    assert a_ok is False
    assert b_ok is True


def test_fork_isolation_report_names_which_side_leaked():
    line = strategy_bench.fork_isolation_report(False, True)

    assert strategy_bench.FORK_BUDGET_A in line
    assert "протекли" in line


def test_fork_isolation_report_confirms_isolation():
    line = strategy_bench.fork_isolation_report(True, True)

    assert "изолированы" in line
    assert strategy_bench.FORK_BUDGET_A in line and strategy_bench.FORK_BUDGET_B in line


# --------------------------------------------------------------------------
# run_scenario / run_turn (bench_core, exercised through a fake agent — no network)
# --------------------------------------------------------------------------


class _FakeAgent:
    """Stand-in for Agent: hands back canned replies, records what it saw."""

    def __init__(self, replies: list[AgentReply]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def ask(
        self,
        question,
        history,
        *,
        summary=None,
        facts=None,
        facts_pinned=(),
        facts_upto=0,
        on_chunk=None,
    ):
        self.calls.append(
            {
                "question": question,
                "history": list(history),
                "summary": summary,
                "facts": dict(facts or {}),
                "facts_upto": facts_upto,
            }
        )
        return self._replies.pop(0)


def _reply_with_history(
    question_marker: str, *, facts: dict[str, str] | None = None, facts_upto: int = 0
):
    return AgentReply(
        text=f"ответ на {question_marker}",
        history=[
            {"role": "user", "content": question_marker},
            {"role": "assistant", "content": f"ответ на {question_marker}"},
        ],
        result=_call_result(f"ответ на {question_marker}"),
        summary=None,
        facts=facts,
        facts_upto=facts_upto,
    )


def test_run_scenario_threads_history_and_facts_between_turns():
    """The engine must carry state OUTSIDE the agent (week_02/cli.py._turn()'s
    own contract) — a bug here would silently replay turn 1's history/facts
    into every later ask() call."""
    agent = _FakeAgent(
        [
            _reply_with_history("q1", facts={"цель.название": "кофейня"}, facts_upto=1),
            _reply_with_history("q2", facts={"цель.название": "кофейня"}, facts_upto=2),
        ]
    )

    replies, state = bench_core.run_scenario(agent, ["q1", "q2"])

    assert len(replies) == 2
    assert agent.calls[0]["history"] == []
    assert agent.calls[1]["history"] == replies[0].history
    assert agent.calls[1]["facts"] == {"цель.название": "кофейня"}
    assert agent.calls[1]["facts_upto"] == 1
    assert state.facts == {"цель.название": "кофейня"}
    assert state.facts_upto == 2


def test_scenario_state_clone_is_independent():
    state = bench_core.ScenarioState(
        history=[{"role": "user", "content": "x"}], facts={"цель.название": "a"}, facts_upto=1
    )

    clone = state.clone()
    clone.history.append({"role": "assistant", "content": "y"})
    clone.facts["цель.название"] = "b"

    assert state.history == [{"role": "user", "content": "x"}]
    assert state.facts == {"цель.название": "a"}


def test_run_scenario_can_continue_from_a_cloned_checkpoint():
    """The fork-isolation section's own trick: run a prefix, clone the
    resulting state, continue each clone independently (SPEC §13.3)."""
    agent = _FakeAgent([_reply_with_history("head")])
    _replies, checkpoint = bench_core.run_scenario(agent, ["head"])

    branch_a = checkpoint.clone()
    agent._replies.append(_reply_with_history("a1"))
    reply_a = bench_core.run_turn(agent, branch_a, "a1")

    branch_b = checkpoint.clone()
    agent._replies.append(_reply_with_history("b1"))
    bench_core.run_turn(agent, branch_b, "b1")

    assert reply_a.text == "ответ на a1"
    # Both branches started from the SAME checkpoint history, not from each
    # other's continuation.
    assert agent.calls[1]["history"] == checkpoint.history
    assert agent.calls[2]["history"] == checkpoint.history


# --------------------------------------------------------------------------
# _print_table — counting assertions, never bare substring membership
# --------------------------------------------------------------------------


def test_print_table_prints_each_header_and_row_exactly_once(capsys):
    """Counting assertions per CLAUDE.md's own "doubled header" trap — but a
    narrow console wraps "старое значение не вернулось" across three lines
    just as easily as a bug would double it, so the console is widened first;
    the point is to catch a duplicate row, not to fight line wrap."""
    results = [
        _result("window", prompt_total=100, all_details=False),
        _result("branch", prompt_total=900),
    ]
    args = argparse.Namespace(limit=4000, model="ministral-14b-latest")

    previous_width = strategy_bench.console.out.width
    strategy_bench.console.out.width = 200
    try:
        strategy_bench._print_table(results, args)
    finally:
        strategy_bench.console.out.width = previous_width

    out = capsys.readouterr().out
    assert out.count("старое значение не вернулось") == 1
    assert out.count(f"детали (n/{len(strategy_bench.DETAILS)})") == 1
    # "window": one table row + one mention in the missing-details line (it's
    # the only one missing anything) + one in its own verdict line (cheapest)
    # + one in the always-printed usability_line() = 4.
    # "branch": one table row + one in verdict line A ("branch —
    # lossless-базлайн: ...") + one in verdict line C ("branch — самый
    # дорогой прогон ...", details and price are now two independent lines,
    # see verdict_lines()) + THREE in usability_line() ("branch —", "/branch",
    # "/branches" — the last two both contain "branch" as a substring) = 6.
    # Counted against the real output rather than hand-derived, because that
    # substring overlap makes hand-counting unreliable.
    assert out.count("window") == 4
    assert out.count("branch") == 6


def test_print_table_shows_trim_column_header_and_values(capsys):
    """A strategy cut by the budget-trim safety net must show it as a
    number on screen, not just a low detail count indistinguishable from
    "this strategy simply forgets less" (2026-09-12 gap)."""
    trimmed = bench_core.TrimTotals(messages=4, tokens=900, turns_trimmed=2)
    results = [
        _result("window", prompt_total=100),
        _result("branch", prompt_total=1000, trim=trimmed),
    ]
    args = argparse.Namespace(limit=4000, model="ministral-14b-latest")

    previous_width = strategy_bench.console.out.width
    strategy_bench.console.out.width = 200
    try:
        strategy_bench._print_table(results, args)
    finally:
        strategy_bench.console.out.width = previous_width

    out = capsys.readouterr().out
    assert "обрезка safety-net" in out
    assert "0/0/0" in out  # window: never trimmed
    assert "2/4/900" in out  # branch: turns/messages/tokens


def test_print_table_shows_a_dash_for_unknown_trim_tokens(capsys):
    trimmed = bench_core.TrimTotals(
        messages=3, tokens=None, turns_trimmed=1, turns_trimmed_unknown_tokens=1
    )
    results = [_result("branch", prompt_total=1000, trim=trimmed)]
    args = argparse.Namespace(limit=4000, model="ministral-14b-latest")

    previous_width = strategy_bench.console.out.width
    strategy_bench.console.out.width = 200
    try:
        strategy_bench._print_table(results, args)
    finally:
        strategy_bench.console.out.width = previous_width

    out = capsys.readouterr().out
    assert "1/3/—" in out


def test_missing_details_line_names_the_labels_by_strategy():
    results = [
        _result("window", prompt_total=100, all_details=False),
        _result("branch", prompt_total=900),
    ]

    line = strategy_bench._missing_details_line(results)

    assert line is not None
    assert "window" in line
    assert strategy_bench.detail_label("budget") in line
    assert "branch" not in line


def test_missing_details_line_is_none_when_everyone_scored_full():
    results = [_result("window", prompt_total=100), _result("branch", prompt_total=900)]

    assert strategy_bench._missing_details_line(results) is None


# --------------------------------------------------------------------------
# score_answer against a natural paraphrase (2026-09-12: is a 5/6 result a
# checker gap or a genuine model omission? — coordinator's own follow-up)
# --------------------------------------------------------------------------


def test_score_answer_recognizes_one_natural_paraphrase_of_the_logo_detail():
    """Confirms the CURRENT forms catch at least one plausible paraphrase that
    is not the planted sentence verbatim — a floor, not a guarantee: a quick
    offline probe (not shipped as a test, since it would assert a guess about
    live model wording) found other equally plausible phrasings the same
    forms miss, e.g. "Логотип не разрабатываем, так как он уже готов." So a
    5/6 score on a live run cannot be resolved as checker-gap-vs-omission from
    this alone — it needs the actual saved answer text."""
    text = (
        "Логотип у вас уже есть в готовом виде, поэтому отдельно разрабатывать "
        "дизайн логотипа не будем."
    )

    score = strategy_bench.score_answer(text)

    assert "logo" in score.present


# --------------------------------------------------------------------------
# --dry-run / argparse guards — no network reached
# --------------------------------------------------------------------------


def test_dry_run_prints_scenario_and_checklist_without_touching_network(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("--dry-run ушёл в сеть")

    monkeypatch.setattr(strategy_bench, "list_models", boom)
    monkeypatch.setattr(strategy_bench.Config, "resolve", boom)

    assert strategy_bench.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert out.count("Плантированные детали") == 1
    assert "480" in out
    assert str(strategy_bench.DEFAULT_LIMIT) in out


def test_dry_run_shows_only_the_turns_that_will_actually_run(capsys):
    assert strategy_bench.main(["--dry-run", "--turns", str(strategy_bench.MIN_TURNS)]) == 0

    out = capsys.readouterr().out
    assert strategy_bench.SCENARIO[-1][:20] in out
    assert strategy_bench.SCENARIO[strategy_bench.HEAD_TURNS][:20] not in out


def test_too_few_turns_are_refused_before_anything_is_spent(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("проверка --turns пропущена")

    monkeypatch.setattr(strategy_bench.Config, "resolve", boom)

    with pytest.raises(SystemExit) as excinfo:
        strategy_bench.main(["--turns", "3"])

    assert excinfo.value.code == 2


def test_a_non_positive_limit_is_refused_before_anything_is_spent(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("проверка лимита пропущена")

    monkeypatch.setattr(strategy_bench.Config, "resolve", boom)

    with pytest.raises(SystemExit) as excinfo:
        strategy_bench.main(["--limit", "0"])

    assert excinfo.value.code == 2


def test_an_unknown_strategy_is_refused_before_anything_is_spent(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("проверка --strategies пропущена")

    monkeypatch.setattr(strategy_bench.Config, "resolve", boom)

    with pytest.raises(SystemExit) as excinfo:
        strategy_bench.main(["--strategies", "window,bogus"])

    assert excinfo.value.code == 2
