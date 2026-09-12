"""tools/bench_core.py: side-call arithmetic (SPEC-w02d10.md §9, §13).

Focused on side_call_totals()/failed_side_calls() rather than the whole
engine — run_scenario()/ScenarioState are already exercised end to end by
tests/test_strategy_bench.py. Expected numbers are literal, not re-derived
from the function under test (CLAUDE.md: a test whose expected value comes
from the same source as the code under test cannot go red).
"""

from __future__ import annotations

from advent_core.agent import AgentReply, Compaction, FactsFailure, FactsUpdate
from advent_core.facts import DeltaResult
from advent_core.telemetry import CallResult, Usage
from tools import bench_core


def _usage(prompt: int | None, completion: int | None) -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion)


def _call_result(prompt: int | None, completion: int | None) -> CallResult:
    return CallResult(usage=_usage(prompt, completion))


def _compaction(prompt: int, completion: int) -> Compaction:
    return Compaction(summary="сжато", covered=2, result=_call_result(prompt, completion))


def _facts_update(prompt: int, completion: int) -> FactsUpdate:
    delta = DeltaResult(facts={}, added=(), updated=(), removed=(), blocked=(), rejected=())
    return FactsUpdate(facts={}, delta=delta, covered=1, result=_call_result(prompt, completion))


def _facts_failure(prompt: int, completion: int, reason: str = "invalid") -> FactsFailure:
    return FactsFailure(result=_call_result(prompt, completion), reason=reason)


def _reply(
    *,
    prompt: int | None = None,
    completion: int | None = None,
    compaction: Compaction | None = None,
    facts_update: FactsUpdate | None = None,
    facts_failed: FactsFailure | None = None,
    dropped: int = 0,
    dropped_tokens: int | None = None,
) -> AgentReply:
    return AgentReply(
        text="",
        history=[],
        result=_call_result(prompt, completion),
        compaction=compaction,
        facts_update=facts_update,
        facts_failed=facts_failed,
        dropped=dropped,
        dropped_tokens=dropped_tokens,
    )


# --------------------------------------------------------------------------
# side_call_totals
# --------------------------------------------------------------------------


def test_side_call_totals_sums_compaction_and_facts_update():
    replies = [
        _reply(prompt=100, completion=10, compaction=_compaction(50, 20)),
        _reply(prompt=100, completion=10, facts_update=_facts_update(30, 4)),
    ]

    calls, prompt, completion = bench_core.side_call_totals(replies)

    assert (calls, prompt, completion) == (2, 80, 24)


def test_side_call_totals_ignores_turns_with_no_side_call():
    replies = [_reply(prompt=100, completion=10)]

    assert bench_core.side_call_totals(replies) == (0, 0, 0)


def test_side_call_totals_counts_a_failed_facts_call_too():
    """A facts call that was billed and then failed to parse still cost real
    tokens (SPEC-w02d10.md §5.3) — dropping it here is exactly the bug a live
    run hit on 2026-09-12: the totals didn't add up."""
    replies = [
        _reply(prompt=100, completion=10, facts_failed=_facts_failure(35, 12)),
    ]

    calls, prompt, completion = bench_core.side_call_totals(replies)

    assert (calls, prompt, completion) == (1, 35, 12)


def test_side_call_totals_adds_a_failure_on_top_of_a_successful_side_call():
    """One turn's compaction succeeding and another turn's facts call failing
    are two separate paid calls — both must land in the same total."""
    replies = [
        _reply(prompt=100, completion=10, compaction=_compaction(50, 20)),
        _reply(prompt=100, completion=10, facts_failed=_facts_failure(35, 12)),
    ]

    calls, prompt, completion = bench_core.side_call_totals(replies)

    assert (calls, prompt, completion) == (2, 85, 32)


# --------------------------------------------------------------------------
# failed_side_calls
# --------------------------------------------------------------------------


def test_failed_side_calls_reports_only_facts_failed():
    replies = [
        _reply(prompt=100, completion=10, compaction=_compaction(50, 20)),
        _reply(prompt=100, completion=10, facts_update=_facts_update(30, 4)),
        _reply(prompt=100, completion=10, facts_failed=_facts_failure(35, 12)),
    ]

    calls, prompt, completion = bench_core.failed_side_calls(replies)

    assert (calls, prompt, completion) == (1, 35, 12)


def test_failed_side_calls_is_zero_when_nothing_failed():
    replies = [
        _reply(prompt=100, completion=10, compaction=_compaction(50, 20)),
        _reply(prompt=100, completion=10, facts_update=_facts_update(30, 4)),
    ]

    assert bench_core.failed_side_calls(replies) == (0, 0, 0)


def test_failed_side_calls_sums_across_several_failures():
    replies = [
        _reply(prompt=100, completion=10, facts_failed=_facts_failure(35, 12, "invalid")),
        _reply(prompt=100, completion=10, facts_failed=_facts_failure(20, 8, "truncated")),
    ]

    calls, prompt, completion = bench_core.failed_side_calls(replies)

    assert (calls, prompt, completion) == (2, 55, 20)


# --------------------------------------------------------------------------
# trim_totals (SPEC-w02d10.md §13, 2026-09-12 live-run gap: the budget-trim
# safety net had no visible number at all)
# --------------------------------------------------------------------------


def test_trim_totals_is_all_zero_when_nothing_was_ever_dropped():
    replies = [_reply(prompt=100, completion=10), _reply(prompt=100, completion=10)]

    totals = bench_core.trim_totals(replies)

    assert (totals.messages, totals.tokens, totals.turns_trimmed) == (0, 0, 0)
    assert totals.turns_trimmed_unknown_tokens == 0


def test_trim_totals_sums_messages_and_tokens_across_several_trimmed_turns():
    replies = [
        _reply(prompt=100, completion=10, dropped=2, dropped_tokens=300),
        _reply(prompt=100, completion=10),  # untouched turn in between
        _reply(prompt=100, completion=10, dropped=4, dropped_tokens=600),
    ]

    totals = bench_core.trim_totals(replies)

    assert (totals.messages, totals.tokens, totals.turns_trimmed) == (6, 900, 2)
    assert totals.turns_trimmed_unknown_tokens == 0


def test_trim_totals_is_none_when_every_trimmed_turn_has_no_token_count():
    """The char-based fallback trim path (no counter/budget) reports messages
    dropped but never a token figure — None must survive to the total, not
    silently become zero (CLAUDE.md: unknown never becomes zero)."""
    replies = [_reply(prompt=100, completion=10, dropped=3, dropped_tokens=None)]

    totals = bench_core.trim_totals(replies)

    assert totals.messages == 3
    assert totals.tokens is None
    assert totals.turns_trimmed == 1
    assert totals.turns_trimmed_unknown_tokens == 1


def test_trim_totals_keeps_the_partial_known_sum_when_only_some_turns_are_unknown():
    """One trimmed turn with a known token count and one without: the known
    figure must not be thrown away just because a later turn couldn't be
    measured — same "partial total plus a missing count" shape as
    cumulative_column's own gap handling."""
    replies = [
        _reply(prompt=100, completion=10, dropped=2, dropped_tokens=300),
        _reply(prompt=100, completion=10, dropped=1, dropped_tokens=None),
    ]

    totals = bench_core.trim_totals(replies)

    assert totals.messages == 3
    assert totals.tokens == 300
    assert totals.turns_trimmed == 2
    assert totals.turns_trimmed_unknown_tokens == 1


def test_run_scenario_reports_progress_once_per_turn_in_order():
    """A silent multi-minute run reads as a hang — and on camera it WAS three
    minutes of a frozen screen. The hook fires before each turn, so the last
    call is (n, n) and not one turn short."""
    from tests.test_strategy_bench import _FakeAgent, _reply_with_history

    agent = _FakeAgent([_reply_with_history("q1"), _reply_with_history("q2")])
    seen: list[tuple[int, int]] = []

    bench_core.run_scenario(
        agent, ["q1", "q2"], on_turn=lambda turn, total: seen.append((turn, total))
    )

    assert seen == [(1, 2), (2, 2)]


def test_run_scenario_without_a_progress_hook_still_runs():
    from tests.test_strategy_bench import _FakeAgent, _reply_with_history

    agent = _FakeAgent([_reply_with_history("q1")])

    replies, _state = bench_core.run_scenario(agent, ["q1"])

    assert len(replies) == 1
