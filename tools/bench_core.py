"""Shared engine for the week 02 comparison harnesses (SPEC-w02d10.md §13).

Extracted from tools/compact_bench.py (day 09) rather than invented fresh:
day 09 is submitted and tagged, and its README command must keep printing the
exact tables it always has — so compact_bench.py is rewired ONTO this module
without changing what it prints, and tools/strategy_bench.py (day 10) is built
on the same pieces instead of a second copy of the same loop.

Three things live here: the scenario-running loop (drives Agent.ask() with
history/summary/facts held OUTSIDE the agent, exactly as week_02/cli.py._turn()
does — the agent itself keeps no memory of its own), the cumulative-column
table helper every per-turn growth table in this family uses, and the
argparse skeleton (--model/--limit/--dry-run) common to both scripts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from advent_core.agent import Agent, AgentReply
from advent_core.chat import Message
from advent_core.config import ConfigError
from advent_core.params import GenerationParams, ParamError


@dataclass(slots=True)
class ScenarioState:
    """Everything week_02/cli.py._turn() holds OUTSIDE the agent between turns.

    The agent keeps no memory of its own (advent_core/agent.py) — history,
    summary and sticky facts travel in AgentReply and the caller must carry
    them into the next ask(). A harness that skipped this (e.g. re-passing
    `summary=None` every turn) would silently measure "window" under every
    strategy label, no matter which one was actually selected.
    """

    history: list[Message] = field(default_factory=list)
    summary: str | None = None
    facts: dict[str, str] = field(default_factory=dict)
    facts_pinned: list[str] = field(default_factory=list)
    facts_upto: int = 0

    def clone(self) -> ScenarioState:
        """Independent copy: a fork test needs two continuations from one point
        that cannot see each other's later edits (SPEC-w02d10.md §13.3)."""
        return ScenarioState(
            history=list(self.history),
            summary=self.summary,
            facts=dict(self.facts),
            facts_pinned=list(self.facts_pinned),
            facts_upto=self.facts_upto,
        )


def run_turn(agent: Agent, state: ScenarioState, question: str) -> AgentReply:
    """One scenario turn: ask, then fold the reply back into `state` in place."""
    reply = agent.ask(
        question,
        state.history,
        summary=state.summary,
        facts=state.facts,
        facts_pinned=state.facts_pinned,
        facts_upto=state.facts_upto,
    )
    state.history = reply.history
    state.summary = reply.summary
    if reply.facts is not None:
        state.facts = reply.facts
        state.facts_upto = reply.facts_upto
    return reply


def run_scenario(
    agent: Agent,
    scenario: Sequence[str],
    *,
    state: ScenarioState | None = None,
    on_turn: Callable[[int, int], None] | None = None,
) -> tuple[list[AgentReply], ScenarioState]:
    """Runs `scenario` end to end from `state` (or a fresh one), in order.

    Returns every turn's reply plus the resulting state. A caller that needs
    to fork mid-scenario runs a prefix here, clones the returned state, and
    calls run_turn()/run_scenario() again on each clone — the checkpoint
    itself is just a ScenarioState, not a Session file (SPEC-w02d10.md §13.3:
    the harness measures the STRATEGY's isolation, not the branch-file
    mechanism, which belongs to advent_core/session.py and its own tests).
    """
    working = state.clone() if state is not None else ScenarioState()
    # `on_turn` exists because silence reads as a hang: four strategies over
    # eight turns is minutes of nothing, and on the w02d10 take that showed as
    # three minutes of a frozen screen — 97% of the video was static. Progress
    # belongs on stderr (the caller decides how to say it); the product of a
    # bench run is its table.
    replies = []
    total = len(scenario)
    for index, question in enumerate(scenario, start=1):
        if on_turn is not None:
            on_turn(index, total)
        replies.append(run_turn(agent, working, question))
    return replies, working


def cumulative_column(
    values: Sequence[int | None],
) -> tuple[list[tuple[str, str]], int | None, int]:
    """Table rows (value, running total) plus the grand total and gap count.

    A gap (missing usage) prints as a dash in BOTH columns of that turn rather
    than silently carrying the previous total forward — "unknown" must never
    turn into "zero" (CLAUDE.md).
    """
    rows: list[tuple[str, str]] = []
    cumulative = 0
    known = False
    missing = 0
    for value in values:
        if value is None:
            missing += 1
            rows.append(("—", "—"))
            continue
        cumulative += value
        known = True
        rows.append((str(value), str(cumulative)))
    return rows, (cumulative if known else None), missing


def side_call_totals(replies: Sequence[AgentReply]) -> tuple[int, int, int]:
    """(calls, prompt, completion) spent on side calls across a run.

    A "side call" is any Compaction, FactsUpdate or FactsFailure a turn
    produced — all three are separate, already-paid model calls whose usage
    never lands in the primary per-turn prompt/completion numbers
    (advent_core/agent.py's own split), so summing them here is the only place
    a harness sees their cost at all. FactsFailure counts too: a facts call
    that was billed and then failed to parse/validate is not free just
    because it produced nothing usable — dropping it here would make the
    day's totals silently understate what was actually spent (2026-09-12 live
    run: the arithmetic didn't add up until this was counted).
    """
    calls = prompt = completion = 0
    for reply in replies:
        for update in (reply.compaction, reply.facts_update, reply.facts_failed):
            if update is None:
                continue
            usage = update.result.usage
            calls += 1
            prompt += usage.prompt_tokens or 0
            completion += usage.completion_tokens or 0
    return calls, prompt, completion


def failed_side_calls(replies: Sequence[AgentReply]) -> tuple[int, int, int]:
    """(calls, prompt, completion) among side calls counted in
    side_call_totals() that produced NOTHING usable (AgentReply.facts_failed).

    Separate from side_call_totals() on purpose: that total already includes
    this cost (paid for either way), but "paid for" and "paid for and
    produced nothing" are different facts, and a caller that only prints the
    total lets the second one vanish silently.
    """
    calls = prompt = completion = 0
    for reply in replies:
        failure = reply.facts_failed
        if failure is None:
            continue
        usage = failure.result.usage
        calls += 1
        prompt += usage.prompt_tokens or 0
        completion += usage.completion_tokens or 0
    return calls, prompt, completion


@dataclass(slots=True)
class TrimTotals:
    """Sums of the per-request budget-trim safety net across a run
    (AgentReply.dropped/dropped_tokens) — the net that runs underneath ALL
    FOUR context strategies, not a strategy's own deliberate forgetting
    (AgentReply.window_dropped is that, and lives outside this dataclass).

    Added because the 2026-09-12 live run had no way to show this at all: a
    strategy gutted by the safety net printed a small token number and a low
    detail count with nothing distinguishing that from "this strategy simply
    forgets less" — the same class of silent-vanishing number as
    side_call_totals' facts_failed fix above.

    `tokens` mirrors cumulative_column's own rule: None only when EVERY
    trimmed turn's freed-token count is unknown; a turn that dropped messages
    with an unknown count doesn't erase turns that do have one. Zero turns
    trimmed means zero tokens freed — that's a known fact, not a gap.
    """

    messages: int = 0
    tokens: int | None = 0
    turns_trimmed: int = 0
    # Turns among turns_trimmed where dropped_tokens was None — surfaced
    # separately so a partial `tokens` total can be flagged as incomplete.
    turns_trimmed_unknown_tokens: int = 0


def trim_totals(replies: Sequence[AgentReply]) -> TrimTotals:
    """(messages dropped, tokens freed, turns trimmed) summed across a run."""
    messages = 0
    tokens = 0
    turns_trimmed = 0
    unknown_tokens = 0
    for reply in replies:
        messages += reply.dropped
        if not reply.dropped:
            continue
        turns_trimmed += 1
        if reply.dropped_tokens is None:
            unknown_tokens += 1
        else:
            tokens += reply.dropped_tokens
    if turns_trimmed == 0:
        tokens_total: int | None = 0
    elif unknown_tokens == turns_trimmed:
        tokens_total = None
    else:
        tokens_total = tokens
    return TrimTotals(
        messages=messages,
        tokens=tokens_total,
        turns_trimmed=turns_trimmed,
        turns_trimmed_unknown_tokens=unknown_tokens,
    )


def apply_param_overrides(base: GenerationParams, **overrides: object) -> GenerationParams:
    """Copy of `base` with each non-None override applied through params.set().

    Only advent_core/params.py knows a parameter's bounds (`keep_last >= 2`,
    the `context_strategy` choices, ...) — duplicating a rule here would drift
    out of sync with it the moment either side changes (generalizes day 09's
    own `compact_bench._mode_params`, CLAUDE.md's "the registry is the only
    place that knows it").
    """
    params = replace(base)
    try:
        for name, value in overrides.items():
            if value is not None:
                params.set(name, value)
    except ParamError as exc:
        raise ConfigError(str(exc)) from exc
    return params


def num(value: int | None) -> str:
    """The number, or "—" for unknown — never zero (shared table/footer convention)."""
    return "—" if value is None else str(value)


def build_parser(
    description: str,
    *,
    default_model: str,
    default_limit: int,
    limit_help: str,
    model_help: str = "модель для прогона",
) -> argparse.ArgumentParser:
    """Common skeleton: --model, --limit, --dry-run. Callers add their own flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=default_model, help=model_help)
    parser.add_argument("--limit", type=int, default=default_limit, help=limit_help)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="напечатать сценарий и настройки прогона, не ходить в сеть",
    )
    return parser


__all__ = [
    "ScenarioState",
    "TrimTotals",
    "apply_param_overrides",
    "build_parser",
    "cumulative_column",
    "failed_side_calls",
    "num",
    "run_scenario",
    "run_turn",
    "side_call_totals",
    "trim_totals",
]
