"""Scenario state and the one-turn carry shared by the week 02 benches and `adventrag chat-eval`.

Lives in advent_core (not tools/) because installed entry points do not package `tools`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from advent_core.agent import Agent, AgentReply
from advent_core.chat import Message
from advent_core.memory import (
    MemoryFailure,
    MemorySnapshot,
    MemoryUpdate,
    ShortTermMemory,
)


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
    # Day 25: the memory strategy's carry (None = not tracked, the pre-day-25 shape).
    # `transcript` is the FULL record the extractor reads (Session.history() in the REPL),
    # `history` the trimmed working copy — they diverge as soon as trim fires.
    memory: MemorySnapshot | None = None
    memory_upto: int = 0
    transcript: list[Message] = field(default_factory=list)

    def clone(self) -> ScenarioState:
        """Independent copy: a fork test needs two continuations from one point
        that cannot see each other's later edits (SPEC-w02d10.md §13.3)."""
        return ScenarioState(
            history=list(self.history),
            summary=self.summary,
            facts=dict(self.facts),
            facts_pinned=list(self.facts_pinned),
            facts_upto=self.facts_upto,
            memory=self.memory,
            memory_upto=self.memory_upto,
            transcript=list(self.transcript),
        )


def _memory_kwargs(state: ScenarioState) -> dict:
    """The memory arguments of ask(), as week_02/cli.py._ask() passes them; {} when untracked."""
    if state.memory is None:
        return {}
    snapshot = MemorySnapshot(
        ShortTermMemory(tuple(state.history), 0, len(state.transcript)),
        state.memory.working,
        state.memory.long_term,
    )
    return {
        "memory": snapshot,
        "memory_upto": state.memory_upto,
        "memory_history": state.transcript,
    }


def run_turn(agent: Agent, state: ScenarioState, question: str) -> AgentReply:
    """One scenario turn: ask, then fold the reply back into `state` in place."""
    reply = agent.ask(
        question,
        state.history,
        summary=state.summary,
        facts=state.facts,
        facts_pinned=state.facts_pinned,
        facts_upto=state.facts_upto,
        **_memory_kwargs(state),
    )
    state.history = reply.history
    state.summary = reply.summary
    if reply.facts is not None:
        state.facts = reply.facts
        state.facts_upto = reply.facts_upto
    if state.memory is not None:
        # Like week_02/cli.py._turn() + _remember(): the exchange joins the full
        # transcript, and a successful extraction is then covered up to its end.
        if reply.memory is not None:
            state.memory = reply.memory
            state.memory_upto = reply.memory_upto
        state.transcript.append({"role": "user", "content": question})
        last = reply.history[-1] if reply.history else None
        if last is not None and last["role"] == "assistant" and last["content"]:
            state.transcript.append({"role": "assistant", "content": last["content"]})
        if reply.memory_update is not None:
            state.memory_upto = len(state.transcript)
    return reply


def salvage_pending_memory(
    agent: Agent, state: ScenarioState
) -> tuple[MemoryUpdate | None, MemoryFailure | None]:
    """Collects a paid extractor call whose turn then failed (like week_02/cli.py._salvage_memory).

    The failed turn's user message was never recorded, so the cursor is clamped to the
    transcript: the update's own cursor counts that virtual message.
    """
    update = agent.take_pending_memory()
    failure = agent.take_pending_memory_failed()
    if update is not None and state.memory is not None:
        state.memory = update.snapshot
        state.memory_upto = min(update.memory_upto, len(state.transcript))
    return update, failure
