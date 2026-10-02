"""Day 25 (SPEC-w05d25.md §9a.5): structural vs per-operation errors at the Agent parser."""

from __future__ import annotations

import json

import pytest

from advent_core.agent import Agent
from advent_core.config import Config
from advent_core.memory import MemorySnapshot
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult


def _agent() -> tuple[Agent, list[str]]:
    seen: list[str] = []
    agent = Agent(
        Config(
            api_key="key",
            model="mistral-small-latest",
            params=GenerationParams(context_strategy="memory"),
        ),
        complete=lambda *a, **k: CallResult(text="x"),
        stream=lambda *a, **k: CallResult(text="x"),
        memory_prompt="route memory",
        on_warning=seen.append,
    )
    return agent, seen


def _good(key: str = "os") -> dict:
    return {"field": "clarified", "key": key, "value": "Windows", "evidence": "я на Windows"}


def _run(raw: object):
    agent, warned = _agent()
    agent._memory_call = lambda snapshot, segment: CallResult(  # type: ignore[method-assign]
        text=raw if isinstance(raw, str) else json.dumps(raw)
    )
    out = agent._run_memory(MemorySnapshot(), [], "я на Windows", 0)
    return out, warned


def test_mixed_delta_applies_good_operation_and_rejects_every_malformed_element() -> None:
    raw = {
        "working": {
            "set": [
                _good(),
                "not an object",
                {"field": "clarified", "key": "nofields"},
                {"field": "clarified", "key": "empty", "value": "", "evidence": "я на Windows"},
                {"field": "bogus", "key": "k", "value": "v", "evidence": "я на Windows"},
                {"field": "clarified", "key": "bad", "value": "v", "evidence": "нет такого"},
            ],
            "delete": [5],
        },
        "long_term": {"set": [], "delete": []},
    }

    (snapshot, upto, update, failure), _ = _run(raw)

    assert failure is None
    assert update is not None
    assert snapshot.working.entries == {"clarified.os": "Windows"}
    assert upto == 0
    assert len(update.rejected) == 6
    assert len(update.applied) == 1


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"working": {"set": [], "delete": []}},
        {"working": [], "long_term": {"set": [], "delete": []}},
        {"working": {"set": {}, "delete": []}, "long_term": {"set": [], "delete": []}},
        {"working": {"set": [], "delete": "x"}, "long_term": {"set": [], "delete": []}},
        {"working": {"set": []}, "long_term": {"set": [], "delete": []}},
    ],
)
def test_structural_errors_reject_the_whole_delta(raw: object) -> None:
    (snapshot, upto, update, failure), warned = _run(raw)

    assert update is None
    assert failure is not None and failure.reason == "invalid"
    assert snapshot == MemorySnapshot()
    assert len(warned) == 1
