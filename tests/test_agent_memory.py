"""Focused Day 11 contracts at the Agent seam."""

from __future__ import annotations

import json

import pytest

from advent_core.agent import Agent
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.memory import MemorySnapshot, ShortTermMemory, StructuredMemory
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult


def _config(**kwargs) -> Config:
    return Config(
        api_key="key",
        model="mistral-small-latest",
        params=GenerationParams(context_strategy="memory", **kwargs),
    )


def _delta(evidence: str = "new preference") -> CallResult:
    return CallResult(
        text=json.dumps(
            {
                "working": {"set": [], "delete": []},
                "long_term": {
                    "set": [
                        {
                            "field": "preferences",
                            "key": "language",
                            "value": "Russian",
                            "evidence": evidence,
                        }
                    ],
                    "delete": [],
                },
            }
        )
    )


class _Calls:
    def __init__(self, *results: CallResult | Exception) -> None:
        self.results = list(results)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, config, messages, capabilities=None):
        self.calls.append(messages)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def stream(self, config, messages, on_chunk, capabilities=None):
        return self.complete(config, messages, capabilities)


def _agent(calls: _Calls, *, counter=None, context_limit=None, warnings=None) -> Agent:
    return Agent(
        _config(),
        complete=calls.complete,
        stream=calls.stream,
        memory_prompt="route memory",
        counter=counter,
        context_limit=context_limit,
        on_warning=(warnings if warnings is not None else lambda _text: None),
    )


def test_memory_extractor_receives_only_structured_layers_and_uncovered_segment():
    calls = _Calls(_delta(), CallResult(text="answer"))
    agent = _agent(calls)
    transcript = [
        {"role": "user", "content": "already seen user"},
        {"role": "assistant", "content": "already seen answer"},
        {"role": "user", "content": "fresh user"},
        {"role": "assistant", "content": "fresh answer"},
    ]
    snapshot = MemorySnapshot(
        short_term=ShortTermMemory(tuple(transcript)),
        working=StructuredMemory({"goal.primary": "ship"}),
    )

    agent.ask(
        "new preference",
        transcript,
        memory=snapshot,
        memory_upto=2,
        memory_history=transcript,
    )

    extractor_text = calls.calls[0][1]["content"]
    assert "goal.primary: ship" in extractor_text
    assert "fresh user" in extractor_text
    assert "new preference" in extractor_text
    assert "already seen user" not in extractor_text
    assert "already seen answer" not in extractor_text


def test_successful_memory_cursor_covers_persisted_user_and_assistant():
    calls = _Calls(_delta(), CallResult(text="answer"))
    agent = _agent(calls)
    transcript = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "ok"}]

    reply = agent.ask(
        "new preference",
        transcript,
        memory=MemorySnapshot(),
        memory_history=transcript,
    )

    assert reply.memory_upto == len(transcript) + 2
    assert reply.memory_update is not None
    assert reply.memory_update.memory_upto == len(transcript) + 2


def test_memory_request_failure_is_a_paid_request_failed_failure():
    calls = _Calls(AdventError("temporary"), CallResult(text="answer"))
    agent = _agent(calls)

    reply = agent.ask("remember this", [], memory=MemorySnapshot(), memory_history=[])

    assert reply.memory_failed is not None
    assert reply.memory_failed.reason == "request_failed"
    assert reply.memory_failed.call_result is not None
    assert reply.memory_failed.call_result.finish_reason == "error"
    assert len(calls.calls) == 2


def test_failed_main_call_keeps_pending_cursor_inside_transcript():
    calls = _Calls(_delta(), AdventError("main failed"))
    agent = _agent(calls)
    transcript = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "ok"}]

    with pytest.raises(AdventError):
        agent.ask(
            "new preference",
            transcript,
            memory=MemorySnapshot(),
            memory_history=transcript,
        )

    pending = agent.take_pending_memory()
    assert pending is not None
    assert pending.memory_upto <= len(transcript)


def test_protected_head_overflow_fails_before_any_paid_request():
    calls = _Calls(_delta(), CallResult(text="answer"))
    agent = _agent(calls, counter=_CharCounter(), context_limit=1100)
    snapshot = MemorySnapshot(
        long_term=StructuredMemory({"knowledge.large": "x" * 100}),
    )

    with pytest.raises(AdventError, match="long-term=.*working=.*summary="):
        agent.ask("question", [], memory=snapshot, memory_history=[])

    assert calls.calls == []


class _CharCounter:
    exact = True

    def count(self, messages):
        return sum(len(message["content"]) for message in messages)

    def calibrate(self, messages, prompt_tokens):
        pass


def test_memory_layer_cap_warns_without_deleting_entries():
    warnings: list[str] = []
    calls = _Calls(
        CallResult(
            text=json.dumps(
                {"working": {"set": [], "delete": []}, "long_term": {"set": [], "delete": []}}
            )
        ),
        CallResult(text="answer"),
    )
    agent = _agent(calls, counter=_CharCounter(), context_limit=4000, warnings=warnings.append)
    snapshot = MemorySnapshot(long_term=StructuredMemory({"knowledge.large": "x" * 1000}))

    reply = agent.ask("question", [], memory=snapshot, memory_history=[])

    assert any("long-term memory" in warning and "cap" in warning for warning in warnings)
    assert reply.memory is not None
    assert reply.memory.long_term.entries["knowledge.large"] == "x" * 1000
