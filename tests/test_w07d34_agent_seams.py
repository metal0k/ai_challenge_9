"""Day 34 Agent seams: on_tool_call, max_tool_calls, per-round snapshots."""

from __future__ import annotations

from dataclasses import replace

from advent_core.agent import TOOL_LIMIT_TEXT, Agent, ToolCallLog
from advent_core.config import Config
from advent_core.mcp_client import ToolCallOutcome
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, RawToolCall, Usage

TOOL = {
    "type": "function",
    "function": {"name": "t", "description": "d", "parameters": {"type": "object"}},
}


def _round(*ids: str, arguments: str = "{}") -> CallResult:
    return CallResult(
        text="",
        model_requested="m",
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        tool_calls=tuple(RawToolCall(id=i, name="t", arguments=arguments) for i in ids),
    )


def _final() -> CallResult:
    return CallResult(
        text="готово",
        model_requested="m",
        finish_reason="stop",
        usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
    )


class _Model:
    """Scripted results; sent_messages aliases the live list on purpose."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results)
        self.n = 0
        self.tools_seen: list = []

    def complete(self, config, messages, capabilities=None, *, tools=None, tool_choice=None):
        self.tools_seen.append(tools)
        result = self.results[min(self.n, len(self.results) - 1)]
        self.n += 1
        return replace(result, sent_messages=messages)

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("tools go through complete()")


class _Counter:
    exact = True

    def __init__(self) -> None:
        self.calibrated = 0

    def count(self, messages):
        return 1

    def calibrate(self, sent, prompt_tokens):
        self.calibrated += 1


def _agent(model: _Model, warnings: list[str], **kwargs) -> Agent:
    executed: list[str] = []

    def call_tool(name, arguments):
        executed.append(name)
        return ToolCallOutcome(text="результат", is_error=False)

    agent = Agent(
        Config(api_key="k", model="m", params=GenerationParams()),
        complete=model.complete,
        stream=model.stream,
        tools=[TOOL],
        call_tool=call_tool,
        mcp_enabled=True,
        on_warning=warnings.append,
        **kwargs,
    )
    agent.executed = executed  # type: ignore[attr-defined]
    return agent


def test_defaults_change_nothing():
    agent = _agent(_Model(_round("a"), _final()), [])
    reply = agent.ask("вопрос", [])
    assert reply.tool_limit_hit is None
    assert [log.is_error for log in reply.tool_calls] == [False]


def test_on_tool_call_fires_for_every_call_including_bad_arguments():
    seen: list[ToolCallLog] = []
    _agent(_Model(_round("a", "b"), _final()), [], on_tool_call=seen.append).ask("вопрос", [])
    assert [log.result_text for log in seen] == ["результат", "результат"]

    seen.clear()
    bad = _Model(_round("a", arguments="[1]"), _final())
    _agent(bad, [], on_tool_call=seen.append).ask("вопрос", [])
    assert len(seen) == 1
    assert seen[0].is_error is True
    assert "аргументы вызова не разобрать" in seen[0].result_text


def test_callback_exception_does_not_kill_the_turn():
    def boom(log):
        raise RuntimeError("телеметрия упала")

    warnings: list[str] = []
    agent = _agent(_Model(_round("a"), _final()), warnings, on_tool_call=boom)
    reply = agent.ask("вопрос", [])
    assert reply.text == "готово"
    assert len(reply.tool_calls) == 1
    assert any("on_tool_call упал: телеметрия упала" in w for w in warnings)


def test_max_tool_calls_skips_the_rest_and_ends_like_round_exhaustion():
    seen: list[ToolCallLog] = []
    warnings: list[str] = []
    model = _Model(_round("a", "b", "c"), _final())
    agent = _agent(model, warnings, max_tool_calls=2, on_tool_call=seen.append)
    reply = agent.ask("вопрос", [])

    assert agent.executed == ["t", "t"], "third call must not be executed"
    assert reply.tool_limit_hit == "calls"
    assert model.n == 1, "no further paid round after the limit"
    assert [log.is_error for log in reply.tool_calls] == [False, False, True]
    assert reply.tool_calls[2].result_text == TOOL_LIMIT_TEXT
    assert seen == list(reply.tool_calls)
    assert any("лимит вызовов инструментов (2) исчерпан" in w for w in warnings)
    # the limited round is parked AND is the reply's result (same overlap as rounds)
    assert reply.tool_round_results == (reply.result,)


def test_limit_counts_across_rounds():
    model = _Model(_round("a"), _round("b"), _final())
    agent = _agent(model, [], max_tool_calls=1)
    reply = agent.ask("вопрос", [])
    assert agent.executed == ["t"]
    assert reply.tool_limit_hit == "calls"
    assert [log.result_text for log in reply.tool_calls] == ["результат", TOOL_LIMIT_TEXT]


def test_round_exhaustion_reports_rounds():
    agent = _agent(_Model(_round("a")), [], max_tool_rounds=2)
    assert agent.ask("вопрос", []).tool_limit_hit == "rounds"


def test_flag_resets_between_asks():
    agent = _agent(_Model(_round("a", "b"), _final(), _final()), [], max_tool_calls=1)
    assert agent.ask("один", []).tool_limit_hit == "calls"
    agent.max_tool_calls = None
    assert agent.ask("два", []).tool_limit_hit is None


def test_sent_messages_are_per_round_snapshots_and_no_calibration_with_tools():
    counter = _Counter()
    warnings: list[str] = []
    model = _Model(_round("a"), _round("b"), _final())
    agent = _agent(model, warnings, counter=counter)
    reply = agent.ask("вопрос", [])

    paid = [*reply.tool_round_results, reply.result]
    assert len(paid) == 3
    assert not any(m["role"] == "tool" for m in paid[0].sent_messages)
    assert sum(m["role"] == "tool" for m in paid[1].sent_messages) == 1
    assert sum(m["role"] == "tool" for m in paid[2].sent_messages) == 2
    assert counter.calibrated == 0
    assert any("сверка счёта токенов пропущена" in w for w in warnings)
    assert all(tools == [TOOL] for tools in model.tools_seen)
