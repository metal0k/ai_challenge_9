"""Day 20: registry, router, facade servers and the agent's round limit.

Nothing here touches the network, npx or Mistral: `connect_and_list` and
`call_tool_once` are replaced by recorders.
"""

from __future__ import annotations

import json
import re

import pytest

from advent_core import mcp_router
from advent_core.errors import MCPError
from advent_core.mcp_client import ListResult, ToolCallOutcome, ToolInfo

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def _listing(*names: str) -> ListResult:
    return ListResult(
        server_name="s",
        server_version="1",
        protocol_version="p",
        instructions=None,
        tools=[ToolInfo(name, f"desc {name}", input_schema=SCHEMA) for name in names],
    )


def _spec(name: str, allow=None, timeout: float = 5.0) -> mcp_router.ServerSpec:
    return mcp_router.ServerSpec(name=name, command=[f"cmd-{name}"], allow=allow, timeout=timeout)


class _Backend:
    """Stands in for the transport: per-server tool lists, failures and a call log."""

    def __init__(self, monkeypatch, tools: dict[str, list[str]], down: set[str] = frozenset()):
        self.tools = tools
        self.down = set(down)
        self.calls: list[tuple[str, str, dict]] = []
        self.quiet_flags: list[bool] = []
        self.list_kwargs: list[dict] = []
        self.call_kwargs: list[dict] = []
        monkeypatch.setattr(mcp_router.mcp_client, "connect_and_list", self._list)
        monkeypatch.setattr(mcp_router.mcp_client, "call_tool_once", self._call)

    def _server(self, command):
        return command[0].removeprefix("cmd-")

    def _list(self, command, *, timeout, raw, sink, cwd=None, quiet=False):
        server = self._server(command)
        self.quiet_flags.append(quiet)
        self.list_kwargs.append({"cwd": cwd, "timeout": timeout})
        if server in self.down:
            raise MCPError(f"{server} не отвечает")
        return _listing(*self.tools[server])

    def _call(self, command, name, arguments, *, timeout, cwd=None, quiet=False):
        server = self._server(command)
        self.quiet_flags.append(quiet)
        self.calls.append((server, name, arguments))
        self.call_kwargs.append({"cwd": cwd, "timeout": timeout})
        if server in self.down:
            raise MCPError(f"{server} упал")
        return ToolCallOutcome(text=f"{server}:{name}", is_error=False)


# --- registry ----------------------------------------------------------------


def _write_registry(tmp_path, servers):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


def test_registry_expands_repo_and_python_placeholders(tmp_path):
    path = _write_registry(
        tmp_path,
        [
            {
                "name": "a",
                "command": "{python}",
                "args": ["-m", "x", "{repo}/logs"],
                "cwd": "{repo}",
                "allow": ["t"],
                "timeout": 42,
                "ensure_dirs": ["{repo}/logs"],
            }
        ],
    )
    (spec,) = mcp_router.load_registry(path, repo_root=tmp_path)
    assert spec.command[0] != "{python}"
    assert spec.command[1:] == ["-m", "x", f"{tmp_path.as_posix()}/logs"]
    assert spec.cwd == tmp_path
    assert spec.allow == ("t",)
    assert spec.timeout == 42
    assert spec.ensure_dirs == (tmp_path / "logs",)


def test_registry_uses_npx_cmd_on_windows_only(tmp_path, monkeypatch):
    path = _write_registry(tmp_path, [{"name": "fs", "command": "npx", "args": []}])
    monkeypatch.setattr(mcp_router.os, "name", "nt")
    assert mcp_router.load_registry(path)[0].command[0] == "npx.cmd"
    monkeypatch.setattr(mcp_router.os, "name", "posix")
    assert mcp_router.load_registry(path)[0].command[0] == "npx"


@pytest.mark.parametrize(
    "servers",
    [[], [{"name": "a__b", "command": "x"}], [{"name": "a", "command": "x"}] * 2, [{"name": "a"}]],
    ids=("empty", "separator-in-name", "duplicate", "no-command"),
)
def test_a_malformed_registry_is_an_mcp_error_not_a_traceback(tmp_path, servers):
    with pytest.raises(MCPError):
        mcp_router.load_registry(_write_registry(tmp_path, servers))


def test_missing_registry_file_is_an_mcp_error(tmp_path):
    with pytest.raises(MCPError):
        mcp_router.load_registry(tmp_path / "nope.json")


def test_the_shipped_registry_pins_the_fs_package_and_limits_its_tools():
    specs = {s.name: s for s in mcp_router.load_registry()}
    assert set(specs) == {"repo", "pipeline", "fs"}
    fs_args = " ".join(specs["fs"].command)
    assert "@modelcontextprotocol/server-filesystem@" in fs_args  # version pinned
    assert specs["fs"].allow == (
        "list_allowed_directories",
        "write_file",
        "read_text_file",
        "list_directory",
    )
    assert specs["pipeline"].allow == ("summarize_text",)


# --- naming ------------------------------------------------------------------


def test_exposed_names_are_prefixed_and_fit_the_function_name_alphabet():
    assert mcp_router.exposed_name("fs", "write_file", set()) == "fs__write_file"
    name = mcp_router.exposed_name("s", "we ird.tool", set())
    assert mcp_router.NAME_BAD.search(name) is None


def test_a_too_long_name_is_shortened_deterministically_and_stays_unique():
    long_a = "a" * 80
    first = mcp_router.exposed_name("s", long_a, set())
    again = mcp_router.exposed_name("s", long_a, set())
    other = mcp_router.exposed_name("s", long_a + "b", set())
    assert len(first) <= mcp_router.NAME_MAX
    assert first == again
    assert first != other


def test_a_collision_between_sanitized_names_gets_a_hash_suffix(monkeypatch):
    # "a.b" and "a_b" sanitize to the same function name; both must stay callable.
    backend = _Backend(monkeypatch, {"s": ["a.b", "a_b"]})
    router = mcp_router.Router([_spec("s")])
    report = router.connect()
    names = [tool["function"]["name"] for tool in report.tools]
    assert len(set(names)) == 2
    router.call(names[0], {})
    router.call(names[1], {})
    assert [call[1] for call in backend.calls] == ["a.b", "a_b"]


def test_a_tool_whose_own_name_contains_the_separator_routes_to_the_right_server(monkeypatch):
    backend = _Backend(monkeypatch, {"s": ["a__b"]})
    router = mcp_router.Router([_spec("s")])
    router.connect()
    router.call("s__a__b", {})
    assert backend.calls == [("s", "a__b", {})]


# --- connect -----------------------------------------------------------------


def test_connect_merges_all_servers_and_filters_by_allow(monkeypatch):
    _Backend(monkeypatch, {"a": ["t1", "t2"], "b": ["w", "r", "d"]})
    router = mcp_router.Router([_spec("a"), _spec("b", allow=("w", "r"))])
    report = router.connect()
    assert [t["function"]["name"] for t in report.tools] == ["a__t1", "a__t2", "b__w", "b__r"]
    assert report.server_tools == {"a": ["t1", "t2"], "b": ["w", "r"]}
    assert report.server_offered["b"] == ["w", "r", "d"]
    assert report.warnings == []
    assert report.tools[0]["function"]["parameters"] == SCHEMA


def test_an_allowed_tool_the_server_does_not_have_is_a_warning(monkeypatch):
    _Backend(monkeypatch, {"a": ["t1"]})
    report = mcp_router.Router([_spec("a", allow=("t1", "ghost"))]).connect()
    assert [r.tool for r in report.routes] == ["t1"]
    assert any("ghost" in warning for warning in report.warnings)


def test_partial_connect_warns_with_the_reason_and_keeps_the_others(monkeypatch):
    _Backend(monkeypatch, {"a": ["t1"], "b": ["t2"]}, down={"b"})
    router = mcp_router.Router([_spec("a"), _spec("b")])
    report = router.connect()
    assert [t["function"]["name"] for t in report.tools] == ["a__t1"]
    assert report.failed == ["b"]
    assert "сервер b недоступен: b не отвечает" in report.warnings
    assert router.failed == ["b"]


def test_registry_servers_are_spawned_with_child_stderr_suppressed(monkeypatch):
    backend = _Backend(monkeypatch, {"a": ["t1"]})
    router = mcp_router.Router([_spec("a")])
    router.connect()
    router.call("a__t1", {})
    assert backend.quiet_flags == [True, True]


def test_cwd_and_timeout_reach_both_connect_and_call(monkeypatch, tmp_path):
    backend = _Backend(monkeypatch, {"a": ["t1"]})
    spec = mcp_router.ServerSpec(name="a", command=["cmd-a"], cwd=tmp_path, timeout=42.0)
    router = mcp_router.Router([spec])
    router.connect()
    router.call("a__t1", {})
    assert backend.list_kwargs == [{"cwd": tmp_path, "timeout": 42.0}]
    assert backend.call_kwargs == [{"cwd": tmp_path, "timeout": 42.0}]


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "a", "command": "c", "allow": "summarize_text"},
        {"name": "a", "command": "c", "args": "-m x"},
        {"name": "a", "command": "c", "ensure_dirs": "logs"},
        {"name": "a", "command": "c", "allow": [1]},
    ],
)
def test_registry_list_fields_must_be_lists_of_strings(tmp_path, entry):
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(MCPError):
        mcp_router.load_registry(path)


# --- dispatch ----------------------------------------------------------------


def test_calls_reach_the_owning_server_in_the_order_they_were_made(monkeypatch):
    backend = _Backend(monkeypatch, {"repo": ["git_log"], "pipe": ["sum"], "fs": ["write"]})
    router = mcp_router.Router([_spec("repo"), _spec("pipe"), _spec("fs")])
    router.connect()
    for name, args in (
        ("repo__git_log", {"n": 3}),
        ("pipe__sum", {"text": "t"}),
        ("fs__write", {"path": "p"}),
    ):
        router.call(name, args)
    assert backend.calls == [
        ("repo", "git_log", {"n": 3}),
        ("pipe", "sum", {"text": "t"}),
        ("fs", "write", {"path": "p"}),
    ]
    records = router.take_records()
    assert [(r.server, r.tool) for r in records] == [
        ("repo", "git_log"),
        ("pipe", "sum"),
        ("fs", "write"),
    ]
    assert router.take_records() == []  # handed over once


@pytest.mark.parametrize("name", ["git_log", "nope__x", "repo__ghost", ""])
def test_an_unknown_tool_is_an_error_outcome_listing_valid_names(monkeypatch, name):
    backend = _Backend(monkeypatch, {"repo": ["git_log"]})
    router = mcp_router.Router([_spec("repo")])
    router.connect()
    outcome = router.call(name, {})
    assert outcome.is_error
    assert "repo__git_log" in outcome.text
    assert backend.calls == []  # nothing was dispatched


def test_a_disallowed_tool_is_unknown_even_though_the_server_has_it(monkeypatch):
    backend = _Backend(monkeypatch, {"fs": ["write", "delete"]})
    router = mcp_router.Router([_spec("fs", allow=("write",))])
    router.connect()
    assert router.call("fs__delete", {}).is_error
    assert backend.calls == []


def test_call_never_raises_even_for_non_mcp_exceptions(monkeypatch):
    _Backend(monkeypatch, {"a": ["t"]})
    router = mcp_router.Router([_spec("a")])
    router.connect()

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mcp_router.mcp_client, "call_tool_once", boom)
    outcome = router.call("a__t", {})
    assert outcome.is_error
    assert "kaboom" in outcome.text


def test_a_down_server_becomes_error_text_for_the_model(monkeypatch):
    backend = _Backend(monkeypatch, {"a": ["t"]})
    router = mcp_router.Router([_spec("a")])
    router.connect()
    backend.down.add("a")
    outcome = router.call("a__t", {})
    assert outcome.is_error
    assert "упал" in outcome.text


def test_trace_and_journal_see_every_call_and_cannot_break_it(monkeypatch):
    _Backend(monkeypatch, {"a": ["t"]})
    lines: list[str] = []
    rows: list[mcp_router.RouteRecord] = []
    rounds = iter([1, 2])
    router = mcp_router.Router(
        [_spec("a")],
        trace=lines.append,
        journal=rows.append,
        current_round=lambda: next(rounds),
    )
    router.connect()
    router.call("a__t", {"x": 1})
    router.call("nope", {})
    assert lines[0].startswith("раунд 1 · a → t (")
    assert lines[1].startswith("раунд 2 ·") and "ОШИБКА" in lines[1]
    assert [(r.round, r.server, r.tool, r.is_error) for r in rows] == [
        (1, "a", "t", False),
        (2, "?", "nope", True),
    ]
    assert rows[0].arguments == {"x": 1}

    def bad(_):
        raise RuntimeError("sink died")

    broken = mcp_router.Router([_spec("a")], trace=bad, journal=bad)
    broken.connect()
    assert broken.call("a__t", {}).is_error is False


# --- facade servers ----------------------------------------------------------


def test_facade_servers_expose_exactly_their_subset_of_server_tools():
    import anyio

    from week_04 import servers_pipeline, servers_repo
    from week_04.server import TOOL_NAMES

    expected = {
        servers_repo: ("list_days", "get_task", "git_log", "count_tokens"),
        servers_pipeline: ("summarize_text", "commit_digest"),
    }
    for module, names in expected.items():
        tools = anyio.run(module.mcp.list_tools)
        assert tuple(t.name for t in tools) == names
        assert set(names) <= set(TOOL_NAMES)
    assert set(servers_repo.TOOL_NAMES).isdisjoint(servers_pipeline.TOOL_NAMES)


def test_facade_registers_the_same_callable_not_a_copy():
    import anyio

    from week_04 import server, servers_repo  # noqa: F401
    from week_04.facade import build

    facade = build("x", "i", ("git_log",))
    (tool,) = anyio.run(facade.list_tools)
    original = next(t for t in anyio.run(server.mcp.list_tools) if t.name == "git_log")
    assert tool.description == original.description
    assert tool.input_schema == original.input_schema


def test_facade_build_fails_loudly_on_an_unknown_tool_name():
    from week_04.facade import build

    with pytest.raises((KeyError, AttributeError)):
        build("x", "i", ("no_such_tool",))


# --- agent round limit -------------------------------------------------------


def test_agent_round_limit_is_a_parameter_defaulting_to_the_old_constant():
    from advent_core.agent import MAX_TOOL_ROUNDS
    from tests.test_agent import _mcp_agent, _Recorder, _ToolDouble

    assert MAX_TOOL_ROUNDS == 3
    agent = _mcp_agent(_Recorder(), _ToolDouble())
    assert agent.max_tool_rounds == 3
    assert agent.tool_round == 0


def test_a_longer_round_limit_lets_a_longer_flow_finish():
    from tests.test_agent import (
        _final_round,
        _mcp_agent,
        _tool_round,
        _ToolDouble,
        _ToolRecorder,
    )

    rounds = [_tool_round('{"n": 1}', call_id=f"c{i}") for i in range(5)] + [_final_round("готово")]
    double = _ToolDouble()
    agent = _mcp_agent(_ToolRecorder(*rounds), double, max_tool_rounds=8)
    reply = agent.ask("длинный флоу", [])
    assert len(double.calls) == 5
    assert reply.text == "готово"
    assert agent.tool_round == 0  # reset once the loop is over


def test_the_round_number_is_visible_to_the_tool_callback():
    from tests.test_agent import _final_round, _mcp_agent, _tool_round, _ToolRecorder

    seen: list[int] = []
    holder: dict = {}

    def call(name, arguments):
        seen.append(holder["agent"].tool_round)
        return ToolCallOutcome(text="ок", is_error=False)

    agent = _mcp_agent(
        _ToolRecorder(_tool_round("{}"), _tool_round("{}"), _final_round()), call, max_tool_rounds=5
    )
    holder["agent"] = agent
    agent.ask("q", [])
    assert seen == [1, 2]


# --- demo scenario -----------------------------------------------------------


def test_demo_steps_w04d20_force_an_explicit_fs_stage_and_a_citation():
    from advent_cli import record as record_mod

    steps = record_mod.demo_steps(4, 20)
    assert steps[0].args == ["servers"]
    question = next(line for step in steps for line in step.stdin_lines if "git log" in line)
    assert record_mod._D20_FILE in question
    markers = ["git log", "сводк", "директори", record_mod._D20_FILE, "прочитай", "процитируй"]
    positions = [question.index(m) for m in markers]
    assert positions == sorted(positions), "the stages must be asked for in the flow's order"
    assert "инструмент" in question  # the summarize stage is demanded, not left to the model
    assert all(step.args == record_mod._DEMO_D20_SESSION_ARGS for step in steps[1:]), (
        "steps 2-3 share one session so /mcp on persists"
    )


def test_preflight_removes_the_previous_demo_file_and_warms_only_fs(tmp_path, monkeypatch):
    from advent_cli import record as record_mod

    monkeypatch.setattr(record_mod, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "logs" / "pipeline" / record_mod._D20_FILE
    target.parent.mkdir(parents=True)
    target.write_text("stale", encoding="utf-8")
    backend = _Backend(monkeypatch, {"fs": ["write_file"]})
    warmed: list[str] = []
    monkeypatch.setattr(
        mcp_router.mcp_client,
        "connect_and_list",
        lambda command, **kw: warmed.append(command[0]) or _listing("write_file"),
    )
    record_mod._prepare_w04d20(4, 20)
    assert not target.exists()
    assert len(warmed) == 1 and warmed[0].startswith("npx")
    assert backend.calls == []
    record_mod._prepare_w04d20(4, 19)  # other days are untouched
    assert len(warmed) == 1


def test_preflight_warns_and_does_not_raise_when_the_warmup_fails(monkeypatch, tmp_path, capsys):
    from advent_cli import record as record_mod

    monkeypatch.setattr(record_mod, "PROJECT_ROOT", tmp_path)

    def boom(command, **kwargs):
        raise MCPError("npx не скачался")

    monkeypatch.setattr(mcp_router.mcp_client, "connect_and_list", boom)
    record_mod._prepare_w04d20(4, 20)
    assert "npx не скачался" in capsys.readouterr().err


# --- transport quiet flag / `adventmcp servers` -------------------------------


def test_stdio_quiet_flag_only_changes_the_child_stderr(monkeypatch):
    from advent_core import mcp_client

    seen: list[dict] = []
    monkeypatch.setattr(
        mcp_client, "stdio_client", lambda params, **kw: seen.append(kw) or object()
    )
    params = object()
    mcp_client._stdio(params, False)
    assert seen == [{}]  # days 16-19: the SDK default, child stderr stays visible
    mcp_client._stdio(params, True)
    assert "errlog" in seen[1]
    seen[1]["errlog"].close()


def test_servers_command_prints_the_registry_and_what_allow_filtered(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from week_04 import cli as w4

    _Backend(monkeypatch, {"a": ["t1", "t2"], "b": ["w", "r"]}, down={"b"})
    registry = tmp_path / "s.json"
    registry.write_text(
        json.dumps(
            {
                "servers": [
                    {"name": "a", "command": "cmd-a", "allow": ["t1"]},
                    {"name": "b", "command": "cmd-b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(w4.app, ["servers", "--registry", str(registry)])
    assert result.exit_code == 1  # b is down: the verdict is not "all fine"
    out = re.sub(r"\[[0-9;]*m", "", result.stdout)
    assert "a__t1" in out and "a__t2" not in out
    assert "недоступен" in out
    assert "Инструментов для модели: 1, серверов на связи: 1 из 2" in " ".join(out.split())


def test_sdk_logger_level_survives_overlapping_calls():
    import logging
    import threading

    from advent_core import mcp_client

    logger = logging.getLogger("mcp.client.stdio")
    logger.setLevel(logging.WARNING)
    first_in, second_in = threading.Event(), threading.Event()
    release_first = threading.Event()

    def first():
        with mcp_client._quiet_sdk_logger():
            first_in.set()
            second_in.wait(5)
            release_first.wait(5)  # leaves BEFORE the second one

    def second():
        first_in.wait(5)
        with mcp_client._quiet_sdk_logger():
            second_in.set()
            release_first.set()
            threading.Event().wait(0.2)  # first exits while we still hold

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert logger.level == logging.WARNING


def test_quiet_stdio_keeps_the_child_stderr_tail_for_the_error(monkeypatch):
    import anyio

    from advent_core import mcp_client

    class Boom:
        async def __aenter__(self):
            raise OSError("spawn failed")

        async def __aexit__(self, *exc):
            return None

    holder: dict = {}

    def fake_stdio(params, errlog=None):
        errlog.write("npm ERR! 404 not found\n")
        holder["errlog"] = errlog
        return Boom()

    monkeypatch.setattr(mcp_client, "stdio_client", fake_stdio)
    with pytest.raises(MCPError) as caught:
        anyio.run(lambda: mcp_client.acall_tool(["x"], "t", {}, timeout=2, quiet=True))
    assert "npm ERR! 404 not found" in caught.value.message
    assert holder["errlog"].closed


def test_quiet_stdio_capture_is_closed_when_the_transport_never_starts(monkeypatch):
    from advent_core import mcp_client

    seen: dict = {}

    def fake_stdio(params, errlog=None):
        seen["errlog"] = errlog
        raise RuntimeError("no transport")

    monkeypatch.setattr(mcp_client, "stdio_client", fake_stdio)
    with pytest.raises(RuntimeError):
        mcp_client._stdio(object(), True)
    assert seen["errlog"].closed
