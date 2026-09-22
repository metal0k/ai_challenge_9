"""MCP client against a REAL local stdio subprocess (no network), plus error mapping."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import anyio
import pytest

from advent_core import mcp_client as client_mod
from advent_core.errors import MCPError
from advent_core.mcp_client import (
    ToolArg,
    ToolCallOutcome,
    acall_tool,
    call_tool_once,
    connect_and_list,
    split_command,
    summarize_args,
)
from week_04.client import default_server_command

TIMEOUT = 30.0


def _list(cmd=None, *, raw=False, timeout=TIMEOUT):
    lines: list[str] = []
    result = connect_and_list(
        default_server_command() if cmd is None else cmd,
        timeout=timeout,
        raw=raw,
        sink=lines.append,
    )
    return result, lines


def _call(name, arguments, *, cmd=None, timeout=TIMEOUT):
    return call_tool_once(
        default_server_command() if cmd is None else cmd,
        name,
        arguments,
        timeout=timeout,
    )


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---- real server ---------------------------------------------------------


def test_lists_the_three_tools_of_the_repo_server():
    result, lines = _list()
    assert result.server_name == "advent-repo"
    assert result.server_version == "0.1.0"
    assert result.protocol_version
    # Containment, not equality: this file tests the transport, not the server's
    # exact tool set (that's test_week04_server.py's contract) — a sibling tool
    # (e.g. git_log) landing on the server must not fail the transport test.
    assert {"list_days", "get_task", "count_tokens"} <= {t.name for t in result.tools}
    assert lines == []  # no tap without --raw


def test_argument_summaries_come_from_the_input_schema():
    result, _ = _list()
    by_name = {t.name: t for t in result.tools}
    assert by_name["list_days"].args == ()
    assert by_name["get_task"].args == (
        ToolArg("week", "integer", True),
        ToolArg("day", "integer", True),
    )
    assert by_name["count_tokens"].args == (
        ToolArg("text", "string", True),
        ToolArg(
            "model",
            "string",
            False,
            ("ministral-3b-latest", "ministral-8b-latest", "ministral-14b-latest"),
            "ministral-14b-latest",
        ),
    )


def test_raw_tap_mirrors_frames_in_both_directions():
    _result, lines = _list(raw=True)
    sent = [line for line in lines if line.startswith("→ ")]
    received = [line for line in lines if line.startswith("← ")]
    assert len(sent) + len(received) == len(lines)
    assert any('"method":"tools/list"' in line for line in sent)
    assert any('"name":"list_days"' in line for line in received)


def test_raw_tap_does_not_change_the_tool_list():
    plain, plain_lines = _list(raw=False)
    tapped, tapped_lines = _list(raw=True)
    assert tapped == plain
    assert plain_lines == []
    assert tapped_lines != []


# ---- tools/call (real server) --------------------------------------------


def test_call_tool_once_returns_the_text_result_of_a_no_arg_tool():
    outcome = _call("list_days", {})
    assert isinstance(outcome, ToolCallOutcome)
    assert outcome.is_error is False
    # list_days() always returns tag lines or an explicit "none found" text —
    # never "": a bare isinstance(str) check would pass even on empty extraction.
    assert outcome.text.strip()


def test_call_tool_once_reports_a_server_side_tool_error():
    # get_task validates its week/day and raises ToolError server-side.
    outcome = _call("get_task", {"week": 999, "day": 1})
    assert outcome.is_error is True
    assert outcome.text  # server's error text, not empty


def test_call_tool_once_passes_arguments_through():
    # count_tokens's answer is a function of `text`'s length: two different
    # values must produce two different results, proving the argument dict
    # itself crossed the wire rather than a stub/default being used.
    short = _call("count_tokens", {"text": "a"})
    long = _call("count_tokens", {"text": "a" * 50})
    assert short.is_error is False
    assert long.is_error is False
    assert "привет" not in short.text  # tool returns a token count, not an echo
    assert "токенов" in short.text
    assert short.text != long.text


def test_acall_tool_mirrors_the_sync_wrapper():
    async def scenario():
        return await acall_tool(default_server_command(), "list_days", {}, timeout=TIMEOUT)

    outcome = anyio.run(scenario)
    assert isinstance(outcome, ToolCallOutcome)
    assert outcome.is_error is False


def test_call_tool_once_missing_executable_raises_mcp_error():
    with pytest.raises(MCPError) as info:
        _call("list_days", {}, cmd=["no-such-mcp-server"])
    assert info.value.message == "исполняемый файл не найден: no-such-mcp-server"
    assert info.value.exit_code == 8


# ---- pagination (fake SDK client) ---------------------------------------


class FakeClient:
    pages: list = []

    def __init__(self, transport):
        self.server_info = SimpleNamespace(name="fake", version="9")
        self.protocol_version = "2026-01-01"
        self.instructions = None
        self.cursors: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def list_tools(self, cursor=None):
        self.cursors.append(cursor)
        return FakeClient.pages[len(self.cursors) - 1]


def _tool(name):
    return SimpleNamespace(name=name, description=None, input_schema=None)


def test_pagination_follows_next_cursor_until_exhausted(monkeypatch):
    FakeClient.pages = [
        SimpleNamespace(tools=[_tool("a")], next_cursor="c1"),
        SimpleNamespace(tools=[_tool("b")], next_cursor="c2"),
        SimpleNamespace(tools=[_tool("c")], next_cursor=None),
    ]
    monkeypatch.setattr(client_mod, "Client", FakeClient)
    result, _ = _list(["whatever"])
    assert [t.name for t in result.tools] == ["a", "b", "c"]
    assert result.server_name == "fake"
    assert result.tools[0].description == ""


def test_repeated_cursor_is_reported_as_a_loop(monkeypatch):
    FakeClient.pages = [SimpleNamespace(tools=[_tool("a")], next_cursor="same")] * 5
    monkeypatch.setattr(client_mod, "Client", FakeClient)
    with pytest.raises(MCPError, match="зациклил пагинацию"):
        _list(["whatever"])


# ---- failures ------------------------------------------------------------


def test_missing_executable_message():
    with pytest.raises(MCPError) as info:
        _list(["no-such-mcp-server"])
    assert info.value.message == "исполняемый файл не найден: no-such-mcp-server"
    assert info.value.exit_code == 8


def test_child_exiting_at_once_is_reported_as_early_exit():
    with pytest.raises(MCPError) as info:
        _list([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=20)
    assert info.value.message == "процесс сервера завершился до конца handshake (этап handshake)"


def test_silent_server_times_out(tmp_path):
    script = tmp_path / "silent.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(MCPError) as info:
        _list([sys.executable, str(script)], timeout=1.5)
    assert info.value.message == "сервер не ответил за 1.5 с на этапе handshake"
    assert time.monotonic() - started < 20


def test_timed_out_child_is_not_leaked(tmp_path):
    script = tmp_path / "pid_server.py"
    pidfile = tmp_path / "pid.txt"
    script.write_text(
        "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    with pytest.raises(MCPError):
        _list([sys.executable, str(script), str(pidfile)], timeout=2)
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(pid)


def test_non_protocol_output_is_reported_as_such_not_as_early_exit(tmp_path):
    script = tmp_path / "chatty.py"
    script.write_text(
        "print('hello, not json', flush=True)\nimport time\ntime.sleep(60)\n", encoding="utf-8"
    )
    with pytest.raises(MCPError) as info:
        _list([sys.executable, str(script)], timeout=5)
    assert info.value.message == "сервер пишет в stdout не JSON-RPC"
    assert "stderr" in info.value.hint


def test_chatty_server_that_exits_is_still_reported_as_non_protocol(tmp_path):
    script = tmp_path / "chatty_exit.py"
    script.write_text("print('hello, not json')\n", encoding="utf-8")
    with pytest.raises(MCPError) as info:
        _list([sys.executable, str(script)], timeout=10)
    assert info.value.message == "сервер пишет в stdout не JSON-RPC"


def test_pagination_stops_after_max_pages_with_unique_cursors(monkeypatch):
    FakeClient.pages = [
        SimpleNamespace(tools=[], next_cursor=f"c{i}") for i in range(client_mod.MAX_PAGES + 5)
    ]
    monkeypatch.setattr(client_mod, "Client", FakeClient)
    with pytest.raises(MCPError) as info:
        _list(["whatever"])
    assert info.value.message == "tools/list не закончился за 100 страниц"


def test_sdk_stdio_logger_level_is_restored_after_a_run(monkeypatch):
    import logging

    logger = logging.getLogger("mcp.client.stdio")
    logger.setLevel(logging.WARNING)
    FakeClient.pages = [SimpleNamespace(tools=[], next_cursor=None)]
    monkeypatch.setattr(client_mod, "Client", FakeClient)
    _list(["whatever"])
    assert logger.level == logging.WARNING


def test_empty_command_is_rejected():
    with pytest.raises(MCPError, match="команда сервера пустая"):
        _list([])


# ---- helpers -------------------------------------------------------------


def test_split_command_keeps_windows_paths_and_strips_quotes():
    parts = split_command(
        r'"C:\Program Files\Python312\python.exe" -m week_04.server --flag', windows=True
    )
    assert parts == [r"C:\Program Files\Python312\python.exe", "-m", "week_04.server", "--flag"]


def test_split_command_unquoted_windows_path():
    assert split_command(r"C:\opt\srv.exe --x 1", windows=True) == [
        r"C:\opt\srv.exe",
        "--x",
        "1",
    ]


def test_split_command_posix_uses_shell_quoting():
    assert split_command("python -m srv 'a b'", windows=False) == ["python", "-m", "srv", "a b"]


def test_split_command_unbalanced_quote_is_an_mcp_error():
    with pytest.raises(MCPError, match="не удалось разобрать команду сервера"):
        split_command('python "unclosed', windows=False)


def test_summarize_args_handles_anyof_and_missing_schema():
    assert summarize_args(None) == ()
    schema = {
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["x"],
    }
    assert summarize_args(schema) == (ToolArg("x", "string|null", True),)
