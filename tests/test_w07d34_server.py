"""Day 34: the stdio MCP file server, in-process and as a real subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from advent_core.errors import MCPError
from advent_core.mcp_client import call_tool_once, connect_and_list
from adventassistant import files_server, files_store
from tests.w07d34_repo import make_repo

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ["edit_file", "list_files", "read_file", "search_text", "write_file"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def _command(run_id: str, repo: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "adventassistant.files_server",
        "--run-id",
        run_id,
        "--root",
        str(repo),
    ]


def _call(command: list[str], name: str, args: dict):
    return call_tool_once(command, name, args, timeout=30, cwd=ROOT, quiet=True)


def test_in_process_tools_map_store_errors_to_tool_errors(repo: Path) -> None:
    run_id, _ = files_store.create_run("g", "m", repo)
    files_server.configure(run_id, repo)
    assert files_server.read_file("docs/a.md").endswith("3: gamma")
    assert files_server.list_files("docs/*").endswith("всего: 2")
    assert files_server.search_text("alpha").endswith("совпадений: 1 в файлах: 1")
    assert files_server.write_file("docs/n.md", "x\n") == "staged docs/n.md (+1)"
    assert files_server.edit_file("docs/n.md", "x", "y") == "staged docs/n.md (+1)"
    for call in (
        lambda: files_server.read_file("..\\secret.md"),
        lambda: files_server.read_file(".env"),
        lambda: files_server.write_file("src/app.py", "x"),
        lambda: files_server.edit_file("docs/a.md", "absent", "x"),
        lambda: files_server.search_text("(", regex=True),
    ):
        with pytest.raises(ToolError):
            call()


def test_server_refuses_to_start_without_a_valid_run(repo: Path) -> None:
    with pytest.raises(SystemExit) as info:
        files_server.main(["--run-id", "20261015-123005-deadbeef", "--root", str(repo)])
    assert info.value.code == 2
    with pytest.raises(SystemExit):
        files_server.main(["--run-id", "../../x", "--root", str(repo)])


def test_real_mcp_stdio_round_trip_keeps_state_on_disk(repo: Path) -> None:
    run_id, run_dir = files_store.create_run("найти", "m", repo)
    command = _command(run_id, repo)
    listing = connect_and_list(command, timeout=30, raw=False, sink=lambda _l: None, quiet=True)
    assert sorted(tool.name for tool in listing.tools) == TOOLS
    assert all(tool.description for tool in listing.tools)

    found = _call(command, "search_text", {"query": "call_tool_once"})
    assert not found.is_error and "src/app.py:1:" in found.text

    read = _call(command, "read_file", {"path": "docs/a.md", "start_line": 2, "end_line": 3})
    assert not read.is_error and read.text.splitlines()[1] == "2: Beta line"

    # each call is a fresh process: the write must be visible to the next one via the run dir
    written = _call(command, "write_file", {"path": "docs/new.md", "content": "hello\n"})
    assert not written.is_error and written.text == "staged docs/new.md (+1)"
    again = _call(command, "read_file", {"path": "docs/new.md"})
    assert again.text.endswith("1: hello")
    edited = _call(command, "edit_file", {"path": "docs/new.md", "old": "hello", "new": "bye"})
    assert edited.text == "staged docs/new.md (+1)"
    assert "+bye" in files_store.read_diff(run_id, repo)
    assert not (repo / "docs" / "new.md").exists()
    assert (run_dir / "files" / "docs" / "new.md").read_bytes() == b"bye\n"


def test_real_mcp_reports_policy_failures_as_tool_errors(repo: Path) -> None:
    run_id, _ = files_store.create_run("g", "m", repo)
    command = _command(run_id, repo)
    for name, args in (
        ("read_file", {"path": "..\\..\\etc\\passwd"}),
        ("read_file", {"path": ".env"}),
        ("write_file", {"path": "README.md", "content": "x"}),
        ("edit_file", {"path": "docs/a.md", "old": "nope", "new": "x"}),
        ("search_text", {"query": "(", "regex": True}),
    ):
        outcome = _call(command, name, args)
        assert outcome.is_error, name
        assert "supersecretvalue" not in outcome.text


def test_real_mcp_server_with_a_bad_run_id_fails_to_start(repo: Path) -> None:
    command = _command("20261015-123005-deadbeef", repo)
    with pytest.raises(MCPError):
        _call(command, "list_files", {})
