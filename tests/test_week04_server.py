"""Repository MCP server: tools as plain functions, plus the stdout contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from week_04 import server

ROOT = Path(__file__).resolve().parent.parent

TASKS = """# Week

## Day 01

first task

## Day 02

second task
line two

## Day 03

third
"""


def test_tool_names_constant_is_the_literal_trio():
    assert server.TOOL_NAMES == ("list_days", "get_task", "count_tokens")


def test_extract_day_returns_only_that_section():
    assert server.extract_day(TASKS, 2) == "## Day 02\n\nsecond task\nline two"


def test_extract_day_accepts_zero_padded_heading_and_last_section():
    assert server.extract_day(TASKS, 3) == "## Day 03\n\nthird"
    assert server.extract_day(TASKS, 1) == "## Day 01\n\nfirst task"


def test_extract_day_missing_returns_none():
    assert server.extract_day(TASKS, 9) is None


def test_task_file_prefers_plural_and_falls_back_to_singular(tmp_path):
    (tmp_path / "week_04").mkdir()
    (tmp_path / "week_04" / "task_04.md").write_text("x", encoding="utf-8")
    (tmp_path / "week_02").mkdir()
    (tmp_path / "week_02" / "tasks_02.md").write_text("y", encoding="utf-8")
    assert server.task_file(4, tmp_path) == tmp_path / "week_04" / "task_04.md"
    assert server.task_file(2, tmp_path) == tmp_path / "week_02" / "tasks_02.md"
    assert server.task_file(7, tmp_path) is None


def test_get_task_reads_the_real_week_04_file():
    text = server.get_task(4, 16)
    assert text.startswith("## Day 16")
    assert "Подключение MCP" in text


def test_get_task_missing_week_and_day_raise_tool_errors():
    with pytest.raises(ToolError, match="нет файла задач для недели 9"):
        server.get_task(9, 1)
    with pytest.raises(ToolError, match="нет раздела Day 99"):
        server.get_task(4, 99)


def test_list_days_survives_missing_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(server.subprocess, "run", no_git)
    assert server.submitted_days() == []
    assert server.list_days() == "тегов wNNdDD не найдено"


def test_list_days_outside_a_repository_is_empty(tmp_path):
    assert server.submitted_days(tmp_path) == []


def test_submitted_days_keeps_only_day_tags(monkeypatch):
    class Done:
        returncode = 0
        stdout = "v1\nw01d01\nrelease\nw03d15\nw3d5\n"

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Done())
    assert server.submitted_days() == ["w01d01", "w03d15"]


def test_count_tokens_estimate_for_a_model_without_exact_tokenizer():
    # 30 chars at the default 3.0 chars/token → 10; no overhead model for estimates.
    assert server.count_tokens("a" * 30, "ministral-8b-latest") == (
        "10 токенов (оценка, ministral-8b-latest)"
    )


def test_server_writes_nothing_to_stdout_without_a_client():
    proc = subprocess.run(
        [sys.executable, "-m", "week_04.server"],
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=60,
    )
    assert proc.stdout == ""
