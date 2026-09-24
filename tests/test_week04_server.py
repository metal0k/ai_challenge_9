"""Repository MCP server: tools as plain functions, plus the stdout contract."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from advent_core.telemetry import CallResult
from week_04 import scheduler, server

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


def test_tool_names_constant_has_all_nine_tools():
    assert server.TOOL_NAMES == (
        "list_days",
        "get_task",
        "count_tokens",
        "git_log",
        "schedule_job",
        "repo_activity_summary",
        "summarize_text",
        "save_to_file",
        "commit_digest",
    )


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


def test_git_log_returns_pinned_line_format():
    lines = server.git_log(3).splitlines()
    # Exactly 3, not "1..3": a range passes even when `n` is ignored outright
    # and the tool always answers with one line. This repo has far more.
    assert len(lines) == 3
    line_re = re.compile(r"^[0-9a-f]+ {2}\d{4}-\d{2}-\d{2} {2}.+ {2}.+$")
    for line in lines:
        assert line_re.match(line), line
    assert len(server.git_log(1).splitlines()) == 1


def test_git_log_rejects_out_of_range_n():
    with pytest.raises(ToolError, match="n должен быть от 1 до 20, получено 0"):
        server.git_log(0)
    with pytest.raises(ToolError, match="n должен быть от 1 до 20, получено 21"):
        server.git_log(21)


def test_git_log_survives_missing_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(server.subprocess, "run", no_git)
    with pytest.raises(ToolError, match="git log не выполнился"):
        server.git_log(5)


def test_git_log_raises_on_nonzero_returncode(monkeypatch):
    class Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository\n"

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Failed())
    with pytest.raises(ToolError, match="fatal: not a git repository"):
        server.git_log(5)


def test_git_log_query_none_is_the_pinned_day17_argv(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stdout = "x"
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda cmd, **k: calls.append(cmd) or Done())
    server.git_log(5)
    assert calls[0] == ["git", "log", "-5", "--pretty=format:%h  %ad  %an  %s", "--date=short"]


def test_git_log_query_filters_via_grep(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stdout = "abc1234  2026-09-24  Denis  fix MCP thing\n"
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda cmd, **k: calls.append(cmd) or Done())
    result = server.git_log(5, "MCP")
    assert result == Done.stdout
    assert calls[0] == [
        "git",
        "log",
        "-5",
        "--grep=MCP",
        "-i",
        "-F",
        "--pretty=format:%h  %ad  %an  %s",
        "--date=short",
    ]


def test_git_log_query_not_found_returns_message(monkeypatch):
    class Empty:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Empty())
    assert server.git_log(5, "no-such-thing") == (
        "коммитов по запросу «no-such-thing» не найдено (среди последних 5)"
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


# --- Day 18: scheduler tools -------------------------------------------------


def _iso(minutes_ago: float) -> str:
    moment = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return moment.isoformat(timespec="seconds")


@pytest.fixture
def job_file(tmp_path, monkeypatch):
    path = tmp_path / "repo_activity.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", path)
    return path


@pytest.mark.parametrize("bad", [0, 4, 3601, -10])
def test_schedule_job_rejects_out_of_range_interval(job_file, bad):
    with pytest.raises(ToolError, match=f"от 5 до 3600, получено {bad}"):
        server.schedule_job(bad)
    assert not job_file.exists()


@pytest.mark.parametrize("edge", [5, 3600])
def test_schedule_job_accepts_bounds(job_file, edge):
    assert f"интервал {edge} с" in server.schedule_job(edge)


def test_schedule_job_create_same_change_messages(job_file):
    assert server.schedule_job(15) == (
        "job repo_activity создан: интервал 15 с. Демон подхватит его на следующем опросе."
    )
    assert server.schedule_job(15) == "job repo_activity: интервал уже 15 с, без изменений."
    assert server.schedule_job(60) == "job repo_activity: интервал изменён 15 → 60 с."
    assert scheduler.load_state().job.interval_seconds == 60


def test_summary_without_job_raises(job_file):
    with pytest.raises(ToolError, match="schedule_job"):
        server.repo_activity_summary()


@pytest.mark.parametrize("bad", [0, -5])
def test_summary_rejects_non_positive_minutes(job_file, bad):
    with pytest.raises(ToolError, match="положительным"):
        server.repo_activity_summary(bad)


def test_summary_job_without_runs_says_no_ticks(job_file):
    server.schedule_job(15)
    assert "тиков ещё не было" in server.repo_activity_summary()


def _seed_runs(runs):
    state = scheduler.load_state()
    scheduler.save_state(scheduler.SchedulerState(job=state.job, runs=runs))


def test_summary_aggregates_recorded_runs(job_file):
    server.schedule_job(15)
    _seed_runs(
        [
            scheduler.Run(
                ts=_iso(3),
                new_commits=0,
                total_commits=40,
                head_hash="a" * 40,
            ),
            scheduler.Run(
                ts=_iso(2),
                new_commits=2,
                total_commits=42,
                commits=[
                    "bbb1111  2026-09-23  Denis  second feature",
                    "ccc2222  2026-09-23  Denis  first feature",
                ],
                head_hash="b" * 40,
            ),
            scheduler.Run(
                ts=_iso(1),
                new_commits=1,
                total_commits=43,
                commits=["ddd3333  2026-09-23  Denis  third feature"],
                head_hash="c" * 40,
            ),
        ]
    )
    text = server.repo_activity_summary()
    assert "тиков в окне: 3 (" in text
    assert "новых коммитов: 3\n" in text
    assert "всего коммитов в репозитории: 43\n" in text
    assert "  ddd3333  2026-09-23  Denis  third feature" in text
    assert "  ccc2222  2026-09-23  Denis  first feature" in text


def test_summary_minutes_filter_drops_old_runs(job_file):
    server.schedule_job(15)
    _seed_runs(
        [
            scheduler.Run(
                ts=_iso(180),
                new_commits=1,
                total_commits=10,
                commits=["old0001  2026-09-23  Denis  ancient commit"],
                head_hash="a" * 40,
            ),
            scheduler.Run(
                ts=_iso(1),
                new_commits=1,
                total_commits=11,
                commits=["new0002  2026-09-23  Denis  fresh commit"],
                head_hash="b" * 40,
            ),
        ]
    )
    text = server.repo_activity_summary(minutes=10)
    assert "тиков в окне: 1 (" in text
    assert "новых коммитов: 1\n" in text
    assert "fresh commit" in text
    assert "ancient commit" not in text
    assert "ancient commit" in server.repo_activity_summary()


def test_summary_minutes_window_empty_reports_no_ticks(job_file):
    server.schedule_job(15)
    _seed_runs([scheduler.Run(ts=_iso(180), new_commits=1, total_commits=10, head_hash="a" * 40)])
    assert "тиков за последние 10 мин не найдено" in server.repo_activity_summary(minutes=10)


def test_tools_turn_oserror_into_toolerror(job_file, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(scheduler, "load_state", boom)
    with pytest.raises(ToolError, match="не удалось обновить состояние"):
        server.schedule_job(15)
    with pytest.raises(ToolError, match="не удалось прочитать состояние"):
        server.repo_activity_summary()


# --- Day 19: summarize_text / save_to_file / commit_digest ------------------


def _fake_result(text: str = "краткая сводка") -> CallResult:
    return CallResult(text=text, model_requested=server.SUMMARIZE_MODEL)


def test_summarize_text_rejects_empty_input():
    with pytest.raises(ToolError, match="text пустой"):
        server.summarize_text("   ")


def test_summarize_text_config_error_becomes_tool_error(monkeypatch):
    from advent_core.config import ConfigError

    def boom(**kwargs):
        raise ConfigError("нет ключа")

    monkeypatch.setattr(server.Config, "resolve", boom)
    with pytest.raises(ToolError, match="нет ключа"):
        server.summarize_text("текст для сводки")


def test_summarize_text_advent_error_becomes_tool_error(monkeypatch):
    from advent_core.errors import AdventError

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    def boom(*a, **k):
        raise AdventError("сеть недоступна")

    monkeypatch.setattr(server.chat_core, "complete", boom)
    monkeypatch.setattr(server.journal, "log_call", lambda *a, **k: None)
    with pytest.raises(ToolError, match="не удалось получить сводку: сеть недоступна"):
        server.summarize_text("текст для сводки")


def test_summarize_text_happy_path_calls_chat_and_journal(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        server.chat_core, "complete", lambda *a, **k: _fake_result("  готовая сводка  ")
    )
    logged = []
    monkeypatch.setattr(server.journal, "log_call", lambda *a, **k: logged.append(k))
    assert server.summarize_text("длинный текст") == "готовая сводка"
    assert logged[0]["week"] == 4
    assert logged[0]["day"] == 19
    assert logged[0]["extra"] == {"kind": "mcp_summarize"}


@pytest.mark.parametrize("bad", ["../x", "a/b", ""])
def test_save_to_file_rejects_unsafe_filenames(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="недопустимое имя файла"):
        server.save_to_file("content", bad)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="backslash is only a path separator on Windows")
def test_save_to_file_rejects_backslash_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="недопустимое имя файла"):
        server.save_to_file("content", "a\\b")
    assert list(tmp_path.iterdir()) == []


def test_save_to_file_rejects_empty_content(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="content пустой"):
        server.save_to_file("   ", "note.md")


def test_save_to_file_accepts_plain_name(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    result = server.save_to_file("hello", "note.md")
    assert result == "Сохранено: logs/pipeline/note.md (5 симв.)"
    assert (tmp_path / "note.md").read_text(encoding="utf-8") == "hello"


def test_commit_digest_rejects_empty_query(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="query пустой"):
        server.commit_digest("   ")


def test_commit_digest_short_circuits_on_no_matches(tmp_path, monkeypatch):
    class Empty:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Empty())
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)

    def must_not_be_called(*a, **k):
        raise AssertionError("summarize_text must not call chat_core.complete")

    monkeypatch.setattr(server.chat_core, "complete", must_not_be_called)

    def save_must_not_be_called(*a, **k):
        raise AssertionError("save_to_file must not be called")

    monkeypatch.setattr(server, "save_to_file", save_must_not_be_called)

    result = server.commit_digest("no-such-query")
    assert result == (
        "Коммитов по запросу «no-such-query» не найдено — Mistral и файл не задействованы."
    )
    assert list(tmp_path.iterdir()) == []


def test_commit_digest_happy_path_writes_file_and_returns_summary(tmp_path, monkeypatch):
    class Done:
        returncode = 0
        stdout = "abc1234  2026-09-24  Denis  fix MCP thing\n"
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Done())
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)

    sent_messages = []

    def fake_complete(config, messages, *a, **k):
        sent_messages.append(messages)
        return _fake_result("итоговая сводка")

    monkeypatch.setattr(server.chat_core, "complete", fake_complete)
    monkeypatch.setattr(server.journal, "log_call", lambda *a, **k: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    result = server.commit_digest("MCP", n=5)
    assert result.startswith("Сохранено: logs/pipeline/mcp-")
    assert "Сводка:\nитоговая сводка" in result

    # Data-handoff check (the day's own headline claim): the commit found in
    # step 1 must be what actually reached summarize_text/chat_core.complete
    # in step 2, not just what step 3 independently re-derives from `commits`.
    assert len(sent_messages) == 1
    sent_text = " ".join(m["content"] for m in sent_messages[0])
    assert "abc1234  2026-09-24  Denis  fix MCP thing" in sent_text
    assert "MCP" in sent_text

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "итоговая сводка" in content
    assert "abc1234  2026-09-24  Denis  fix MCP thing" in content


def test_commit_digest_survives_missing_git(tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(server.subprocess, "run", no_git)
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="git log не выполнился"):
        server.commit_digest("MCP")
    assert list(tmp_path.iterdir()) == []


def test_commit_digest_raises_on_nonzero_returncode(tmp_path, monkeypatch):
    class Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository\n"

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: Failed())
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match="fatal: not a git repository"):
        server.commit_digest("MCP")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad_n", [0, 21])
def test_commit_digest_rejects_out_of_range_n(tmp_path, monkeypatch, bad_n):
    monkeypatch.setattr(server, "PIPELINE_DIR", tmp_path)
    with pytest.raises(ToolError, match=f"n должен быть от 1 до 20, получено {bad_n}"):
        server.commit_digest("MCP", n=bad_n)
    assert list(tmp_path.iterdir()) == []
