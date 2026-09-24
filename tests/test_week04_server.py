"""Repository MCP server: tools as plain functions, plus the stdout contract."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

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


def test_tool_names_constant_has_all_six_tools():
    assert server.TOOL_NAMES == (
        "list_days",
        "get_task",
        "count_tokens",
        "git_log",
        "schedule_job",
        "repo_activity_summary",
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
