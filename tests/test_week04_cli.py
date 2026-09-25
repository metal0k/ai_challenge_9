"""`adventmcp tools` end to end, in a subprocess (real stdio server, no network)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from week_04 import cli, scheduler

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "COLUMNS": "200"}


def run_cli(*args: str, timeout: float = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "week_04.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env=ENV,
        timeout=timeout,
    )


def own_server_arg() -> str:
    return f'"{sys.executable}" -m week_04.server'


def test_default_server_lists_tools_and_verifies_them():
    proc = run_cli("tools")
    assert proc.returncode == 0
    # Счётчик сверки растёт вместе с TOOL_NAMES, и число здесь литеральное
    # намеренно: взятое из того же списка, что и код, оно не могло бы покраснеть.
    for name in (
        "list_days",
        "get_task",
        "count_tokens",
        "git_log",
        "schedule_job",
        "repo_activity_summary",
    ):
        assert name in proc.stdout
    assert "✓ 9 из 9 инструментов совпадают с ожидаемыми" in proc.stdout
    assert "week: integer · обязательный" in proc.stdout
    assert "model: string · опциональный" in proc.stdout
    assert "по умолчанию ministral-14b-latest" in proc.stdout


def test_stdout_table_header_is_printed_exactly_once():
    proc = run_cli("tools")
    assert proc.stdout.count("Инструмент ") == 1
    assert proc.stdout.count("Описание") == 1
    assert proc.stdout.count("Аргументы") == 1


def test_initialize_header_goes_to_stderr_not_stdout():
    proc = run_cli("tools")
    assert "сервер: advent-repo 0.1.0" in proc.stderr
    assert "advent-repo" not in proc.stdout


def test_mismatch_prints_missing_and_extra_and_exits_1():
    proc = run_cli("tools", "--expect", "list_days,get_task,delete_all")
    assert proc.returncode == 1
    assert (
        "✗ совпало 2 из 3; не хватает: delete_all; "
        "лишние: commit_digest, count_tokens, git_log, repo_activity_summary, "
        "save_to_file, schedule_job, summarize_text"
    ) in proc.stdout


def test_foreign_server_without_expect_is_not_verified():
    proc = run_cli("tools", "--server", own_server_arg())
    assert proc.returncode == 0
    assert "list_days" in proc.stdout
    assert "✓" not in proc.stdout
    assert "✗" not in proc.stdout


def test_expect_applies_to_a_foreign_server_too():
    proc = run_cli("tools", "--server", own_server_arg(), "--expect", "list_days")
    assert proc.returncode == 1
    assert (
        "лишние: commit_digest, count_tokens, get_task, git_log, "
        "repo_activity_summary, save_to_file, schedule_job, summarize_text"
    ) in proc.stdout


def test_nonexistent_command_exits_8_with_readable_reason_and_no_traceback():
    proc = run_cli("tools", "--server", "no-such-mcp-server")
    assert proc.returncode == 8
    assert "Ошибка: исполняемый файл не найден: no-such-mcp-server" in proc.stderr
    assert "Проверьте значение --server и что команда есть в PATH." in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_crashing_child_exits_8(tmp_path):
    script = tmp_path / "crash.py"
    script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    proc = run_cli("tools", "--server", f'"{sys.executable}" "{script}"')
    assert proc.returncode == 8
    assert "Ошибка: процесс сервера завершился до конца handshake" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_silent_server_times_out_with_exit_8(tmp_path):
    script = tmp_path / "silent.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = run_cli("tools", "--server", f'"{sys.executable}" "{script}"', "--timeout", "1")
    assert proc.returncode == 8
    assert "Ошибка: сервер не ответил за 1 с на этапе handshake" in proc.stderr
    assert "увеличьте --timeout" in proc.stderr


def test_raw_frames_go_to_stderr_and_stdout_stays_the_table():
    proc = run_cli("tools", "--raw")
    assert proc.returncode == 0
    assert any(line.startswith("→ {") for line in proc.stderr.splitlines())
    assert any(line.startswith("← {") for line in proc.stderr.splitlines())
    assert "→" not in proc.stdout
    assert "←" not in proc.stdout
    assert "jsonrpc" not in proc.stdout


def test_raw_does_not_change_stdout():
    plain = run_cli("tools")
    raw = run_cli("tools", "--raw")
    assert raw.stdout == plain.stdout


def test_no_arguments_shows_help():
    proc = run_cli()
    assert "tools" in proc.stdout + proc.stderr


def test_chatty_server_gives_one_clean_error_without_sdk_traceback(tmp_path):
    script = tmp_path / "chatty.py"
    script.write_text("print('hello', flush=True)\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    proc = run_cli("tools", "--server", f'"{sys.executable}" "{script}"', "--timeout", "5")
    assert proc.returncode == 8
    assert "Ошибка: сервер пишет в stdout не JSON-RPC" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "ValidationError" not in proc.stderr


def test_raw_tools_list_frame_shows_every_tool_name_on_the_wire():
    proc = run_cli("tools", "--raw")
    frame = next(line for line in proc.stderr.splitlines() if '"tools":[' in line)
    # Six tools no longer fit RAW_LINE_LIMIT: the middle is elided, so only the
    # head of the list is guaranteed on screen.
    assert '"name":"list_days"' in frame


def test_scheduler_help_lists_run_and_flags():
    proc = run_cli("scheduler", "run", "--help")
    assert proc.returncode == 0
    assert "--once" in proc.stdout
    assert "--interval" in proc.stdout


def test_scheduler_without_subcommand_shows_help():
    proc = run_cli("scheduler")
    assert "run" in proc.stdout + proc.stderr


def test_tools_stays_a_real_subcommand_next_to_scheduler():
    proc = run_cli("--help")
    assert "tools" in proc.stdout
    assert "scheduler" in proc.stdout


def test_scheduler_once_runs_one_due_tick_without_sleeping(tmp_path, monkeypatch):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    scheduler.upsert_job(5, now=lambda: 0, path=job_file)  # created in 1970: due now

    def no_sleep(_seconds):
        raise AssertionError("--once must not sleep")

    monkeypatch.setattr(cli.time, "sleep", no_sleep)
    result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once"])
    assert result.exit_code == 0
    state = scheduler.load_state(job_file)
    assert len(state.runs) == 1
    assert state.job is not None and state.job.last_run_at is not None


def test_scheduler_once_without_job_exits_0_and_writes_nothing(tmp_path, monkeypatch):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once"])
    assert result.exit_code == 0
    assert not job_file.exists()


def test_scheduler_interval_out_of_bounds_exits_1_and_creates_no_job(tmp_path, monkeypatch):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    for bad in ("1", "3601"):
        result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once", "--interval", bad])
        assert result.exit_code == 1
    assert not job_file.exists()


def test_scheduler_interval_upserts_job_before_once(tmp_path, monkeypatch):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once", "--interval", "30"])
    assert result.exit_code == 0
    state = scheduler.load_state(job_file)
    assert state.job is not None and state.job.interval_seconds == 30


def test_scheduler_ctrl_c_exits_130(monkeypatch):
    def boom(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler, "scheduler_loop", boom)
    result = CliRunner().invoke(cli.app, ["scheduler", "run"])
    assert result.exit_code == 130


def test_tick_line_formats_ok_overflow_and_error_runs():
    ok = scheduler.Run(ts="T", new_commits=2, total_commits=9)
    assert cli._tick_line(ok) == "тик T: новых коммитов 2, всего 9"
    over = scheduler.Run(ts="T", new_commits=20, total_commits=90, overflow=True)
    assert cli._tick_line(over) == "тик T: новых коммитов 20+, всего 90"
    bad = scheduler.Run(ts="T", new_commits=None, total_commits=None, error="[boom]")
    assert cli._tick_line(bad) == "тик T: ошибка — \[boom]"  # rich-escaped


def test_once_tick_line_goes_to_stderr_and_stdout_stays_empty(tmp_path, monkeypatch, capsys):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    scheduler.upsert_job(5, now=lambda: 0, path=job_file)
    fake = scheduler.Run(ts="T1", new_commits=1, total_commits=3, head_hash="h")
    monkeypatch.setattr(scheduler, "run_tick", lambda job, **_k: fake)
    result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once"])
    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert result.stdout == "" and captured.out == ""
    # Rich may colour the line when the env forces colour; compare the plain text
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr + captured.err)
    assert "тик T1: новых коммитов 1, всего 3" in plain


def test_interval_bounds_error_is_on_stderr_and_writes_nothing(tmp_path, monkeypatch, capsys):
    job_file = tmp_path / "job.json"
    monkeypatch.setattr(scheduler, "JOB_FILE", job_file)
    result = CliRunner().invoke(cli.app, ["scheduler", "run", "--once", "--interval", "3601"])
    captured = capsys.readouterr()
    assert result.exit_code == 1
    assert result.stdout == "" and captured.out == ""
    assert "от 5 до 3600, получено 3601" in result.stderr + captured.err
    assert not job_file.exists()
