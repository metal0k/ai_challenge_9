"""`adventmcp tools` end to end, in a subprocess (real stdio server, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    for name in ("list_days", "get_task", "count_tokens"):
        assert name in proc.stdout
    assert "✓ 3 из 3 инструментов совпадают с ожидаемыми" in proc.stdout
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
    assert "✗ совпало 2 из 3; не хватает: delete_all; лишние: count_tokens" in proc.stdout


def test_foreign_server_without_expect_is_not_verified():
    proc = run_cli("tools", "--server", own_server_arg())
    assert proc.returncode == 0
    assert "list_days" in proc.stdout
    assert "✓" not in proc.stdout
    assert "✗" not in proc.stdout


def test_expect_applies_to_a_foreign_server_too():
    proc = run_cli("tools", "--server", own_server_arg(), "--expect", "list_days")
    assert proc.returncode == 1
    assert "лишние: count_tokens, get_task" in proc.stdout


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
    for name in ('"name":"list_days"', '"name":"get_task"', '"name":"count_tokens"'):
        assert name in frame
