"""Day 35 recording plan and preflight; no network, no live OBS."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from advent_cli import record
from advent_core.errors import AdventError

DAY = ["--day", "35", "--week", "7"]


def test_step_list_order_titles_and_args() -> None:
    steps = record.demo_steps(7, 35)
    assert [s.title for s in steps] == [
        "1. Таблица стадий дня: что сделано, что ещё нет",
        "2. run --dry-run: готовые стадии пропущены, review идёт вживую",
        "3. sheet-probe: ячейка и число комментариев, ничего не пишется",
        "4. Реальный запуск без терминала: gate отказывает до любых изменений",
    ]
    assert {s.module for s in steps} == {"adventship.cli"}
    assert [s.args for s in steps] == [
        ["status", *DAY],
        ["run", *DAY, "--dry-run"],
        ["sheet-probe", *DAY],
        ["run", *DAY],
    ]
    assert record.MODULE_COMMANDS["adventship.cli"] == "adventship"


def test_last_step_expects_exit_8_without_a_tty_and_holds_the_screen() -> None:
    steps = record.demo_steps(7, 35)
    last = steps[-1]
    assert last.expect_exit == 8
    assert last.no_tty is True
    assert last.line_pause == 8.0
    assert all(s.expect_exit is None for s in steps[:-1])
    # review runs live inside step 2; step 4 is a real run that may re-check stages
    assert steps[1].timeout is not None and steps[1].timeout >= 600
    assert last.timeout is not None and last.timeout >= 600
    assert "--dry-run" not in last.args


def test_check_demands_the_exact_exit_code() -> None:
    step = record.demo_steps(7, 35)[-1]
    record._check(step, 8)
    for code in (0, 2, 9):
        with pytest.raises(AdventError, match="кодом 8"):
            record._check(step, code)


def test_no_tty_step_gets_devnull_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(command, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(command, 8)

    monkeypatch.setattr(record.subprocess, "run", fake_run)
    monkeypatch.setattr(record, "_reset_sessions", lambda names: None)
    record._run_step(record.demo_steps(7, 35)[-1], pause=0)
    assert seen["stdin"] is subprocess.DEVNULL


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _repo(tmp_path: Path) -> None:
    """Every prepare test runs in a real temp repo whose HEAD is the day's commit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "feat(w07d35): ship pipeline")


def _write_state(root: Path, *, probe_age: timedelta = timedelta(hours=1), ok: bool = True) -> None:
    now = datetime.now().astimezone()
    stages = {n: {"status": "done"} for n in ("intake", "checks", "describe", "review")}
    doc = {
        "stages": stages,
        "sheet_probe": {
            "at": (now - probe_age).isoformat(timespec="seconds"),
            "result": {"ok": ok, "cell": "AJ15"},
        },
    }
    path = root / "logs" / "ship" / "w07d35" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_prepare_ignores_other_days_and_refuses_without_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    record._prepare_w07d35(7, 34)
    with pytest.raises(AdventError) as info:
        record._prepare_w07d35(7, 35)
    assert "adventship run --day 35 --week 7 --dry-run" in (info.value.hint or "")
    assert "sheet-probe" in (info.value.hint or "")


def test_prepare_passes_with_a_synthetic_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    _write_state(tmp_path)
    record._prepare_w07d35(7, 35)


def test_prepare_refuses_missing_stage_stale_or_failed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    _write_state(tmp_path, probe_age=timedelta(hours=30))
    with pytest.raises(AdventError, match="sheet-probe"):
        record._prepare_w07d35(7, 35)

    path = tmp_path / "logs" / "ship" / "w07d35" / "state.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["sheet_probe"]["at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    doc["stages"]["checks"]["status"] = "failed"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AdventError, match="checks"):
        record._prepare_w07d35(7, 35)

    doc["stages"]["checks"]["status"] = "done"
    doc["sheet_probe"]["result"]["ok"] = False
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AdventError, match="sheet-probe"):
        record._prepare_w07d35(7, 35)


def test_prepare_requires_review_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    _write_state(tmp_path)
    path = tmp_path / "logs" / "ship" / "w07d35" / "state.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["stages"]["review"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AdventError):
        record._prepare_w07d35(7, 35)


def test_prepare_requires_the_days_commit_at_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    _write_state(tmp_path)
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "b.py")
    _git(tmp_path, "commit", "-q", "-m", "fix(w07d34): something else")
    with pytest.raises(AdventError, match="HEAD не коммит дня"):
        record._prepare_w07d35(7, 35)


def test_prepare_refuses_dirty_tracked_tree_but_ignores_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)
    _write_state(tmp_path)
    (tmp_path / "AGENTS.md").write_text("n\n", encoding="utf-8")  # untracked: fine
    record._prepare_w07d35(7, 35)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(AdventError, match="незакоммиченные"):
        record._prepare_w07d35(7, 35)


def test_rehearsal_steps_are_free_and_use_adventship() -> None:
    steps = record.rehearsal_steps(7, 35)
    assert [s.title for s in steps] == [
        "репетиция: status",
        "репетиция: нет файла задания",
    ]
    assert {s.module for s in steps} == {"adventship.cli"}
    assert steps[0].args == ["status", *DAY]
    assert steps[0].expect_failure is False
    assert steps[1].args == [
        "run",
        *DAY,
        "--dry-run",
        "--only",
        "intake",
        "--task-file",
        "нет-такого-файла.md",
    ]
    assert steps[1].expect_failure is True
    assert steps[1].stdin_lines == ["Проверка кириллицы через пайп"]
