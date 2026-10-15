"""Day 34 recording plan and the demo runner's guards; no live calls."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from advent_cli import record
from advent_core.errors import AdventError
from adventassistant import files_demo

RUN_ID = "20261015-101500-0a1b2c3d"
GOAL_1 = "Найди все места в коде, где вызывается `call_tool_once`, и объясни, зачем каждое"
GOAL_2 = "Напиши README недели 7: какие команды есть у adventassistant и что делает каждая"


def test_step_list_and_verbatim_goals() -> None:
    steps = record.demo_steps(7, 34)
    assert len(steps) == 6
    assert steps[0].note is not None and steps[0].args == []
    assert steps[-1].note is not None and steps[-1].line_pause == 8.0

    search, readme, apply_, again = steps[1:5]
    assert search.module == "adventassistant.cli"
    assert search.args == ["files", GOAL_1]
    assert readme.module == "adventassistant.files_demo"
    assert readme.args == ["--run", GOAL_2]
    assert apply_.module == again.module == "adventassistant.files_demo"
    assert apply_.args == ["--apply"]
    assert again.args == ["--apply-again"]
    # run_id is never baked into the plan
    assert not any(RUN_ID[:8] in " ".join(s.args) for s in steps)


def test_model_steps_get_a_per_step_timeout_above_the_shared_one() -> None:
    steps = record.demo_steps(7, 34)
    assert steps[1].timeout is not None and steps[1].timeout > record.STEP_TIMEOUT
    assert steps[2].timeout is not None and steps[2].timeout > record.STEP_TIMEOUT


def test_rehearsal_is_the_free_preflight() -> None:
    steps = record.rehearsal_steps(7, 34)
    assert len(steps) == 1
    assert steps[0].module == "adventassistant.files_demo"
    assert steps[0].args == ["--preflight"]


def test_record_preparation_refuses_a_leftover_readme_and_never_deletes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "week_07").mkdir()
    leftover = tmp_path / "week_07" / "README.md"
    leftover.write_text("rehearsal\n", encoding="utf-8")
    monkeypatch.setattr(record, "PROJECT_ROOT", tmp_path)

    record._prepare_w07d34(7, 33)  # other days are untouched
    with pytest.raises(AdventError, match="уже существует") as info:
        record._prepare_w07d34(7, 34)
    assert "вручную" in (info.value.hint or "")
    assert leftover.read_text(encoding="utf-8") == "rehearsal\n"

    leftover.unlink()
    record._prepare_w07d34(7, 34)


def test_extract_run_id_takes_the_run_line_only() -> None:
    out = f"текст\n\nИзменения:\n+run: fake\n\nrun: {RUN_ID}\nприменить: x files-apply {RUN_ID}\n"
    assert files_demo.extract_run_id(out) == RUN_ID
    assert files_demo.extract_run_id("Изменений нет.\n") is None


def _proc(code: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(files_demo, "RUN_ID_FILE", tmp_path / "run_id.txt")
    monkeypatch.setattr(files_demo, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_run_readme_saves_the_run_id_from_stdout(state: Path, monkeypatch, capsys) -> None:
    calls: list[list[str]] = []

    def fake(args, **_kw):
        calls.append(args)
        return _proc(0, f"Изменения:\n+x\n\nrun: {RUN_ID}\nприменить: y\n")

    monkeypatch.setattr(files_demo, "_cli", fake)
    assert files_demo.run_readme(GOAL_2) == RUN_ID
    assert calls == [["files", GOAL_2]]
    assert (state / "run_id.txt").read_text(encoding="utf-8") == RUN_ID
    assert f"run: {RUN_ID}" in capsys.readouterr().out


def test_run_without_a_diff_fails_and_apply_then_refuses(state: Path, monkeypatch) -> None:
    monkeypatch.setattr(files_demo, "_cli", lambda *a, **k: _proc(0, "Изменений нет.\n"))
    with pytest.raises(AdventError, match="применять нечего"):
        files_demo.run_readme(GOAL_2)
    assert not (state / "run_id.txt").exists()
    with pytest.raises(AdventError, match="run_id предыдущего шага не найден"):
        files_demo.apply_saved()


def test_run_refuses_to_start_over_an_existing_readme(state: Path, monkeypatch) -> None:
    (state / "week_07").mkdir()
    (state / "week_07" / "README.md").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        files_demo, "_cli", lambda *a, **k: pytest.fail("the model must not be called")
    )
    with pytest.raises(AdventError, match="уже существует"):
        files_demo.run_readme(GOAL_2)


def test_apply_runs_files_apply_with_the_saved_id_then_prints_git_status(
    state: Path, monkeypatch, capsys
) -> None:
    (state / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(files_demo, "_cli", lambda args, **_kw: calls.append(args) or _proc(0))
    monkeypatch.setattr(
        files_demo.files_store, "git_status_short", lambda paths, root=None: "?? week_07/README.md"
    )
    files_demo.apply_saved()
    assert calls == [["files-apply", RUN_ID]]
    out = capsys.readouterr().out
    assert "$ git status --short week_07/README.md" in out
    assert "?? week_07/README.md" in out


def test_apply_failure_is_unexpected(state: Path, monkeypatch) -> None:
    (state / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
    monkeypatch.setattr(files_demo, "_cli", lambda *a, **k: _proc(12))
    with pytest.raises(AdventError, match="files-apply завершился с кодом 12"):
        files_demo.apply_saved()


def test_second_apply_exit_12_is_an_expected_outcome(state: Path, monkeypatch, capsys) -> None:
    (state / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
    monkeypatch.setattr(
        files_demo, "_cli", lambda *a, **k: _proc(12, stderr="ОШИБКА: Прогон уже применён: x\n")
    )
    files_demo.apply_again()
    captured = capsys.readouterr()
    assert "Ожидаемый отказ: exit 12" in captured.out
    assert "уже применён" in captured.err


@pytest.mark.parametrize(
    ("code", "stderr"),
    [(0, ""), (1, "ОШИБКА: что-то другое"), (12, "ОШИБКА: конфликт sha256"), (13, "уже применён")],
)
def test_second_apply_with_any_other_outcome_stays_a_failure(
    state: Path, monkeypatch, code: int, stderr: str
) -> None:
    (state / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
    monkeypatch.setattr(files_demo, "_cli", lambda *a, **k: _proc(code, stderr=stderr))
    with pytest.raises(AdventError, match="Повторный files-apply: ждали код 12"):
        files_demo.apply_again()


def test_main_turns_an_unexpected_failure_into_a_nonzero_exit(state: Path) -> None:
    with pytest.raises(SystemExit) as info:
        files_demo.main(["--apply"])  # no saved run id
    assert info.value.code != 0
