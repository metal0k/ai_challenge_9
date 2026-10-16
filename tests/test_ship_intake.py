import pytest

from adventship.ctx import ShipError
from adventship.intake import default_task_file, load_task, parse_day_section

DOC = """# Week

## Day 34

старое задание

## Day 35

Заголовок дня
- пункт

## Day 36

следующее
"""


def test_extracts_exact_section():
    section = parse_day_section(DOC, 35)
    assert section.startswith("## Day 35")
    assert "пункт" in section
    assert "старое" not in section
    assert "следующее" not in section


def test_last_section_runs_to_eof():
    assert parse_day_section(DOC, 36).endswith("следующее")


def test_crlf_normalised():
    assert "\r" not in parse_day_section(DOC.replace("\n", "\r\n"), 35)


def test_day_3_does_not_match_day_35():
    assert parse_day_section(DOC, 3) is None


def test_missing_day_is_none():
    assert parse_day_section(DOC, 99) is None


def test_load_task_default_path(tmp_path):
    path = default_task_file(tmp_path, 7)
    path.parent.mkdir()
    path.write_text(DOC, encoding="utf-8")
    task = load_task(tmp_path, 7, 35)
    assert task.source == "week_07/task_07.md"
    assert task.title == "## Day 35"
    assert len(task.task_hash) == 64


def test_task_hash_follows_text(tmp_path):
    path = tmp_path / "t.md"
    path.write_text(DOC, encoding="utf-8")
    first = load_task(tmp_path, 7, 35, path).task_hash
    path.write_text(DOC.replace("пункт", "другой"), encoding="utf-8")
    assert load_task(tmp_path, 7, 35, path).task_hash != first


def test_task_file_outside_root_shows_name_only(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "t.md").write_text(DOC, encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    assert load_task(root, 7, 35, outside / "t.md").source == "t.md"


def test_missing_file_exit_2(tmp_path):
    with pytest.raises(ShipError) as caught:
        load_task(tmp_path, 7, 35)
    assert caught.value.exit_code == 2


def test_missing_section_exit_2(tmp_path):
    path = tmp_path / "t.md"
    path.write_text(DOC, encoding="utf-8")
    with pytest.raises(ShipError) as caught:
        load_task(tmp_path, 7, 99, path)
    assert caught.value.exit_code == 2


def test_run_intake_state_data_has_no_task_text(tmp_path):
    from types import SimpleNamespace

    from adventship.intake import run_intake

    path = tmp_path / "t.md"
    path.write_text(DOC, encoding="utf-8")
    ctx = SimpleNamespace(root=tmp_path, week=7, day=35, task_file=path, task=None)
    result = run_intake(ctx)
    assert set(result.data) == {"task_hash", "week", "day", "source"}
    assert result.data["week"] == 7 and result.data["day"] == 35
    assert ctx.task.text.startswith("## Day 35")  # stages still get the text via ctx.task
