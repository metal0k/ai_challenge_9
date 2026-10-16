"""Stage `sheet` against a fake SheetDriver: idempotency, mismatch stop, reconcile, probe gate."""

from __future__ import annotations

import subprocess

import pytest

from adventship import sheet
from adventship.ctx import ShipContext, ShipError
from adventship.sheet_driver import PlaywrightDriver, ProbeResult
from adventship.state import ShipState

SHEET_URL = "https://docs.example.test/spreadsheets/d/SECRET"
CODE = "https://github.com/o/r/tree/w07d35"
VIDEO = "https://disk.example.test/i/abc"


class FakeDriver:
    def __init__(self, comments=None, probe=None, drop_writes=False):
        self.comments = list(comments or [])
        self._probe = probe or ProbeResult(
            ok=True, slot="15", cell="B7", comment_controls=True, comments_before=0
        )
        self.drop_writes = drop_writes
        self.added: list[tuple[str, str]] = []

    def probe(self):
        return self._probe

    def read_comments(self, cell):
        return list(self.comments)

    def add_comment(self, cell, text):
        self.added.append((cell, text))
        if not self.drop_writes:
            self.comments.append(text)


def make_ctx(tmp_path, driver, *, gate=None, now="2026-10-16T10:00:00+03:00"):
    workdir = tmp_path / "ship"
    state = ShipState.load(workdir / "state.json")
    state.mark("submit", "done", at=now, data={"code_url": CODE, "video_url": VIDEO})
    state.mark("describe", "done", at=now, data={"description": "Автоматизация\nсдачи дня"})
    asked: list[tuple[str, str]] = []

    def default_gate(question, details):
        asked.append((question, details))
        return True

    ctx = ShipContext(
        week=7,
        day=35,
        tag="w07d35",
        root=tmp_path,
        workdir=workdir,
        env={"COURSE_SHEET_URL": SHEET_URL, "SHEET_SLOT": "15"},
        state=state,
        run=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""),
        complete=lambda *a, **k: None,
        gate=gate or default_gate,
        driver_factory=lambda: driver,
        now=lambda: now,
    )
    ctx.asked = asked  # type: ignore[attr-defined]
    return ctx


def expected(ctx):
    return sheet.build_comment(ctx)


def test_comment_format_is_four_lines(tmp_path):
    text = expected(make_ctx(tmp_path, FakeDriver()))
    assert text.splitlines() == [
        "Week 07, Day 35",
        f"Код: {CODE}",
        f"Видео: {VIDEO}",
        "Задача: Автоматизация сдачи дня",
    ]


def test_sheet_requires_probe(tmp_path):
    ctx = make_ctx(tmp_path, FakeDriver())
    with pytest.raises(ShipError, match="sheet-probe"):
        sheet.run_sheet(ctx)
    assert ctx.asked == []


def test_stale_probe_is_refused(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    later = make_ctx(tmp_path, driver, now="2026-10-17T10:00:01+03:00")
    with pytest.raises(ShipError, match="24"):
        sheet.run_sheet(later)
    assert driver.added == []


def test_probe_for_other_slot_is_refused(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    ctx.env["SHEET_SLOT"] = "16"
    with pytest.raises(ShipError, match="другой"):
        sheet.run_sheet(ctx)


def test_failed_probe_is_not_usable(tmp_path):
    driver = FakeDriver(probe=ProbeResult(ok=False, slot="15", notes="раскладка неизвестна"))
    ctx = make_ctx(tmp_path, driver)
    with pytest.raises(ShipError):
        sheet.run_probe(ctx)
    with pytest.raises(ShipError, match="не нашёл слот"):
        sheet.run_sheet(ctx)


def test_adds_one_comment_after_gate_and_verifies(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    result = sheet.run_sheet(ctx)
    assert driver.added == [("B7", expected(ctx))]
    assert result.data["cell"] == "B7"
    _, details = ctx.asked[0]
    assert "B7" in details and expected(ctx) in details
    assert "sheet_pending" not in ctx.state.doc
    assert (ctx.workdir / "comment.txt").read_text(encoding="utf-8").strip() == expected(ctx)


def test_second_run_adds_nothing(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    sheet.run_sheet(ctx)
    again = sheet.run_sheet(ctx)
    assert len(driver.added) == 1
    assert again.data["skipped"] == "reuse"
    assert len(ctx.asked) == 1


def test_whitespace_differences_still_match(tmp_path):
    text = expected(make_ctx(tmp_path, FakeDriver()))
    driver = FakeDriver(comments=[text.replace("\n", "  \n ")])
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    assert sheet.run_sheet(ctx).data["skipped"] == "reuse"
    assert driver.added == []


def test_same_day_other_text_stops_without_writing(tmp_path):
    driver = FakeDriver(comments=[f"Week 07, Day 35\nКод: {CODE}\nВидео: https://old.test/v"])
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    with pytest.raises(ShipError, match="другим текстом"):
        sheet.run_sheet(ctx)
    assert driver.added == [] and ctx.asked == []


def test_unrelated_comment_does_not_block(tmp_path):
    driver = FakeDriver(comments=["что-то чужое"])
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    sheet.run_sheet(ctx)
    assert len(driver.added) == 1


def test_interrupted_run_is_reconciled_not_duplicated(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    # The process died after the comment went out but before state was written.
    driver.comments.append(expected(ctx))
    ctx.state.doc["sheet_pending"] = {"cell": "B7", "comment_sha256": "x"}
    ctx.state.save()
    result = sheet.run_sheet(ctx)
    assert driver.added == []
    assert result.data["reconciled"] is True
    assert "sheet_pending" not in ctx.state.doc


def test_verification_failure_raises_and_keeps_pending(tmp_path):
    driver = FakeDriver(drop_writes=True)
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    with pytest.raises(ShipError, match="после отправки"):
        sheet.run_sheet(ctx)
    assert "sheet_pending" in ctx.state.doc


def test_gate_denied_writes_nothing(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver, gate=lambda q, d: False)
    sheet.run_probe(ctx)
    with pytest.raises(ShipError) as caught:
        sheet.run_sheet(ctx)
    assert caught.value.exit_code == 9
    assert driver.added == []


def test_private_data_in_comment_is_refused(tmp_path):
    ctx = make_ctx(tmp_path, FakeDriver())
    ctx.state.mark("submit", "done", at="t", data={"code_url": SHEET_URL, "video_url": VIDEO})
    with pytest.raises(ShipError, match="личные данные"):
        sheet.build_comment(ctx)


def test_long_description_is_cut_to_one_line(tmp_path):
    ctx = make_ctx(tmp_path, FakeDriver())
    ctx.state.mark("describe", "done", at="t", data={"description": "слово " * 100})
    ctx.state.mark("submit", "done", at="t", data={"code_url": CODE, "video_url": VIDEO})
    line = sheet.build_comment(ctx).splitlines()[3]
    assert line.startswith("Задача: ") and len(line) - len("Задача: ") <= 200
    assert line.endswith("...")


def test_recheck_sheet(tmp_path):
    driver = FakeDriver()
    ctx = make_ctx(tmp_path, driver)
    sheet.run_probe(ctx)
    ctx.state.mark("sheet", "done", at="t", data=sheet.run_sheet(ctx).data)
    assert sheet.recheck_sheet(ctx) is True
    driver.comments.clear()
    assert sheet.recheck_sheet(ctx) is False


def test_playwright_driver_refuses_write_without_probed_cell(tmp_path):
    driver = PlaywrightDriver.from_env({"COURSE_SHEET_URL": SHEET_URL}, tmp_path)
    for cell in (None, "", "not a cell"):
        with pytest.raises(ShipError, match="сначала запусти sheet-probe"):
            driver.add_comment(cell, "x")
    assert driver.profile_dir == (tmp_path / "data" / "ship" / "chrome-profile").resolve()


def test_comment_line_from_describe_wins_over_long_description(tmp_path):
    ctx = make_ctx(tmp_path, FakeDriver())
    ctx.state.mark(
        "describe",
        "done",
        at="t",
        data={
            "description": "Длинное описание.\nВторой абзац.",
            "comment_line": "Короткая строка.",
        },
    )
    ctx.state.mark("submit", "done", at="t", data={"code_url": CODE, "video_url": VIDEO})
    assert sheet.build_comment(ctx).splitlines()[3] == "Задача: Короткая строка."


def test_playwright_driver_owner_from_github_url_and_override(tmp_path):
    env = {"COURSE_SHEET_URL": SHEET_URL, "GITHUB_URL": "https://github.com/fakeowner/repo.git"}
    assert PlaywrightDriver.from_env(env, tmp_path, 35).owner == "fakeowner"
    assert PlaywrightDriver.from_env(env, tmp_path, 35).day == 35
    assert PlaywrightDriver.from_env({**env, "SHEET_OWNER": "other"}, tmp_path).owner == "other"


def test_gate_text_names_the_override_file_by_relative_path(tmp_path):
    ctx = make_ctx(tmp_path, FakeDriver())
    sheet.run_probe(ctx)
    sheet.run_sheet(ctx)
    question, details = ctx.asked[0]
    text = question + details
    assert "description.override.txt" in details
    assert "ship/description.override.txt" in details
    assert str(tmp_path) not in text
