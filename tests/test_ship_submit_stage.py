"""Stages `record` and `submit` with a fake runner and a fake Disk."""

from __future__ import annotations

import os
import subprocess

import httpx
import pytest

from advent_cli import submit as submit_module
from advent_cli.submit import Submission
from adventship import submit_stage
from adventship.ctx import ShipContext, ShipError
from adventship.state import ShipState

TAG_AT = "2026-10-16T10:00:00+00:00"
TAG_EPOCH = 1792144800.0  # 2026-10-16T10:00:00Z


def make_ctx(tmp_path, runner, **env):
    workdir = tmp_path / "ship"
    state = ShipState.load(workdir / "state.json")
    state.mark("tag", "done", at=TAG_AT, data={"sha": "abc"})
    return ShipContext(
        week=7,
        day=35,
        tag="w07d35",
        root=tmp_path,
        workdir=workdir,
        env={"VIDEO_DIR": str(tmp_path / "videos"), **env},
        state=state,
        run=runner,
        complete=lambda *a, **k: None,
        gate=lambda q, d: True,
        driver_factory=lambda: None,
        now=lambda: TAG_AT,
    )


class Runner:
    def __init__(self, *, head="abc", dirty="", duration="42.5", record_writes=None, record_rc=0):
        self.head, self.dirty, self.duration = head, dirty, duration
        self.record_writes, self.record_rc = record_writes, record_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, timeout, input=None, env=None):
        self.calls.append(argv)
        out = ""
        if argv[:2] == ["git", "rev-parse"]:
            out = self.head
        elif argv[:2] == ["git", "status"]:
            out = self.dirty
        elif argv[0] == "ffprobe":
            out = self.duration
        elif argv[:3] == ["uv", "run", "advent"]:
            if self.record_writes:
                self.record_writes()
            return subprocess.CompletedProcess(argv, self.record_rc, "", "boom")
        return subprocess.CompletedProcess(argv, 0, out + "\n", "")

    def recorded(self):
        return any(c[:4] == ["uv", "run", "advent", "record"] for c in self.calls)


def write_video(tmp_path, *, size=100, mtime=TAG_EPOCH + 60):
    path = tmp_path / "videos" / "0735.mp4"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_existing_good_video_is_reused_without_recording(tmp_path):
    write_video(tmp_path)
    runner = Runner()
    result = submit_stage.run_record(make_ctx(tmp_path, runner))
    assert result.data["skipped"] == "reuse"
    assert result.data["video"] == "0735.mp4" and result.data["duration"] == 42.5
    assert not runner.recorded()


def test_video_older_than_tag_is_not_reused(tmp_path):
    write_video(tmp_path, mtime=TAG_EPOCH - 3600)
    runner = Runner(dirty="M file.py")
    with pytest.raises(ShipError, match="не чистое"):
        submit_stage.run_record(make_ctx(tmp_path, runner))
    assert not runner.recorded()


def test_record_refused_when_head_is_not_the_tag(tmp_path):
    runner = Runner(head="other")
    with pytest.raises(ShipError, match="HEAD не совпадает"):
        submit_stage.run_record(make_ctx(tmp_path, runner))
    assert not runner.recorded()


def test_record_runs_and_verifies_new_video(tmp_path):
    runner = Runner(record_writes=lambda: write_video(tmp_path))
    result = submit_stage.run_record(make_ctx(tmp_path, runner))
    assert runner.recorded()
    assert "skipped" not in result.data and result.data["size"] == 100


def test_record_failure_and_missing_file(tmp_path):
    with pytest.raises(ShipError, match="кодом 1"):
        submit_stage.run_record(make_ctx(tmp_path, Runner(record_rc=1)))
    with pytest.raises(ShipError, match="запись не прошла"):
        submit_stage.run_record(make_ctx(tmp_path, Runner()))


@pytest.mark.parametrize("duration", ["3.0", "5.0"])
def test_short_video_fails_after_record(tmp_path, duration):
    runner = Runner(duration=duration, record_writes=lambda: write_video(tmp_path))
    with pytest.raises(ShipError, match="короче"):
        submit_stage.run_record(make_ctx(tmp_path, runner))


def test_empty_video_fails_after_record(tmp_path):
    runner = Runner(record_writes=lambda: write_video(tmp_path, size=0))
    with pytest.raises(ShipError, match="пустое"):
        submit_stage.run_record(make_ctx(tmp_path, runner))


def fake_get(status=200, name="0735.mp4"):
    seen = {}

    def get(url, **kwargs):
        seen["url"], seen["params"] = url, kwargs.get("params")
        return httpx.Response(status, json={"name": name}, request=httpx.Request("GET", url))

    get.seen = seen
    return get


def patch_submission(monkeypatch):
    monkeypatch.setattr(
        submit_stage,
        "build_submission",
        lambda week, day: Submission("https://c", "https://disk/i/x", "t"),
    )


def test_submit_verifies_public_url(tmp_path, monkeypatch):
    patch_submission(monkeypatch)
    get = fake_get()
    monkeypatch.setattr(submit_stage.httpx, "get", get)
    result = submit_stage.run_submit(make_ctx(tmp_path, Runner()))
    assert result.data == {"code_url": "https://c", "video_url": "https://disk/i/x"}
    assert get.seen["url"].endswith("/v1/disk/public/resources")
    assert get.seen["params"] == {"public_key": "https://disk/i/x"}


@pytest.mark.parametrize(("status", "name"), [(404, "0735.mp4"), (200, "other.mp4")])
def test_submit_rejects_bad_public_url(tmp_path, monkeypatch, status, name):
    patch_submission(monkeypatch)
    monkeypatch.setattr(submit_stage.httpx, "get", fake_get(status, name))
    with pytest.raises(ShipError):
        submit_stage.run_submit(make_ctx(tmp_path, Runner()))


def test_recheck_submit(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, Runner())
    ctx.state.mark("submit", "done", at="t", data={"video_url": "https://disk/i/x"})
    monkeypatch.setattr(submit_stage.httpx, "get", fake_get())
    assert submit_stage.recheck_submit(ctx) is True
    monkeypatch.setattr(submit_stage.httpx, "get", fake_get(404))
    assert submit_stage.recheck_submit(ctx) is False


def test_build_submission_text_and_cli_output(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_URL", "https://github.com/o/r")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = submit_module.build_submission(7, 35, video="https://disk/i/x")
    assert result.comment_text == (
        "Week 07, Day 35\nКод: https://github.com/o/r/tree/w07d35\nВидео: https://disk/i/x"
    )
    monkeypatch.setattr(submit_module, "load_env", lambda: None)
    submit_module.submit(
        day=35, week=7, video="https://disk/i/x", no_publish=False, create_tag=False
    )
    out = capsys.readouterr().out
    assert out.count("Week 07, Day 35") == 1
    assert "Код: https://github.com/o/r/tree/w07d35" in out
    assert "Видео: https://disk/i/x" in out
