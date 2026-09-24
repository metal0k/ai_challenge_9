"""week_04.scheduler: real temp git repo, fake clock, no real sleeping."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from week_04 import scheduler as sch

T0 = 1_800_000_000.0


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def _commit(root: Path, name: str) -> None:
    (root / f"{name}.txt").write_text(name, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", f"commit {name}")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Tester")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, "c1")
    _commit(root, "c2")
    return root


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "job.json"


def _job(cursor: str | None = None) -> sch.Job:
    return sch.Job(
        job_id=sch.JOB_ID,
        interval_seconds=10,
        created_at="2027-01-01T00:00:00+00:00",
        updated_at="2027-01-01T00:00:00+00:00",
        cursor_hash=cursor,
    )


def test_missing_file_is_empty_state(state_path):
    state = sch.load_state(state_path)
    assert state.job is None
    assert state.runs == []


def test_corrupt_file_is_empty_state_and_warns(state_path, capsys):
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{not json", encoding="utf-8")
    state = sch.load_state(state_path)
    assert state.job is None
    assert "повреждён" in capsys.readouterr().err


def test_wrong_shape_file_is_empty_state(state_path):
    state_path.parent.mkdir(parents=True)
    state_path.write_text("[1, 2]", encoding="utf-8")
    assert sch.load_state(state_path).job is None


def test_save_load_roundtrip(state_path):
    run = sch.Run(ts="2027-01-01T00:00:10+00:00", new_commits=2, total_commits=5, commits=["a"])
    sch.save_state(sch.SchedulerState(job=_job("abc"), runs=[run]), state_path)
    loaded = sch.load_state(state_path)
    assert loaded.job == _job("abc")
    assert loaded.runs == [run]


def test_save_leaves_no_tmp_files_and_uses_os_replace(state_path, monkeypatch):
    calls = []
    real = sch.os.replace
    monkeypatch.setattr(sch.os, "replace", lambda a, b: (calls.append((a, b)), real(a, b))[1])
    sch.save_state(sch.SchedulerState(job=_job()), state_path)
    sch.save_state(sch.SchedulerState(job=_job()), state_path)
    assert len(calls) == 2
    assert [p.name for p in state_path.parent.iterdir()] == [state_path.name]


def test_failed_write_cleans_tmp_and_keeps_old_file(state_path, monkeypatch):
    sch.save_state(sch.SchedulerState(job=_job("old")), state_path)

    def boom(a, b):
        raise OSError("nope")

    monkeypatch.setattr(sch.os, "replace", boom)
    with pytest.raises(OSError):
        sch.save_state(sch.SchedulerState(job=_job("new")), state_path)
    assert [p.name for p in state_path.parent.iterdir()] == [state_path.name]
    assert sch.load_state(state_path).job.cursor_hash == "old"


def test_sentinel_path_reads_module_constant_at_call_time(tmp_path, monkeypatch):
    target = tmp_path / "x" / "f.json"
    monkeypatch.setattr(sch, "JOB_FILE", target)
    sch.upsert_job(30, now=Clock().now)
    assert target.exists()
    assert sch.load_state().job.interval_seconds == 30


def test_upsert_creates_then_updates_keeping_cursor(state_path):
    clock = Clock()
    job = sch.upsert_job(15, now=clock.now, path=state_path)
    assert job.cursor_hash is None
    assert job.interval_seconds == 15
    state = sch.load_state(state_path)
    sch.save_state(
        sch.SchedulerState(
            job=sch.replace(state.job, cursor_hash="h", last_run_at="x"),
            runs=[sch.Run(ts="t", new_commits=0, total_commits=1)],
        ),
        state_path,
    )
    clock.t += 100
    updated = sch.upsert_job(60, now=clock.now, path=state_path)
    assert updated.interval_seconds == 60
    assert updated.cursor_hash == "h"
    assert updated.last_run_at == "x"
    assert updated.created_at == job.created_at
    assert updated.updated_at != job.updated_at
    assert len(sch.load_state(state_path).runs) == 1


def test_first_tick_is_baseline(repo):
    run = sch.run_tick(_job(), root=repo)
    assert run.error is None
    assert run.new_commits == 0
    assert run.commits == []
    assert run.total_commits == 2
    assert run.head_hash == _git(repo, "rev-parse", "HEAD").strip()


def test_tick_counts_new_commits_newest_first(repo):
    cursor = _git(repo, "rev-parse", "HEAD").strip()
    _commit(repo, "c3")
    _commit(repo, "c4")
    run = sch.run_tick(_job(cursor), root=repo)
    assert run.new_commits == 2
    assert run.total_commits == 4
    assert not run.overflow
    assert "commit c4" in run.commits[0]
    assert "commit c3" in run.commits[1]
    assert run.commits[0].count("  ") == 3  # short hash, date, author, subject
    assert run.head_hash == _git(repo, "rev-parse", "HEAD").strip()


def test_tick_no_new_commits(repo):
    cursor = _git(repo, "rev-parse", "HEAD").strip()
    run = sch.run_tick(_job(cursor), root=repo)
    assert run.new_commits == 0
    assert run.commits == []


def test_overflow_when_cursor_outside_window(repo, monkeypatch):
    cursor = _git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(sch, "GIT_ACTIVITY_WINDOW", 3)
    for i in range(5):
        _commit(repo, f"n{i}")
    run = sch.run_tick(_job(cursor), root=repo)
    assert run.overflow is True
    assert run.new_commits == 3
    assert len(run.commits) == 3
    assert run.head_hash == _git(repo, "rev-parse", "HEAD").strip()


def test_not_a_repo_is_error_run(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    run = sch.run_tick(_job("abc"), root=plain)
    assert run.error
    assert run.new_commits is None
    assert run.total_commits is None
    assert run.head_hash is None


def test_git_launch_failure_is_error_run(repo, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("git", 10)

    monkeypatch.setattr(sch.subprocess, "run", boom)
    run = sch.run_tick(_job("abc"), root=repo)
    assert run.error
    assert run.head_hash is None
    assert run.new_commits is None


def test_run_once_not_due_returns_none(repo, state_path):
    clock = Clock()
    sch.upsert_job(10, now=clock.now, path=state_path)
    clock.t += 9
    assert sch.run_once_check(root=repo, path=state_path, now=clock.now) is None
    assert sch.load_state(state_path).runs == []


def test_run_once_no_job_returns_none(repo, state_path):
    assert sch.run_once_check(root=repo, path=state_path, now=Clock().now) is None
    assert not state_path.exists()


def test_run_once_disabled_job_returns_none(repo, state_path):
    clock = Clock()
    job = sch.upsert_job(10, now=clock.now, path=state_path)
    sch.save_state(sch.SchedulerState(job=sch.replace(job, enabled=False)), state_path)
    clock.t += 100
    assert sch.run_once_check(root=repo, path=state_path, now=clock.now) is None


def test_run_once_never_sleeps_and_reports_via_on_tick(repo, state_path):
    clock = Clock()
    sch.upsert_job(10, now=clock.now, path=state_path)
    clock.t += 10
    seen = []
    run = sch.run_once_check(root=repo, path=state_path, now=clock.now, on_tick=seen.append)
    assert run is not None
    assert seen == [run]
    assert clock.sleeps == []
    state = sch.load_state(state_path)
    assert state.job.cursor_hash == run.head_hash
    assert state.job.last_run_at is not None


def test_error_run_keeps_cursor_but_advances_last_run(tmp_path, state_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    clock = Clock()
    job = sch.upsert_job(10, now=clock.now, path=state_path)
    sch.save_state(sch.SchedulerState(job=sch.replace(job, cursor_hash="keep")), state_path)
    clock.t += 10
    run = sch.run_once_check(root=plain, path=state_path, now=clock.now)
    assert run.error
    state = sch.load_state(state_path)
    assert state.job.cursor_hash == "keep"
    assert state.job.last_run_at == sch._iso(clock.t)
    assert len(state.runs) == 1


def test_runs_capped_at_runs_max(state_path, monkeypatch):
    monkeypatch.setattr(sch, "RUNS_MAX", 5)
    state = sch.SchedulerState(job=_job())
    for i in range(12):
        state = sch.record_tick(
            state, sch.Run(ts=f"t{i}", new_commits=0, total_commits=1), state.job
        )
    assert [r.ts for r in state.runs] == [f"t{i}" for i in range(7, 12)]


def test_loop_ticks_at_interval_without_real_sleep(repo, state_path):
    clock = Clock()
    sch.upsert_job(10, now=clock.now, path=state_path)
    iterations = {"n": 0}

    def should_stop() -> bool:
        iterations["n"] += 1
        return iterations["n"] > 100

    sch.scheduler_loop(
        root=repo,
        path=state_path,
        now=clock.now,
        sleep=clock.sleep,
        should_stop=should_stop,
        poll_interval=1.0,
    )
    assert len(clock.sleeps) == 100
    runs = sch.load_state(state_path).runs
    assert 9 <= len(runs) <= 10  # 100 s at a 10 s interval


def test_loop_picks_up_interval_change_mid_run(repo, state_path):
    clock = Clock()
    sch.upsert_job(50, now=clock.now, path=state_path)
    ticks: list[float] = []
    iterations = {"n": 0}

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if iterations["n"] == 20:  # like schedule_job called by the agent
            sch.upsert_job(5, now=clock.now, path=state_path)

    def should_stop() -> bool:
        iterations["n"] += 1
        return iterations["n"] > 60

    sch.scheduler_loop(
        root=repo,
        path=state_path,
        now=clock.now,
        sleep=sleep,
        should_stop=should_stop,
        on_tick=lambda run: ticks.append(clock.t),
    )
    before = [t for t in ticks if t < T0 + 20]
    after = [t for t in ticks if t >= T0 + 20]
    assert before == []  # interval 50 s: nothing fired in the first 20 s
    assert len(after) >= 6  # interval 5 s over the remaining ~40 s


def test_idle_loop_without_job_does_nothing(repo, state_path):
    clock = Clock()
    iterations = {"n": 0}

    def should_stop() -> bool:
        iterations["n"] += 1
        return iterations["n"] > 30

    sch.scheduler_loop(
        root=repo,
        path=state_path,
        now=clock.now,
        sleep=clock.sleep,
        should_stop=should_stop,
    )
    assert len(clock.sleeps) == 30
    assert not state_path.exists()


def test_summarize_without_job():
    assert "schedule_job" in sch.summarize(sch.SchedulerState(job=None))


def test_summarize_job_without_runs_differs_from_zero():
    text = sch.summarize(sch.SchedulerState(job=_job()))
    assert "тиков ещё не было" in text
    assert "новых коммитов" not in text


def test_summarize_totals_errors_overflow_and_dedup():
    runs = [
        sch.Run(ts=sch._iso(T0), new_commits=0, total_commits=3),
        sch.Run(ts=sch._iso(T0 + 10), new_commits=2, total_commits=5, commits=["b  x", "a  x"]),
        sch.Run(ts=sch._iso(T0 + 20), new_commits=None, total_commits=None, error="boom"),
        sch.Run(
            ts=sch._iso(T0 + 30),
            new_commits=3,
            total_commits=8,
            commits=["c  x", "b  x"],
            overflow=True,
        ),
    ]
    text = sch.summarize(sch.SchedulerState(job=_job(), runs=runs), now=lambda: T0 + 40)
    assert "тиков в окне: 4" in text
    assert "новых коммитов: 5+" in text
    assert "тиков с ошибкой: 1" in text
    assert "всего коммитов в репозитории: 8" in text
    assert text.count("  b  x") == 1
    assert text.count("  c  x") == 1


def test_summarize_minutes_window():
    runs = [
        sch.Run(ts=sch._iso(T0), new_commits=1, total_commits=2, commits=["old  x"]),
        sch.Run(ts=sch._iso(T0 + 3000), new_commits=1, total_commits=3, commits=["new  x"]),
    ]
    state = sch.SchedulerState(job=_job(), runs=runs)
    text = sch.summarize(state, minutes=5, now=lambda: T0 + 3010)
    assert "тиков в окне: 1" in text
    assert "new  x" in text
    assert "old  x" not in text
    empty = sch.summarize(state, minutes=1, now=lambda: T0 + 9000)
    assert "не найдено" in empty
    assert sch._iso(T0 + 3000) in empty


def test_daemon_save_keeps_interval_changed_mid_tick(repo, state_path, monkeypatch):
    clock = Clock()
    sch.upsert_job(10, now=clock.now, path=state_path)
    clock.t += 10
    real_tick = sch.run_tick

    def slow_tick(job, **kw):
        run = real_tick(job, **kw)
        sch.upsert_job(77, now=clock.now, path=state_path)  # agent calls schedule_job
        return run

    monkeypatch.setattr(sch, "run_tick", slow_tick)
    run = sch.run_once_check(root=repo, path=state_path, now=clock.now)
    state = sch.load_state(state_path)
    assert state.job.interval_seconds == 77
    assert state.job.cursor_hash == run.head_hash
    assert state.job.last_run_at == sch._iso(clock.t)
    assert len(state.runs) == 1


def test_job_deleted_mid_tick_is_not_resurrected(repo, state_path, monkeypatch):
    clock = Clock()
    sch.upsert_job(10, now=clock.now, path=state_path)
    clock.t += 10
    real_tick = sch.run_tick

    def tick_then_wipe(job, **kw):
        run = real_tick(job, **kw)
        sch.save_state(sch.SchedulerState(job=None), state_path)
        return run

    monkeypatch.setattr(sch, "run_tick", tick_then_wipe)
    assert sch.run_once_check(root=repo, path=state_path, now=clock.now) is not None
    assert sch.load_state(state_path).job is None


@pytest.fixture
def pauses(monkeypatch):
    seen: list[float] = []
    monkeypatch.setattr(sch, "_read_pause", seen.append)
    return seen


def _flaky_read(monkeypatch, failures: int):
    real = Path.read_text
    calls = {"n": 0}

    def read_text(self, *a, **k):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise PermissionError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    return calls


def test_transient_permission_error_is_retried_not_corruption(state_path, pauses, monkeypatch):
    sch.save_state(sch.SchedulerState(job=_job("keep")), state_path)
    calls = _flaky_read(monkeypatch, 2)
    assert sch.load_state(state_path).job.cursor_hash == "keep"
    assert calls["n"] == 3
    assert pauses == [sch.READ_RETRY_PAUSE] * 2


def test_persistent_permission_error_raises_and_upsert_keeps_file(state_path, pauses, monkeypatch):
    sch.save_state(sch.SchedulerState(job=_job("keep"), runs=[sch.Run("t", 0, 1)]), state_path)
    before = state_path.read_bytes()
    _flaky_read(monkeypatch, 99)
    with pytest.raises(PermissionError):
        sch.load_state(state_path)
    with pytest.raises(PermissionError):
        sch.upsert_job(30, now=Clock().now, path=state_path)
    monkeypatch.undo()
    assert state_path.read_bytes() == before


def test_loop_survives_a_failing_iteration_and_ticks_next(repo, state_path, monkeypatch):
    clock = Clock()
    sch.upsert_job(5, now=clock.now, path=state_path)
    real = sch.save_state
    calls = {"n": 0}

    def flaky_save(state, path=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("locked")
        real(state, path)

    monkeypatch.setattr(sch, "save_state", flaky_save)
    errors: list[Exception] = []
    iterations = {"n": 0}

    def should_stop() -> bool:
        iterations["n"] += 1
        return iterations["n"] > 12

    sch.scheduler_loop(
        root=repo,
        path=state_path,
        now=clock.now,
        sleep=clock.sleep,
        should_stop=should_stop,
        on_error=errors.append,
    )
    assert len(errors) == 1 and isinstance(errors[0], PermissionError)
    assert len(sch.load_state(state_path).runs) >= 1


def test_loop_default_error_report_is_one_stderr_line(repo, state_path, monkeypatch, capsys):
    monkeypatch.setattr(sch, "run_once_check", lambda **_k: (_ for _ in ()).throw(OSError("x")))
    sch.scheduler_loop(
        path=state_path, sleep=lambda _s: None, should_stop=iter([False, True]).__next__
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_loop_lets_interrupts_propagate(state_path, monkeypatch, exc):
    monkeypatch.setattr(sch, "run_once_check", lambda **_k: (_ for _ in ()).throw(exc()))
    with pytest.raises(exc):
        sch.scheduler_loop(path=state_path, sleep=lambda _s: None)


def test_unparsable_timestamps_do_not_crash_and_job_is_due(repo, state_path):
    clock = Clock()
    job = sch.upsert_job(10, now=clock.now, path=state_path)
    sch.save_state(
        sch.SchedulerState(job=sch.replace(job, last_run_at="garbage", created_at="also bad")),
        state_path,
    )
    run = sch.run_once_check(root=repo, path=state_path, now=clock.now)
    assert run is not None and run.error is None


def test_bad_last_run_at_falls_back_to_created_at(repo, state_path):
    clock = Clock()
    job = sch.upsert_job(10, now=clock.now, path=state_path)
    sch.save_state(sch.SchedulerState(job=sch.replace(job, last_run_at="garbage")), state_path)
    clock.t += 9
    assert sch.run_once_check(root=repo, path=state_path, now=clock.now) is None
    clock.t += 1
    assert sch.run_once_check(root=repo, path=state_path, now=clock.now) is not None
