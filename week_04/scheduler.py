"""Background job scheduler for week 04 day 18: one repo-activity job, JSON state.

MCP-agnostic on purpose (server.py imports this, never the reverse).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_DIR = REPO_ROOT / "logs" / "scheduler"
JOB_FILE = SCHEDULER_DIR / "repo_activity.json"
JOB_ID = "repo_activity"
MIN_INTERVAL_SECONDS = 5
MAX_INTERVAL_SECONDS = 3600
RUNS_MAX = 500
GIT_ACTIVITY_WINDOW = 20
GIT_TIMEOUT = 10.0  # own copy: importing server.py here would be a cycle
DEFAULT_POLL_INTERVAL = 1.0
SUMMARY_COMMITS_MAX = 10
READ_ATTEMPTS = 3
READ_RETRY_PAUSE = 0.05
_read_pause: Callable[[float], None] = time.sleep  # module-level so tests can patch it


@dataclass(slots=True)
class Job:
    job_id: str
    interval_seconds: int
    created_at: str
    updated_at: str
    enabled: bool = True
    last_run_at: str | None = None
    cursor_hash: str | None = None


@dataclass(slots=True)
class Run:
    ts: str
    new_commits: int | None  # None = not measured, never 0
    total_commits: int | None
    commits: list[str] = field(default_factory=list)
    overflow: bool = False
    error: str | None = None
    head_hash: str | None = None  # None = tick failed, do not move the cursor


@dataclass(slots=True)
class SchedulerState:
    job: Job | None
    runs: list[Run] = field(default_factory=list)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


def _parse_ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _try_parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return _parse_ts(value)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_state(path: Path | None = None) -> SchedulerState:
    # Sentinel: JOB_FILE is read at call time so monkeypatching it works.
    target = path if path is not None else JOB_FILE
    for attempt in range(READ_ATTEMPTS):
        try:
            text = target.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            return SchedulerState(job=None)
        except OSError:
            # Transient (Windows PermissionError racing os.replace) is not corruption.
            if attempt == READ_ATTEMPTS - 1:
                raise
            _read_pause(READ_RETRY_PAUSE)
    try:
        raw = json.loads(text)
        job = _from_dict(Job, raw["job"]) if raw.get("job") else None
        runs = [_from_dict(Run, r) for r in raw.get("runs", [])]
    except (ValueError, KeyError, TypeError, AttributeError):
        print("файл состояния планировщика повреждён, начинаю с пустого", file=sys.stderr)
        return SchedulerState(job=None)
    return SchedulerState(job=job, runs=runs)


def save_state(state: SchedulerState, path: Path | None = None) -> None:
    target = path if path is not None else JOB_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": asdict(state.job) if state.job is not None else None,
        "runs": [asdict(r) for r in state.runs],
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def upsert_job(
    interval_seconds: int, *, now: Callable[[], float] = time.time, path: Path | None = None
) -> Job:
    state = load_state(path)
    stamp = _iso(now())
    if state.job is None:
        job = Job(
            job_id=JOB_ID,
            interval_seconds=interval_seconds,
            created_at=stamp,
            updated_at=stamp,
        )
    else:
        job = replace(state.job, interval_seconds=interval_seconds, updated_at=stamp)
    save_state(SchedulerState(job=job, runs=state.runs), path)
    return job


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT,
        check=False,
    )


def _failed(ts: str, message: str) -> Run:
    return Run(ts=ts, new_commits=None, total_commits=None, error=message, head_hash=None)


def run_tick(job: Job, *, root: Path = REPO_ROOT, now: Callable[[], float] = time.time) -> Run:
    ts = _iso(now())
    try:
        count = _git(["rev-list", "--count", "HEAD"], root)
        if count.returncode != 0:
            return _failed(ts, count.stderr.strip() or "git rev-list завершился с ошибкой")
        log = _git(
            [
                "log",
                f"-{GIT_ACTIVITY_WINDOW}",
                "--pretty=format:%H  %h  %ad  %an  %s",
                "--date=short",
            ],
            root,
        )
        if log.returncode != 0:
            return _failed(ts, log.stderr.strip() or "git log завершился с ошибкой")
    except (OSError, subprocess.SubprocessError) as exc:
        return _failed(ts, str(exc))
    try:
        total = int(count.stdout.strip())
    except ValueError:
        return _failed(ts, f"неожиданный вывод git rev-list: {count.stdout.strip()!r}")

    entries = [line.split("  ", 1) for line in log.stdout.splitlines() if line.strip()]
    if not entries:
        return _failed(ts, "git log не вернул коммитов")
    head = entries[0][0]

    if job.cursor_hash is None:
        return Run(ts=ts, new_commits=0, total_commits=total, head_hash=head)

    new: list[str] = []
    for full, short in ((e[0], e[1] if len(e) > 1 else "") for e in entries):
        if full == job.cursor_hash:
            return Run(
                ts=ts, new_commits=len(new), total_commits=total, commits=new, head_hash=head
            )
        new.append(short)
    return Run(
        ts=ts,
        new_commits=GIT_ACTIVITY_WINDOW,
        total_commits=total,
        commits=new,
        overflow=True,
        head_hash=head,
    )


def record_tick(state: SchedulerState, run: Run, job: Job) -> SchedulerState:
    runs = [*state.runs, run][-RUNS_MAX:]
    return SchedulerState(job=job, runs=runs)


def summarize(
    state: SchedulerState, *, minutes: int | None = None, now: Callable[[], float] = time.time
) -> str:
    job = state.job
    if job is None:
        return "job ещё не создан — сначала `schedule_job`"
    runs = state.runs
    if minutes is not None:
        cutoff = now() - minutes * 60
        runs = [r for r in runs if _parse_ts(r.ts) >= cutoff]
    if not runs:
        if minutes is not None and state.runs:
            return (
                f"тиков за последние {minutes} мин не найдено; последний тик — {state.runs[-1].ts}."
            )
        return (
            f"job {JOB_ID} создан {job.created_at}, интервал {job.interval_seconds} с — "
            "тиков ещё не было."
        )

    ok = [r for r in runs if r.error is None]
    failed = len(runs) - len(ok)
    new_total = sum(r.new_commits or 0 for r in ok)
    overflow = any(r.overflow for r in ok)
    last_total = next((r.total_commits for r in reversed(ok) if r.total_commits is not None), None)

    seen: set[str] = set()
    recent: list[str] = []
    for r in reversed(runs):
        for line in r.commits:
            if line not in seen:
                seen.add(line)
                recent.append(line)
    recent = recent[:SUMMARY_COMMITS_MAX]

    lines = [
        f"job {JOB_ID}: интервал {job.interval_seconds} с, создан {job.created_at}",
        f"тиков в окне: {len(runs)} ({runs[0].ts} — {runs[-1].ts})",
        f"новых коммитов: {new_total}{'+' if overflow else ''}",
    ]
    if overflow:
        lines.append("есть тики с overflow — точность занижена")
    if failed:
        lines.append(f"тиков с ошибкой: {failed}")
    if last_total is not None:
        lines.append(f"всего коммитов в репозитории: {last_total}")
    if recent:
        lines.append("последние новые коммиты:")
        lines.extend(f"  {c}" for c in recent)
    return "\n".join(lines)


def run_once_check(
    *,
    root: Path = REPO_ROOT,
    path: Path | None = None,
    now: Callable[[], float] = time.time,
    on_tick: Callable[[Run], None] | None = None,
) -> Run | None:
    job = load_state(path).job
    if job is None or not job.enabled:
        return None
    # Unparsable timestamps (hand-edited file) fall back, and finally mean "due".
    baseline = _try_parse_ts(job.last_run_at)
    if baseline is None:
        baseline = _try_parse_ts(job.created_at)
    if baseline is not None and now() - baseline < job.interval_seconds:
        return None
    run = run_tick(job, root=root, now=now)
    # git is slow: reload so a concurrent schedule_job change is not overwritten.
    fresh = load_state(path)
    if fresh.job is not None:
        cursor = fresh.job.cursor_hash if run.error is not None else run.head_hash
        merged = replace(fresh.job, last_run_at=_iso(now()), cursor_hash=cursor)
        save_state(record_tick(fresh, run, merged), path)
    if on_tick is not None:
        on_tick(run)
    return run


def scheduler_loop(
    *,
    root: Path = REPO_ROOT,
    path: Path | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    on_tick: Callable[[Run], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    # State is re-read on every pass: that is how interval changes get picked up.
    while not should_stop():
        try:
            run_once_check(root=root, path=path, now=now, on_tick=on_tick)
        except Exception as exc:  # daemon must outlive one bad iteration
            if on_error is not None:
                on_error(exc)
            else:
                print(f"ошибка итерации планировщика: {exc}", file=sys.stderr)
        sleep(poll_interval)
