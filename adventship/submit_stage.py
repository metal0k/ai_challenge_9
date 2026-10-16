"""Stages `record` and `submit`: video from the pushed tag, public link on Yandex Disk."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import httpx

from advent_cli.submit import build_submission, video_name
from advent_core.config import redact
from adventship.ctx import EXIT_CONFIG, ShipContext, ShipError, StageResult

MIN_DURATION = 5.0
PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources"
RECORD_TIMEOUT = 3600.0


def _video_path(ctx: ShipContext) -> Path:
    video_dir = ctx.env.get("VIDEO_DIR")
    if not video_dir:
        raise ShipError("Не найден VIDEO_DIR в .env", exit_code=EXIT_CONFIG)
    return Path(video_dir) / video_name(ctx.week, ctx.day)


def _git(ctx: ShipContext, *args: str) -> str:
    result = ctx.run(["git", *args], timeout=60)
    if result.returncode != 0:
        raise ShipError(f"git {' '.join(args)}: {redact((result.stderr or '').strip())[:300]}")
    return (result.stdout or "").strip()


def _tag_time(ctx: ShipContext) -> float:
    """When the tag stage finished (state), else the tagged commit's time."""
    entry = ctx.state.entry("tag")
    if entry and entry.get("at"):
        moment = datetime.fromisoformat(entry["at"])
        return moment.timestamp()
    return float(_git(ctx, "log", "-1", "--format=%ct", f"{ctx.tag}^{{commit}}"))


def _duration(ctx: ShipContext, path: Path) -> float:
    result = ctx.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )
    try:
        if result.returncode != 0:
            raise ValueError(result.stderr)
        return float((result.stdout or "").strip())
    except ValueError:
        raise ShipError(
            f"ffprobe не прочитал длительность {path.name}",
            hint=redact((result.stderr or "").strip())[:300] or "Проверь, что ffprobe в PATH.",
        ) from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_problem(ctx: ShipContext, path: Path) -> tuple[str | None, float]:
    """(reason the file is unfit or None, duration)."""
    if not path.exists():
        return f"видео {path.name} не найдено", 0.0
    if path.stat().st_size == 0:
        return f"видео {path.name} пустое", 0.0
    duration = _duration(ctx, path)
    if duration <= MIN_DURATION:
        return f"видео короче {MIN_DURATION:g} с ({duration:.1f} с)", duration
    if path.stat().st_mtime <= _tag_time(ctx):
        return "видео записано раньше тега", duration
    return None, duration


def _record_result(path: Path, duration: float, **extra: object) -> StageResult:
    data = {
        "video": path.name,
        "size": path.stat().st_size,
        "duration": round(duration, 1),
        "video_sha256": _sha256(path),
        **extra,
    }
    return StageResult(data=data, summary=f"{path.name}, {duration:.0f} с")


def run_record(ctx: ShipContext) -> StageResult:
    path = _video_path(ctx)
    problem, duration = _video_problem(ctx, path)
    if problem is None:
        return _record_result(path, duration, skipped="reuse")

    sha = ctx.state.data("tag").get("sha") or _git(ctx, "rev-parse", f"{ctx.tag}^{{commit}}")
    if _git(ctx, "rev-parse", "HEAD") != sha:
        raise ShipError(
            f"HEAD не совпадает с тегом {ctx.tag}: запись только с запушенного тега",
            hint=f"git checkout {ctx.tag} или вернись на коммит тега.",
        )
    if _git(ctx, "status", "--porcelain", "--untracked-files=no"):
        raise ShipError(
            "рабочее дерево не чистое: запись только с тега без правок",
            hint="Закоммить или убери изменения и повтори.",
        )

    others = _git(ctx, "ls-files", "--others", "--exclude-standard")
    untracked = len([line for line in others.splitlines() if line.strip()])

    result = ctx.run(
        ["uv", "run", "advent", "record", "--day", str(ctx.day), "--week", str(ctx.week)],
        timeout=RECORD_TIMEOUT,
    )
    if result.returncode != 0:
        raise ShipError(
            f"advent record завершился с кодом {result.returncode}",
            hint=redact(((result.stderr or "") + (result.stdout or "")).strip()[-400:]),
        )
    problem, duration = _video_problem(ctx, path)
    if problem is not None:
        raise ShipError(f"запись не прошла проверку: {problem}")
    extra = {"warning": f"untracked: {untracked}"} if untracked else {}
    return _record_result(path, duration, **extra)


def recheck_record(ctx: ShipContext) -> bool:
    try:
        path = _video_path(ctx)
        return path.exists() and _sha256(path) == ctx.state.data("record").get("video_sha256")
    except (ShipError, OSError):
        return False


def verify_public_url(url: str, expected_name: str) -> None:
    """Public resource answers 200 and carries the expected file name."""
    try:
        response = httpx.get(PUBLIC_API, params={"public_key": url}, timeout=30.0)
    except httpx.HTTPError as error:
        raise ShipError(
            f"Диск недоступен: {redact(str(error))[:200]}", hint="Повтори позже."
        ) from None
    if response.status_code != 200:
        raise ShipError(f"public_url не отвечает: Диск вернул {response.status_code}")
    try:
        name = response.json().get("name")
    except ValueError:
        name = None
    if name != expected_name:
        raise ShipError(f"public_url ведёт на {name!r}, ожидалось {expected_name!r}")


def run_submit(ctx: ShipContext) -> StageResult:
    submission = build_submission(ctx.week, ctx.day)
    verify_public_url(submission.video_url, video_name(ctx.week, ctx.day))
    return StageResult(
        data={"code_url": submission.code_url, "video_url": submission.video_url},
        summary=submission.video_url,
    )


def recheck_submit(ctx: ShipContext) -> bool:
    url = ctx.state.data("submit").get("video_url")
    if not url:
        return False
    try:
        verify_public_url(url, video_name(ctx.week, ctx.day))
    except ShipError:
        return False
    return True
