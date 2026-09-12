"""Cuts frozen stretches out of a screen-capture demo video.

A terminal demo has minutes where nothing changes on screen (a subprocess
works silently); the video should keep a short hold of each such moment and
drop the dead wait. Demo tool, not part of the agent — lives in tools/, no
video is re-encoded here without the user asking for it explicitly.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from advent_core import console
from advent_core.errors import AdventError

DEFAULT_MIN_FREEZE = 2.0
DEFAULT_KEEP = 1.5
DEFAULT_NOISE = "-60dB"
# How close to EOF a freeze must end to count as the take's closing screen.
# Generous on purpose: freezedetect reports in frame steps, and the last frame
# lands a fraction short of the container duration.
TAIL_EPS = 0.5

# Matches both freeze_start and freeze_end; freeze_duration is redundant once
# start/end are paired, so it is not captured.
_FREEZE_LINE = re.compile(r"lavfi\.freezedetect\.(freeze_start|freeze_end):\s*([0-9.]+)")

Interval = tuple[float, float]


# --------------------------------------------------------------------------
# Pure logic — no subprocess, fully covered by tests.
# --------------------------------------------------------------------------


def parse_freezedetect(stderr_text: str, duration: float) -> list[Interval]:
    """Extracts (start, end) freeze pairs from freezedetect's stderr log.

    A freeze_start with no matching freeze_end means the freeze runs to EOF
    (ffmpeg never emits the closing line because nothing unfroze) — clamped
    to `duration` rather than dropped, so the tail is still eligible for a cut.
    """
    freezes: list[Interval] = []
    pending_start: float | None = None
    for match in _FREEZE_LINE.finditer(stderr_text):
        kind, value = match.group(1), float(match.group(2))
        if kind == "freeze_start":
            pending_start = value
        elif kind == "freeze_end" and pending_start is not None:
            freezes.append((pending_start, value))
            pending_start = None
    if pending_start is not None:
        freezes.append((pending_start, duration))
    return freezes


def compute_keep_intervals(
    freezes: list[Interval], duration: float, keep: float, *, hold_tail: bool = True
) -> list[Interval]:
    """Turns a freeze list into the intervals to keep in the output.

    Everything outside a freeze is kept whole. Inside a freeze, only the
    first `keep` seconds survive — `min(start + keep, end)` also covers "a
    freeze shorter than keep is left alone" for free, since the minimum picks
    the freeze's own end when it never reaches start + keep.

    The freeze that runs to EOF is the exception (`hold_tail`): it is the
    take's closing screen — the result the viewer is meant to read — and
    cutting it down to `keep` ends the video mid-thought. It stays whole.
    """
    intervals: list[Interval] = []
    cursor = 0.0
    for start, end in freezes:
        if start > cursor:
            intervals.append((cursor, start))
        closes_the_take = hold_tail and end >= duration - TAIL_EPS
        held_end = end if closes_the_take else min(start + keep, end)
        if held_end > start:
            intervals.append((start, held_end))
        cursor = end
    if cursor < duration:
        intervals.append((cursor, duration))
    return intervals


def cut_summary(freezes: list[Interval], intervals: list[Interval], duration: float) -> str:
    """Human report: either 'nothing to cut' or how much was dropped.

    Checked against `intervals`, not just `freezes` being empty: a freeze
    shorter than --keep is detected but never trims anything, and that must
    read the same as no freeze at all, not as a false "cut" claim.
    """
    dropped = duration - sum(end - start for start, end in intervals)
    if not freezes or dropped <= 1e-6:
        return "нечего вырезать: заморозок длиннее --keep не найдено"
    return f"заморозок: {len(freezes)}, вырезано {dropped:.1f} с из {duration:.1f} с"


def _fmt(value: float) -> str:
    """Trims a fixed-point float to the shortest form ffmpeg's expr parser accepts."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _select_expr(intervals: list[Interval]) -> str:
    return "+".join(f"between(t,{_fmt(start)},{_fmt(end)})" for start, end in intervals)


def build_filter_complex(intervals: list[Interval], *, with_audio: bool) -> str:
    """trim/atrim + concat over the kept intervals — one list, both streams.

    Not select/setpts: measured on the w02d10 take (OBS records desktop audio,
    so the file has a second stream), `-vf select` cut the video to 97 s while
    `-af aselect` left the audio at 570 s, and the container then reported the
    ORIGINAL length — a file that looks uncut to every player and to Yandex
    Disk. concat takes both streams from the same interval list, so they cannot
    drift apart.
    """
    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(intervals):
        span = f"start={_fmt(start)}:end={_fmt(end)}"
        parts.append(f"[0:v]trim={span},setpts=PTS-STARTPTS[v{index}]")
        labels.append(f"[v{index}]")
        if with_audio:
            parts.append(f"[0:a]atrim={span},asetpts=PTS-STARTPTS[a{index}]")
            labels.append(f"[a{index}]")
    audio_flag = 1 if with_audio else 0
    tail = "".join(labels) + f"concat=n={len(intervals)}:v=1:a={audio_flag}"
    parts.append(tail + ("[v][a]" if with_audio else "[v]"))
    return ";".join(parts)


# --------------------------------------------------------------------------
# subprocess-backed I/O — exercised only by a real run, never by tests.
# --------------------------------------------------------------------------


def _tail(stderr_text: str) -> str:
    # ffmpeg prints a banner and stream layout first; the real reason is last.
    return " | ".join((stderr_text or "").strip().splitlines()[-5:])


def probe_duration(ffprobe: str, input_path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (result.stdout or "").strip()
    if result.returncode != 0 or not text or text == "N/A":
        # Unknown is never zero: a silent 0.0 would make every freeze look
        # like it runs past the end of a video ffprobe could not read.
        raise AdventError(
            f"ffprobe не смог определить длительность файла: {input_path}",
            hint="Проверь, что это валидный видеофайл.",
        )
    try:
        return float(text)
    except ValueError as exc:
        raise AdventError(f"ffprobe вернул нечисловую длительность: {text!r}") from exc


def has_audio_stream(ffprobe: str, input_path: Path) -> bool:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return bool((result.stdout or "").strip())


def run_freezedetect(ffmpeg: str, input_path: Path, *, noise: str, min_freeze: float) -> str:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-vf",
            f"freezedetect=n={noise}:d={min_freeze}",
            "-map",
            "0:v",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise AdventError(f"ffmpeg freezedetect упал: {_tail(result.stderr) or 'без вывода'}")
    return result.stderr or ""


def run_cut(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    filter_complex: str,
    with_audio: bool,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ]
    command += ["-map", "[a]", "-c:a", "aac"] if with_audio else ["-an"]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    command.append(str(output_path))

    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise AdventError(
            f"ffmpeg не смог собрать результат: {_tail(result.stderr) or 'без вывода'}"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    parser = argparse.ArgumentParser(
        description="Вырезает застывшие участки из демо-видео по freezedetect."
    )
    parser.add_argument("--input", required=True, type=Path, help="исходное видео")
    parser.add_argument("--output", required=True, type=Path, help="куда писать результат")
    parser.add_argument(
        "--min-freeze",
        dest="min_freeze",
        type=float,
        default=DEFAULT_MIN_FREEZE,
        help="минимальная длительность заморозки для freezedetect (по умолчанию 2.0с)",
    )
    parser.add_argument(
        "--keep",
        type=float,
        default=DEFAULT_KEEP,
        help="сколько секунд каждой заморозки оставить (по умолчанию 1.5с)",
    )
    parser.add_argument(
        "--noise", default=DEFAULT_NOISE, help="порог шума freezedetect (по умолчанию -60dB)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="только показать, что будет вырезано"
    )
    args = parser.parse_args(argv)

    if args.min_freeze <= 0 or args.keep <= 0:
        parser.error("--min-freeze и --keep должны быть положительными")
    if args.keep >= args.min_freeze:
        parser.error("--keep должен быть меньше --min-freeze — иначе нечего вырезать")

    try:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise AdventError(
                "ffmpeg/ffprobe не найдены в PATH.",
                hint="Поставь ffmpeg (он несёт и ffprobe) и добавь в PATH.",
            )
        if not args.input.exists():
            raise AdventError(f"Файл не найден: {args.input}")

        duration = probe_duration(ffprobe, args.input)
        stderr_text = run_freezedetect(
            ffmpeg, args.input, noise=args.noise, min_freeze=args.min_freeze
        )
        freezes = parse_freezedetect(stderr_text, duration)
        intervals = compute_keep_intervals(freezes, duration, args.keep)
        summary = cut_summary(freezes, intervals, duration)

        dropped = duration - sum(end - start for start, end in intervals)
        if not freezes or dropped <= 1e-6:
            # Nothing survives to build a filter from — report and stop rather
            # than run ffmpeg over a select expression that changes nothing.
            console.warn(summary)
            return 0

        console.note(summary)
        if args.dry_run:
            for start, end in intervals:
                console.note(f"  оставить {start:.2f}–{end:.2f}с")
            return 0

        with_audio = has_audio_stream(ffprobe, args.input)
        filter_complex = build_filter_complex(intervals, with_audio=with_audio)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run_cut(
            ffmpeg,
            args.input,
            args.output,
            filter_complex=filter_complex,
            with_audio=with_audio,
        )
    except AdventError as exc:
        console.fail(exc)
        return exc.exit_code

    console.out.print(str(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
