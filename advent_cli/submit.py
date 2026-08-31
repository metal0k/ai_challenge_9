"""`advent submit --day 01` — собрать текст комментария для таблицы курса."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

from advent_cli.links import (
    create_remote_tag,
    disk_path,
    publish_and_get_url,
    remote_tag_exists,
    tag_url,
)
from advent_core import console
from advent_core.config import PROJECT_ROOT, load_env
from advent_core.errors import AdventError


def day_tag(week: int, day: int) -> str:
    return f"w{week:02d}d{day:02d}"


def video_name(week: int, day: int) -> str:
    return f"{week:02d}{day:02d}.mp4"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise AdventError(f"git {' '.join(args)} завершился с ошибкой: {result.stderr.strip()}")
    return result.stdout.strip()


def submit(
    day: int = typer.Option(..., "--day", "-d", help="Номер дня внутри недели."),
    week: int = typer.Option(1, "--week", "-w", help="Номер недели."),
    video: str | None = typer.Option(None, "--video", help="Готовая ссылка на видео."),
    no_publish: bool = typer.Option(
        False, "--no-publish", help="Не публиковать видео на Яндекс.Диске."
    ),
    create_tag: bool = typer.Option(
        False, "--create-tag", help="Создать тег на remote, если его нет."
    ),
) -> None:
    """Проверяет тег и видео, публикует файл и печатает готовый комментарий."""
    load_env()

    github_url = os.getenv("GITHUB_URL")
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_url:
        raise AdventError("Не найден GITHUB_URL в .env")

    tag = day_tag(week, day)

    # --- 1. тег на remote ---
    if github_token:
        if remote_tag_exists(github_url, github_token, tag):
            console.note(f"тег {tag} на remote есть")
        elif create_tag:
            sha = _git("rev-parse", "HEAD")
            create_remote_tag(github_url, github_token, tag, sha)
            console.note(f"тег {tag} создан на {sha[:8]}")
        else:
            raise AdventError(
                f"Тега {tag} нет на remote — ссылка в таблице отдаст 404.",
                hint=(
                    f"Запушь его: git tag {tag} && git push origin {tag}\n"
                    f"Либо повтори с флагом --create-tag."
                ),
            )
    else:
        console.warn("GITHUB_TOKEN не задан — наличие тега на remote не проверено")

    code_url = tag_url(github_url, tag)

    # --- 2. видео ---
    video_url = video
    if video_url is None:
        if no_publish:
            raise AdventError(
                "Публикация отключена, а ссылка не передана.",
                hint="Добавь --video <url> или убери --no-publish.",
            )

        video_file = _locate_video(week, day)
        console.note(f"видео: {video_file} ({video_file.stat().st_size // 1024} КБ)")

        token = os.getenv("YANDEX_DISK_TOKEN")
        if not token:
            raise AdventError(
                "Не найден YANDEX_DISK_TOKEN — не могу получить ссылку на видео.",
                hint=(
                    "Получи OAuth-токен со scopes cloud_api:disk.read и cloud_api:disk.write,\n"
                    "либо передай ссылку руками: advent submit --day "
                    f"{day:02d} --video <url>"
                ),
            )
        remote = disk_path(video_file)
        console.note(f"публикую {remote}")
        video_url = publish_and_get_url(token, remote)

    # --- 3. готовый текст ---
    console.out.print()
    console.out.print("[bold]Комментарий для таблицы:[/bold]")
    console.out.print()
    console.out.print(f"Week {week:02d}, Day {day:02d}")
    console.out.print(f"Код: {code_url}")
    console.out.print(f"Видео: {video_url}")


def _locate_video(week: int, day: int) -> Path:
    video_dir = os.getenv("VIDEO_DIR")
    if not video_dir:
        raise AdventError("Не найден VIDEO_DIR в .env")

    path = Path(video_dir) / video_name(week, day)
    if not path.exists():
        raise AdventError(
            f"Видео не найдено: {path}",
            hint=f"Запиши его: advent record --day {day:02d}",
        )
    if path.stat().st_size == 0:
        raise AdventError(f"Файл видео пустой: {path}")
    return path
