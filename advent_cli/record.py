"""`advent record --day 01` — записать демо через OBS.

Сценарий фиксированный: LLM отвечает каждый раз по-разному, а порядок шагов
и то, что показано на экране, — нет. Это делает дубли сравнимыми.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import typer

from advent_cli import obs
from advent_core import console
from advent_core.config import PROJECT_ROOT, load_env
from advent_core.errors import AdventError

STEP_PAUSE = 2.0
TITLE_PAUSE = 1.5
REPL_TYPING_PAUSE = 1.2

# Своя папка под сырую запись OBS: активный профиль пользователя может
# писать в каталог другого проекта.
RAW_DIR = PROJECT_ROOT / "logs" / "obs_raw"

# Заголовок окна, в котором идёт демо. По нему источник выбирает нужный
# терминал, если открыто несколько.
DEMO_WINDOW_TITLE = "AI Advent"


@dataclass(slots=True)
class Step:
    title: str
    args: list[str]
    stdin_lines: list[str] = field(default_factory=list)
    expect_failure: bool = False
    env: dict[str, str] = field(default_factory=dict)


def demo_steps(week: int) -> list[Step]:
    prefix = f"w{week:02d}"
    return [
        Step(
            title="1. Какие модели доступны аккаунту — список приходит из живого API",
            args=[prefix, "models"],
        ),
        Step(
            title="2. Один вопрос — ответ стримится по мере генерации",
            args=[prefix, "chat", "Объясни в двух предложениях, что такое LLM"],
        ),
        Step(
            title="3. Диалог с историей: второй вопрос опирается на первый",
            args=[prefix, "chat"],
            stdin_lines=[
                "Назови три языка программирования для бэкенда",
                "А какой из них быстрее всего учить?",
                "/reset",
                "/exit",
            ],
        ),
        Step(
            title="4. Настройки модели: карточка, параметры и отсев по возможностям",
            args=[prefix, "chat"],
            stdin_lines=[
                "/model info",
                "/params",
                "/set temperature 0",
                "/set reasoning_effort high",
                "Одним словом: столица Франции?",
                "/model codestral-latest",
                "/exit",
            ],
        ),
        Step(
            title="5. Неверный ключ — понятная ошибка вместо traceback",
            args=[prefix, "chat", "привет"],
            env={"MISTRAL_API_KEY": "invalid_key_for_demo_0000000000"},
            expect_failure=True,
        ),
    ]


def rehearsal_step(week: int) -> Step:
    """Дешёвый прогон той же машинерии, что ведёт демо.

    Гоняет subprocess, пайп stdin с кириллицей и коды возврата — всё, что
    ломалось, — но не тратит токены на генерацию. Кириллица в вводе здесь
    обязательна: именно на ней кодировка пайпа и падала.
    """
    return Step(
        title="репетиция",
        args=[f"w{week:02d}", "chat"],
        stdin_lines=["/params", "/модель-которой-нет", "/exit"],
    )


def record(
    day: int = typer.Option(..., "--day", "-d", help="Номер дня внутри недели."),
    week: int = typer.Option(1, "--week", "-w", help="Номер недели."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Прогнать сценарий без записи."),
    rehearse: bool = typer.Option(
        True, "--rehearse/--no-rehearse", help="Проверить машинерию до старта записи."
    ),
    keep_original: bool = typer.Option(
        False, "--keep-original", help="Не удалять исходный файл OBS."
    ),
) -> None:
    """Записать демо дня и положить файл как WWDD.mp4."""
    load_env()
    target = _target_path(week, day)

    if dry_run:
        console.note("dry-run: OBS не задействован")
        _play(demo_steps(week))
        return

    if rehearse:
        # Дешевле поймать поломку здесь, чем обнаружить её в записанном файле.
        console.note("репетиция: проверяю запуск, пайп и кодировку…")
        _run_step(rehearsal_step(week), pause=0.1)
        console.note("репетиция прошла")

    client = obs.connect()

    # Сцена настраивается и проверяется УЖЕ будучи активной: window capture
    # обновляет текстуру только пока источник показывается, поэтому probe вне
    # program scene вернул бы чёрный кадр на исправной конфигурации.
    with obs.record_directory(client, RAW_DIR), obs.program_scene(client, obs.SCENE_NAME):
        obs.ensure_scene(client, prefer_title=DEMO_WINDOW_TITLE)
        obs.verify_capture(client)
        console.note(f"сцена «{obs.SCENE_NAME}» активна, источник даёт картинку")

        obs.start_recording(client)
        console.note("запись пошла")
        time.sleep(TITLE_PAUSE)
        try:
            _play(demo_steps(week))
        finally:
            time.sleep(STEP_PAUSE)
            source = obs.stop_recording(client)

    console.note(f"OBS сохранил {source.name} ({source.suffix})")
    _deliver(source, target, keep_original=keep_original)
    console.note(f"готово: {target} ({target.stat().st_size // 1024 // 1024} МБ)")


def _play(steps: list[Step]) -> None:
    """Прогоняет сценарий в текущем терминале — именно его снимает OBS."""
    for step in steps:
        console.out.print()
        console.out.print(f"[bold cyan]{step.title}[/bold cyan]")
        console.out.print(f"[dim]$ advent {' '.join(step.args)}[/dim]")
        time.sleep(TITLE_PAUSE)
        _run_step(step)
        time.sleep(STEP_PAUSE)


def _run_step(step: Step, pause: float | None = None) -> None:
    typing_pause = REPL_TYPING_PAUSE if pause is None else pause
    step_pause = STEP_PAUSE if pause is None else pause
    env = {**os.environ, **step.env, "PYTHONIOENCODING": "utf-8"}
    command = [sys.executable, "-m", "advent_cli", *step.args]

    if not step.stdin_lines:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
        _check(step, result.returncode)
        return

    # REPL: подаём строки с паузами, чтобы зритель успевал прочитать ответ
    # перед следующим вопросом.
    # encoding задаётся явно: text=True берёт кодировку из локали, а в
    # Windows Terminal это cp1252 — кириллица в stdin ребёнка не проходит.
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        for line in step.stdin_lines:
            time.sleep(typing_pause)
            process.stdin.write(line + "\n")
            process.stdin.flush()
            time.sleep(step_pause)
    except (BrokenPipeError, OSError):
        pass
    finally:
        # Без закрытия stdin дочерний REPL ждёт ввода до самого таймаута.
        with suppress(BrokenPipeError, OSError, ValueError):
            process.stdin.close()

    try:
        code = process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        raise AdventError(f"Шаг «{step.title}» завис и был снят по таймауту.") from None
    _check(step, code)


def _check(step: Step, code: int) -> None:
    if step.expect_failure:
        if code == 0:
            raise AdventError(f"Шаг «{step.title}» должен был упасть, но вернул 0.")
        return
    if code != 0:
        raise AdventError(f"Шаг «{step.title}» завершился с кодом {code}.")


def _target_path(week: int, day: int) -> Path:
    video_dir = os.getenv("VIDEO_DIR")
    if not video_dir:
        raise AdventError("Не найден VIDEO_DIR в .env")
    directory = Path(video_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{week:02d}{day:02d}.mp4"


def _deliver(source: Path, target: Path, *, keep_original: bool) -> None:
    """Кладёт запись как WWDD.mp4, при необходимости меняя контейнер.

    Профиль OBS может писать в mkv. Просто переименовать в .mp4 нельзя —
    получится файл с враньём в расширении, который часть плееров не откроет.
    Поэтому не-mp4 переупаковывается ffmpeg без перекодирования.
    """
    if not source.exists():
        raise AdventError(f"Файл записи не найден: {source}")
    if target.exists():
        console.warn(f"перезаписываю существующий {target.name}")

    if source.suffix.lower() == ".mp4":
        try:
            if keep_original:
                shutil.copy2(source, target)
            else:
                shutil.move(str(source), str(target))
        except PermissionError as exc:
            raise AdventError(
                f"Файл записи занят другим процессом: {source}",
                hint="Скорее всего OBS ещё не отпустил его. Подожди и повтори доставку.",
            ) from exc
        return

    console.note(f"переупаковываю {source.suffix} → .mp4 без перекодирования")
    _remux(source, target)
    if not keep_original:
        source.unlink(missing_ok=True)


def _remux(source: Path, target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AdventError(
            f"OBS записал {source.suffix}, а нужен .mp4 — но ffmpeg не найден.",
            hint="Поставь ffmpeg в PATH либо переключи формат записи OBS на mp4.",
        )
    # Пишем во временный файл рядом с целью: упавший ffmpeg не должен ни
    # удалить уже сданное видео, ни оставить вместо него битый огрызок.
    # Расширение .mp4 сохраняется в имени — ffmpeg выбирает контейнер по нему,
    # и на «0101.mp4.part» он просто не понимает, что от него хотят.
    temporary = target.with_name(f"{target.stem}.part{target.suffix}")
    temporary.unlink(missing_ok=True)

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-c", "copy",
            "-f", "mp4",  # формат задаём явно, не полагаясь на расширение
            "-movflags", "+faststart",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not temporary.exists() or temporary.stat().st_size == 0:
        # Хвост stderr, а не начало: ffmpeg сначала печатает баннер и раскладку
        # потоков, а настоящая причина — последней строкой.
        tail = " | ".join((result.stderr or "").strip().splitlines()[-5:])
        temporary.unlink(missing_ok=True)
        raise AdventError(f"ffmpeg не смог переупаковать запись: {tail or 'без вывода'}")

    temporary.replace(target)
