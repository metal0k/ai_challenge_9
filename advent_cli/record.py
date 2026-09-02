"""`advent record --day N` — записать демо через OBS.

Сценарий фиксированный на каждый день: LLM отвечает каждый раз по-разному,
а порядок шагов и то, что показано на экране, — нет. Это делает дубли
сравнимыми. Сценарии прошлых дней остаются в коде рядом с новыми (см.
`demo_steps`) — старые команды `advent record --day N` продолжают работать.
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
from rich.markup import escape as rich_escape

from advent_cli import obs
from advent_core import console
from advent_core.config import PROJECT_ROOT, load_env
from advent_core.errors import AdventError

# Сценарий дня 03 берёт условие задачи из банка недели, а не из копии в коде.
# Зависимость от week_01 у этого модуля уже есть по существу — сценарии здесь
# понедельные, — и направлена она в одну сторону: week_01 про advent_cli не
# знает, цикла нет.
from week_01 import strategies

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


def demo_steps(week: int, day: int) -> list[Step]:
    """Сценарий демо для конкретного дня недели.

    Сценарии прошлых дней остаются в коде и выбираются по номеру —
    `advent record --day 1` обязана продолжать работать ровно как раньше.
    """
    if week == 1 and day == 3:
        return _demo_steps_w01d03()
    if week == 1 and day == 2:
        return _demo_steps_w01d02()
    return _demo_steps_w01d01(week)


def _demo_steps_w01d01(week: int) -> list[Step]:
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


# Сквозной вопрос дня 02: ложится и в items[{name, amount}] (json/schema), и в
# таблицу Markdown, и в YAML-список, и связан с диалогом про салат из шага 5 —
# видео читается как единый сюжет, а не пять несвязанных прогонов.
_SALAD_QUESTION = "Назови 3 ингредиента для греческого салата с количеством"


def _demo_steps_w01d02() -> list[Step]:
    """Day 02 — формат ответа: переключалки формата, длины и условия завершения.

    Каждый REPL-шаг начинается с `/reset`. Без него первый вопрос шага уходит
    вместе с историей, подхваченной из прошлого запуска (REPL восстанавливает
    её при старте), и «тот же самый запрос» перестаёт быть тем же самым —
    ровно то, что этот день и демонстрирует. На живом прогоне история
    подхватывалась в 10 сообщений.
    """
    return [
        Step(
            title="1. Тот же вопрос без ограничений — базовая линия",
            args=["w01", "chat", _SALAD_QUESTION],
        ),
        # /again работает только на «последний вопрос обычного REPL»
        # (Session.last_question заполняется внутри _repl, не в one-shot) —
        # поэтому здесь и в шаге 3 вопрос идёт первой строкой stdin в REPL,
        # а не позиционным аргументом chat (который вернул бы ответ и вышел
        # раньше, чем прочитались бы команды /set//again).
        Step(
            title="2. format: json → schema → yaml → md через /set + /again",
            args=["w01", "chat"],
            stdin_lines=[
                "/reset",
                _SALAD_QUESTION,
                "/set format json",
                "/again",
                "/set format schema",
                "/set schema_file week_01/schemas/ingredients.json",
                "/again",
                "/set format yaml",
                "/again",
                "/set format md",
                "/again",
                "/exit",
            ],
        ),
        Step(
            title="3. /again ×3 при format=schema — структура одинакова",
            args=["w01", "chat"],
            stdin_lines=[
                "/reset",
                _SALAD_QUESTION,
                "/set format schema",
                "/set schema_file week_01/schemas/ingredients.json",
                "/again",
                "/again",
                "/again",
                "/exit",
            ],
        ),
        # 4а/4б — два одношотовых вызова, не REPL: `/set system` в REPL не
        # существует (system prompt — это только файл, флаг --system или
        # ADVENT_SYSTEM_PROMPT в .env, см. advent_core/config.py), поэтому
        # инструкция «не более 3 предложений» идёт через отдельный файл
        # week_01/prompts/brief_system.md, а не через /set внутри одной сессии.
        # 20, а не 40: на живом прогоне модель укладывает ответ про салат в 34
        # токена и завершает сама — finish=stop, и шаг не показывает ровно то,
        # ради чего он есть. Порог должен быть заведомо ниже длины ответа.
        Step(
            title="4а. max_tokens 20 — обрыв генерации на полуслове, finish=length",
            args=["w01", "chat", _SALAD_QUESTION, "--max-tokens", "20"],
        ),
        Step(
            title="4б. инструкция «не более 3 предложений» — модель сама завершает, finish=stop",
            args=[
                "w01",
                "chat",
                _SALAD_QUESTION,
                "--system",
                "week_01/prompts/brief_system.md",
            ],
        ),
        Step(
            title="5. mode=dialog, done=json:done — уточняющие вопросы про салат",
            args=["w01", "chat"],
            stdin_lines=[
                "/reset",
                "/set mode dialog",
                "/set done json:done",
                "Хочу приготовить салат",
                "Греческий, овощной, без мяса",
                "На двоих, без особых ограничений",
                "/exit",
            ],
        ),
        # Одношотовый вызов по той же причине, что и шаг 4б: маркер завершения
        # задаётся системным промптом из файла, а stop — флагом CLI.
        Step(
            title="6. stop с тем же маркером — ловушка: API вырезает стоп-строку из вывода",
            args=[
                "w01",
                "chat",
                _SALAD_QUESTION,
                "--stop",
                "ГОТОВО",
                "--system",
                "week_01/prompts/stop_marker_system.md",
            ],
        ),
        # Здесь был шаг «модель без structured output отказывает». Он выкинут:
        # на этом аккаунте отказа добиться нечем. Проверены codestral,
        # ministral-3b, voxtral, mistral-code-fim, magistral-medium — все 29
        # chat-моделей принимают json_schema, и у всех 29 стоит
        # function_calling=true. То есть и гейтинг по этому флагу не сработал бы
        # никогда: решение слать response_format всегда — единственное рабочее.
        #
        # Замена сильнее исходного шага: она доказывает, что вердикт формата
        # настоящий, а не печатает «✓» при любом ответе.
        Step(
            title="7а. json + жёсткий потолок длины — JSON рвётся, вердикт это видит",
            args=[
                "w01",
                "chat",
                "Назови 5 ингредиентов для греческого салата с количеством",
                "--format",
                "json",
                "--max-tokens",
                "25",
            ],
        ),
        # Ошибку конфигурации показываем клиентскую: она не зависит ни от сети,
        # ни от того, какие модели доступны аккаунту, поэтому дубль не сорвётся.
        Step(
            title="7б. format=schema без schema_file — понятная ошибка вместо traceback",
            args=["w01", "chat", _SALAD_QUESTION, "--format", "schema"],
            expect_failure=True,
        ),
    ]


# Задача дня 03 задаётся id, а не текстом: условие для шага с `chat` берётся из
# того же файла банка, который решает `solve`. Скопированный в сценарий текст
# рано или поздно разъедется с week_01/problems/alice.json, и видео покажет
# две разные задачи под одним эталоном.
#
# Именно alice, а не более трудная children: на children пошаговое рассуждение
# ошибается так же часто, как прямой ответ (по 5 прогонов: 2/5 и 2/5), и
# сравнивать нечего. На alice разрыв измерен и устойчив — 1/5 против 4/5.
_PROBLEM_ID = "alice"


def _demo_steps_w01d03() -> list[Step]:
    """Day 03 — четыре способа рассуждения над одной задачей.

    Задача (`alice`) подобрана живым перебором: прямой ответ промахивается,
    пошаговый — нет. Поэтому `--problem` задан явно везде, а не берётся из
    флага `"default": true` в банке: если умолчание переедет на другую задачу,
    сценарий должен сломаться заметно, а не тихо снять видео про другое.

    REPL-шагов в этом дне нет: обе поверхности дня (`solve` и
    `chat --strategy`) одношотовые, а стратегия и в REPL строит свои messages
    с нуля, не трогая историю. Правило Day 02 при этом остаётся в силе — если
    шаг с REPL сюда добавится, он обязан начинаться с `/reset`, иначе
    подхваченная при старте история прошлого запуска подмешается в вопрос и
    сравнение способов перестанет быть сравнением.

    Шаг 5 повторяет вызовы шагов 1–4 (девять обращений к API). Сокращать при
    нехватке времени надо промежуточные шаги, а не его: таблица — итог дня.
    """
    problem = strategies.load_problem(_PROBLEM_ID)
    return [
        # Шаги «показать задачу» и «прямой ответ» — один вызов, а не два:
        # `solve` печатает условие, эталон и note перед первым обращением к
        # API (strategies.print_problem), а отдельной команды «показать
        # задачу» в CLI нет.
        #
        # --runs 3, хотя по умолчанию 1: на одном броске прямой ответ угадает
        # верно примерно в одном случае из пяти (замер на alice: 1/5 против
        # 4/5 пошагово), и вся посылка дня развалится прямо в кадре. Три прогона
        # показывают промах как устойчивое свойство способа — колонка эталона
        # печатает k/N.
        Step(
            title="1. Задача, эталон и прямой ответ ×3 — модель промахивается устойчиво",
            args=[
                "w01",
                "solve",
                "--problem",
                _PROBLEM_ID,
                "--strategy",
                "direct",
                "--runs",
                "3",
                # Судья на одной стратегии ранжирует сам себя: лишний вызов
                # ради строки «#1». Он включается на финальном шаге, где ему
                # есть что сравнивать.
                "--no-judge",
            ],
        ),
        # Вторая поверхность дня. Одношотовый chat, а не REPL: истории у
        # стратегии нет по определению, и REPL добавил бы к шагу только
        # приглашение ввода.
        Step(
            title="2. Та же задача через chat --strategy steps — вторая поверхность, ответ верный",
            args=["w01", "chat", "--strategy", "steps", problem.statement],
        ),
        Step(
            title="3. meta: модель сначала пишет промпт для решения, потом решает по нему",
            args=[
                "w01",
                "solve",
                "--problem",
                _PROBLEM_ID,
                "--strategy",
                "meta",
                "--no-judge",
            ],
        ),
        Step(
            title="4. panel: аналитик, инженер и критик решают независимо, четвёртый вызов сводит",
            args=[
                "w01",
                "solve",
                "--problem",
                _PROBLEM_ID,
                "--strategy",
                "panel",
                "--no-judge",
            ],
        ),
        # Финал: все четыре способа подряд, судья и таблица. Судья эталона не
        # видит, поэтому его первое место может не совпасть с колонкой
        # эталона — это результат дня, а не сбой, и вывод так и подписан.
        Step(
            title="5. Полный прогон: четыре способа, судья и таблица сравнения",
            args=["w01", "solve", "--problem", _PROBLEM_ID],
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
        _play(demo_steps(week, day))
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
            _play(demo_steps(week, day))
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
        # rich_escape: с дня 03 в args подставляется условие задачи из банка,
        # а условие пишет человек. Квадратные скобки в нём («последовательности
        # [a, b, c]») Rich съел бы как незакрытый тег — кусок подписи пропал бы
        # прямо в кадре, без единой ошибки.
        console.out.print(f"[dim]$ advent {rich_escape(' '.join(step.args))}[/dim]")
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
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-c",
            "copy",
            "-f",
            "mp4",  # формат задаём явно, не полагаясь на расширение
            "-movflags",
            "+faststart",
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
