"""Порядок подготовки записи: он несущий, и переставить его молча нельзя.

OBS снимает ОКНО, а не поток вывода, поэтому всё, что осталось на экране от
подготовки, попадает в первые кадры ролика. На записи Day 04 2026-09-03 туда
уехал хвост репетиции пайпа — `/params`, заведомо неверная команда, `/exit`, —
и начало читалось как обрывок чужой сессии.

Лечится очисткой экрана, но ровно в одной точке: ПОСЛЕ verify_capture и ДО
start_recording. Раньше нельзя — verify_capture отбраковывает чёрный кадр, и
на очищенной консоли он забракует исправную конфигурацию. Позже нельзя —
первые кадры уже записаны с мусором. Оба соседних порядка выглядят
работающими и оба ломают ролик молча, поэтому порядок закреплён тестом.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from advent_cli import record as record_mod
from advent_core import console
from advent_core.errors import AdventError
from tools import strategy_bench


@contextmanager
def _noop_context(*args, **kwargs):
    yield


@pytest.fixture(autouse=True)
def restore_record_console(monkeypatch):
    """record() rebuilds global Rich consoles; no test may leak that into the suite."""
    previous_out, previous_err = console.out, console.err
    previous_color = console.os.environ.get("ADVENT_RECORD_COLOR")
    yield
    console.out, console.err = previous_out, previous_err
    if previous_color is None:
        monkeypatch.delenv("ADVENT_RECORD_COLOR", raising=False)
    else:
        monkeypatch.setenv("ADVENT_RECORD_COLOR", previous_color)


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Пишет порядок вызовов подготовки, не трогая ни OBS, ни сеть, ни экран."""
    order: list[str] = []

    def track(name, result=None):
        def _fn(*args, **kwargs):
            order.append(name)
            return result

        return _fn

    monkeypatch.setattr(record_mod, "load_env", track("load_env"))
    monkeypatch.setattr(record_mod, "_play", track("play"))
    monkeypatch.setattr(record_mod, "_run_step", track("rehearsal"))
    monkeypatch.setattr(record_mod, "_deliver", track("deliver"))

    # Файл создаётся заранее: финальная строка record() зовёт target.stat(),
    # и f-строка считается ДО того, как console.note окажется заглушкой.
    target = tmp_path / "0104.mp4"
    target.write_bytes(b"")
    monkeypatch.setattr(record_mod, "_target_path", lambda week, day: target)
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: None)

    monkeypatch.setattr(record_mod.console, "note", lambda *a, **k: None)
    monkeypatch.setattr(record_mod.console, "clear_screen", track("clear_screen"))

    monkeypatch.setattr(record_mod.obs, "connect", track("connect", result=object()))
    monkeypatch.setattr(record_mod.obs, "ensure_scene", track("ensure_scene"))
    monkeypatch.setattr(record_mod.obs, "verify_capture", track("verify_capture"))
    monkeypatch.setattr(record_mod.obs, "start_recording", track("start_recording"))
    monkeypatch.setattr(record_mod.obs, "program_scene", _noop_context)
    monkeypatch.setattr(record_mod.obs, "record_directory", _noop_context)

    raw = tmp_path / "raw.mkv"
    raw.write_bytes(b"")
    monkeypatch.setattr(record_mod.obs, "stop_recording", track("stop_recording", result=raw))

    return order


def test_screen_is_cleared_after_verify_capture_and_before_start_recording(calls):
    record_mod.record(day=4, week=1, dry_run=False, rehearse=True, keep_original=False)

    assert "clear_screen" in calls, "экран не чистится — хвост репетиции уедет в первые кадры"
    cleared = calls.index("clear_screen")

    # verify_capture до очистки: ему нужна непустая картинка, иначе он
    # забракует исправную конфигурацию как чёрный кадр.
    assert calls.index("verify_capture") < cleared
    # start_recording после очистки: иначе мусор уже в файле.
    assert cleared < calls.index("start_recording")


def test_recording_starts_only_after_the_clear_had_time_to_repaint(calls, monkeypatch):
    """Measured, not theoretical: the w02d10 take's first 0.4 s still showed
    the rehearsal tail. The clear is an escape sequence the terminal repaints
    on, and OBS records whatever the window last composited, so the two must
    not be issued back to back."""
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: calls.append("sleep"))

    record_mod.record(day=4, week=1, dry_run=False, rehearse=True, keep_original=False)

    cleared = calls.index("clear_screen")
    assert calls.index("sleep") > cleared, "между очисткой и стартом записи нет паузы"
    assert calls.index("sleep") < calls.index("start_recording")


def test_rehearsal_runs_before_the_screen_is_cleared(calls):
    """Иначе чистить нечего: репетиция сама и печатает то, что мешает."""
    record_mod.record(day=4, week=1, dry_run=False, rehearse=True, keep_original=False)

    assert calls.index("rehearsal") < calls.index("clear_screen")


def test_demo_plays_only_after_recording_started(calls):
    record_mod.record(day=4, week=1, dry_run=False, rehearse=True, keep_original=False)

    assert calls.index("start_recording") < calls.index("play") < calls.index("stop_recording")


def test_dry_run_touches_neither_obs_nor_the_screen(calls):
    """dry-run — прогон для оператора: чистить экран не за чем, OBS не трогаем."""
    record_mod.record(day=4, week=1, dry_run=True, rehearse=True, keep_original=False)

    assert calls == ["load_env", "play"]


def test_no_rehearsal_still_clears_the_screen(calls):
    """--no-rehearse убирает репетицию, но не мусор: на экране остаётся всё,
    что оператор напечатал в этом окне до запуска."""
    record_mod.record(day=4, week=1, dry_run=False, rehearse=False, keep_original=False)

    assert "rehearsal" not in calls
    assert calls.index("verify_capture") < calls.index("clear_screen")
    assert calls.index("clear_screen") < calls.index("start_recording")


def test_clear_screen_wipes_scrollback_not_just_the_visible_screen(capsys):
    """Только 2J недостаточно: Windows Terminal оставит прежние строки в
    буфере прокрутки, они уедут вверх, а не исчезнут, и в кадре это так же
    грязно. 3J чистит буфер, H возвращает курсор в начало."""
    from advent_core import console

    console.clear_screen()
    written = capsys.readouterr()

    assert "\033[3J" in written.err
    assert "\033[2J" in written.err
    assert "\033[H" in written.err
    # Контракт проекта: в stdout только ответ модели.
    assert written.out == ""


def test_target_path_is_named_after_week_and_day(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_DIR", str(tmp_path))

    assert record_mod._target_path(1, 4) == tmp_path / "0104.mp4"
    assert record_mod._target_path(1, 1).name == "0101.mp4"


def test_target_path_refuses_without_video_dir(monkeypatch):
    """Лучше отказ до записи, чем готовый файл, который некуда положить."""
    monkeypatch.delenv("VIDEO_DIR", raising=False)

    with pytest.raises(AdventError):
        record_mod._target_path(1, 4)


def test_every_recorded_day_has_a_scenario():
    """Сценарии прошлых дней остаются в коде: `advent record --day 1` обязана
    продолжать работать после того, как неделя ушла вперёд."""
    for day in (1, 2, 3, 4):
        steps = record_mod.demo_steps(1, day)
        assert steps, f"день {day} остался без сценария"
        assert all(step.args or step.note for step in steps), (
            f"день {day}: шаг без действия и без текста"
        )


def test_day_04_scenario_covers_all_three_axes_of_the_task(monkeypatch):
    """Задание требует сравнения по точности, креативности и разнообразию И
    вывода «для каких задач какая настройка». В первой записи разбор был один
    и только про точность — остальное осталось колонками таблиц, которые
    зритель должен истолковать сам. Метрика, посчитанная, но не названная
    вслух, задание не закрывает, поэтому состав разбора закреплён здесь."""
    titles = " ".join(step.title.lower() for step in record_mod.demo_steps(1, 4))

    assert "точность" in titles
    assert "разнообразие" in titles
    assert "креативность" in titles
    assert "для каких задач" in titles


def test_day_04_does_not_promise_token_decay_it_cannot_reproduce():
    """Шаг про потолок 1.5 заведён ради распада токенов, но распад — лотерея
    (2 прогона из 5 в замере, 0 из 1 на dry-run). Заголовок, обещающий его,
    оказывался враньём про собственный экран — ловушка Day 02."""
    for step in record_mod.demo_steps(1, 4):
        assert "распад" not in step.title.lower()


# --------------------------------------------------------------------------
# Day 05 (SPEC-w01d05.md §16): свой сценарий, дни 01-04 не тронуты — ни один
# из тестов выше не переписан, только добавлены новые ниже.
# --------------------------------------------------------------------------


def test_day_05_has_its_own_scenario_not_the_day_01_fallback():
    """demo_steps() выбирает сценарий по (week, day) через цепочку if/return
    (см. её докстринг) — без ветки на day==5 вызов молча проваливается в
    _demo_steps_w01d01() и `advent record --day 5` записал бы День 01 под
    видом Дня 05. Проверяется содержательно (зовёт `bench`, которого не было
    в дне 01), а не просто «списки разные»."""
    day5 = record_mod.demo_steps(1, 5)
    assert day5 != record_mod.demo_steps(1, 1)
    assert any("bench" in (step.args or []) for step in day5), (
        "сценарий дня 05 обязан звать команду bench"
    )


def test_day_05_scenario_names_latency_tokens_price_and_which_model_aloud():
    """SPEC-w01d05.md §16: демо обязано назвать вслух время, токены, цену и
    «какую модель выбирать» — не оставлять зрителю самому читать колонки
    таблицы. Та же ловушка, что чинил test_day_04_scenario_covers_all_three_
    axes_of_the_task: метрика, посчитанная, но не произнесённая, задание не
    закрывает."""
    text = " ".join(
        f"{step.title} {step.note or ''}" for step in record_mod.demo_steps(1, 5)
    ).lower()

    assert "врем" in text or "latency" in text or " мс" in text, "время ответа не названо"
    assert "токен" in text, "токены не названы"
    assert "цен" in text or "$" in text or "стоимост" in text, "цена не названа"
    assert "какую модель" in text, "вывод «какую модель выбирать» не произнесён"


def test_day_05_scenario_has_no_step_without_an_action_or_a_note():
    """Тот же контроль качества, что test_every_recorded_day_has_a_scenario
    держит для дней 01-04 — здесь не тронут тот тест (дни 01-04 не менялись),
    а дню 05 заведена своя проверка."""
    steps = record_mod.demo_steps(1, 5)
    assert steps, "день 05 остался без сценария"
    assert all(step.args or step.note for step in steps), "день 05: шаг без действия и без текста"


# --------------------------------------------------------------------------
# Week 02, Day 06 (SPEC-w02d06.md §17): три поломки машинерии записи, каждая
# из которых сломала бы день молча. Тесты дней 01-05 выше не тронуты.
# --------------------------------------------------------------------------


def test_demo_steps_raises_on_an_unknown_week_and_day():
    """Раньше здесь стоял безусловный fallback в день 01, и
    `record --week 2 --day 6` молча строил шаги `advent w02 chat` — команды,
    которой нет. Проверка, чей failure path продолжается, хуже отсутствующей:
    она читается как пройденная (CLAUDE.md).

    Пары держим всегда за пределами реализованных сценариев: (2, 7) стоял
    здесь как «несуществующий», и день 7, получив свой сценарий, сломал тест —
    сам по себе он ничего не проверял о дне 7. То же повторилось с (2, 9) в
    день 09 (заменено на (2, 10)) и с (2, 10) в день 10; заменено на (2, 11)."""
    for week, day in ((2, 11), (3, 1), (1, 9)):
        with pytest.raises(AdventError):
            record_mod.demo_steps(week, day)


def test_day_01_scenario_is_still_returned_for_week_1_day_1():
    """Падение на незнакомой паре не должно было задеть день, который раньше
    возвращался тем же fallback'ом."""
    steps = record_mod.demo_steps(1, 1)
    assert steps
    assert any("models" in (step.args or []) for step in steps)


def test_days_01_to_05_still_run_through_the_common_cli():
    """Поле Step.module добавлено с умолчанием ровно ради этого: сценарии
    прошлых дней не меняются ни на символ."""
    for day in (1, 2, 3, 4, 5):
        for step in record_mod.demo_steps(1, day):
            assert step.module == record_mod.DEFAULT_MODULE


def _capture_command(monkeypatch) -> list[list[str]]:
    """Перехватывает запуск subprocess: нужен состав команды, а не запуск."""
    seen: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_run(command, **kwargs):
        seen.append(command)
        return _Result()

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: None)
    return seen


def test_run_step_launches_the_module_named_by_the_step(monkeypatch):
    """`_run_step()` зашивал "advent_cli", и отдельную точку входа
    `adventagent` было физически нечем запустить (SPEC-w02d06.md §17.3)."""
    seen = _capture_command(monkeypatch)

    record_mod._run_step(record_mod.Step(title="общий CLI", args=["w01", "models"]))
    record_mod._run_step(
        record_mod.Step(title="агент", module="week_02.cli", args=["--session", "demo"])
    )

    assert seen[0][1:3] == ["-m", "advent_cli"]
    assert seen[1][1:3] == ["-m", "week_02.cli"]
    assert seen[1][3:] == ["--session", "demo"]


def test_week_02_scenario_runs_the_agent_entry_point(monkeypatch):
    """Ни одного `advent w02 chat`: такой команды не существует — неделя 02
    запускается отдельной точкой входа."""
    steps = record_mod.demo_steps(2, 6)

    assert steps
    assert all(step.module == "week_02.cli" for step in steps if step.args)
    for step in steps:
        assert "w02" not in (step.args or [])
        assert "chat" not in (step.args or [])


def test_week_02_scenario_shows_memory_tokens_dialog_model_change_and_sessions():
    """SPEC-w02d06.md §18: сценарий обязан показать все шесть вещей дня, а не
    оставить их зрителю в виде колонок — та же дисциплина, что закреплена для
    дней 04 и 05 выше."""
    titles = " ".join(step.title.lower() for step in record_mod.demo_steps(2, 6))

    assert "токен" in titles
    assert "помнит" in titles
    assert "диалог" in titles
    assert "смена модели" in titles
    assert "/sessions" in titles


def test_week_02_scenario_starts_from_an_empty_session():
    """Сессия подхватывается с диска: без `/new` первый вопрос уедет вместе с
    разговором прошлого дубля — то же правило, что `/reset` в дне 02."""
    first = record_mod.demo_steps(2, 6)[0]

    assert first.stdin_lines[0] == "/new"
    assert "--session" in first.args


def test_rehearsal_for_week_02_uses_the_agent_and_its_own_session():
    """Репетиция недели 02 зашивала `w02 chat` и уехала бы в ошибку раньше,
    чем проверила пайп и кодировку; демо-сессию она при этом трогать не должна."""
    step = record_mod.rehearsal_step(2)

    assert step.module == "week_02.cli"
    assert "chat" not in step.args
    assert step.args[:2] == ["--session", "rehearsal"]
    assert step.args[1] != record_mod._DEMO_SESSION


def test_rehearsal_for_week_01_is_unchanged():
    step = record_mod.rehearsal_step(1)

    assert step.module == record_mod.DEFAULT_MODULE
    assert step.args == ["w01", "chat"]


# --------------------------------------------------------------------------
# Week 02, Day 08 (SPEC-w02d08.md §7): рост по ходам, override лимита,
# серверное переполнение. Подача гигантского ввода — новое поле Step.stdin_file.
# --------------------------------------------------------------------------


def test_day_08_scenario_starts_from_a_new_demo08_session():
    """Та же дисциплина, что test_week_02_scenario_starts_from_an_empty_session:
    сессия подхватывается с диска, и без /new второй дубль уехал бы в историю
    первого."""
    first = record_mod.demo_steps(2, 8)[0]

    assert first.stdin_lines[0] == "/new"
    assert "--session" in first.args
    assert "demo08" in first.args


def test_day_08_scenario_covers_growth_override_and_server_overflow():
    """SPEC-w02d08.md §7: сценарий обязан показать все три вещи дня — рост
    токенов по ходам, переполнение через клиентский override и серверный 400
    на гигантском вводе — и назвать их вслух (та же дисциплина, что у дней
    04-06: метрика, посчитанная, но не названная, задание не закрывает)."""
    titles = " ".join(step.title.lower() for step in record_mod.demo_steps(2, 8))

    assert "/tokens" in titles
    assert "рост" in titles
    assert "context_limit" in titles
    assert "trim" in titles
    assert "400" in titles


def test_day_08_giant_input_is_a_generator_step_plus_a_stdin_file_step():
    """§7, шаг 4: сначала шаг-генератор пишет файл и печатает точный счёт,
    потом шаг агента подаёт этот файл в stdin. Порядок несущий — 400 без
    показанных чисел читался бы как магия, а не как следствие."""
    steps = record_mod.demo_steps(2, 8)

    generator = next(step for step in steps if step.module == "tools.make_biginput")
    assert generator.stdin_file is None

    last = steps[-1]
    assert last.module == "week_02.cli"
    assert last.stdin_file == record_mod._BIGINPUT_FILE
    assert "/set context_limit default" in last.stdin_lines
    # Гигантская строка — последнее, что уходит в stdin: закрытие за ней и
    # есть выход (EOF), /exit после неё доехать бы не успел.
    assert last.stdin_lines[-1].startswith("/set")


def test_day_08_scenario_has_no_step_without_an_action():
    steps = record_mod.demo_steps(2, 8)
    assert steps, "день 08 остался без сценария"
    # Действие шага — args, подаваемый ввод или сам запуск не дефолтного
    # модуля (шаг-генератор идёт без аргументов: у make_biginput их нет).
    assert all(
        step.args or step.note or step.stdin_file or step.module != record_mod.DEFAULT_MODULE
        for step in steps
    ), "день 08: шаг без действия и без текста"


# --------------------------------------------------------------------------
# Week 02, Day 09 (SPEC-w02d09.md §14): the same dialog with compaction off and
# on, then the offline comparison in numbers.
# --------------------------------------------------------------------------


def test_day_09_plays_the_dialog_live_only_with_compaction_on():
    """The live "compaction off" run was cut on 2026-09-10: the bench's own off
    column shows the same thing, and playing the 12-turn dialog four times in
    one take (twice live, twice in the bench) made the video ten minutes long.

    The `off` half is not gone from the day — it moved into the harness, which
    is what the last step asserts."""
    steps = record_mod.demo_steps(2, 9)
    live = steps[0]

    assert live.stdin_lines[0] == "/new"
    assert "/set compact on" in live.stdin_lines
    assert "/set context_limit 2500" in live.stdin_lines
    assert not any("/set compact off" in step.stdin_lines for step in steps), (
        "живой прогон без сжатия вернулся в сценарий — его показывает bench"
    )

    asked = [line for line in live.stdin_lines if not line.startswith("/")]
    assert asked[0].startswith("Запомни кодовое слово")
    assert "кодовое слово" in asked[-1].lower()


def test_day_09_keeps_the_dialog_and_the_summary_in_one_session():
    """`/summary` after a restart only means something if it reads the session
    the dialog just filled."""
    live, restart = record_mod.demo_steps(2, 9)[:2]

    assert live.args == restart.args
    assert restart.stdin_lines[0] == "/summary"


def test_day_09_names_compaction_the_summary_and_the_numbers():
    """Same discipline as days 04-08: what the day claims has to be said out
    loud in the titles, not left to the viewer to infer from output."""
    titles = " ".join(step.title.lower() for step in record_mod.demo_steps(2, 9))

    assert "сжат" in titles
    assert "пересказ" in titles
    assert "/summary" in titles
    assert "/tokens" in titles


def test_day_09_ends_with_a_bench_run_pinned_to_one_that_measures_something():
    """All four numbers hang together and most combinations show nothing: at the
    real window trim never fires, and at a low window with the default tail trim
    holds history at keep_last, so `older` stays empty and compaction cannot
    fire either. Literals, not the module's constants — an expectation taken
    from the same source as the code under test cannot go red."""
    last = record_mod.demo_steps(2, 9)[-1]

    assert last.module == "tools.compact_bench"
    assert last.args == [
        "--limit",
        "1700",
        "--turns",
        "6",
        "--keep-last",
        "2",
        "--compact-every",
        "2",
    ]


def test_day_09_long_steps_get_more_time_than_the_shared_limit():
    """Twelve turns plus typing pauses in the live step, and both bench arms in
    one process: at STEP_TIMEOUT the take would die on the scenario being long,
    not on a hang. A raised timeout is only ever raised."""
    steps = record_mod.demo_steps(2, 9)

    assert steps[-1].timeout is not None
    assert steps[-1].timeout > record_mod.STEP_TIMEOUT
    assert all(step.timeout is None or step.timeout > record_mod.STEP_TIMEOUT for step in steps), (
        "таймаут шага занижен ниже общего потолка"
    )


def test_day_09_dialog_holds_the_screen_longer_than_the_shared_pause():
    """The answer naming the codeword is the day's headline and it is ONE line;
    the very next input (/tokens) prints a twenty-line table over it. Measured
    on frames of the first w02d09 take: readable for ~0.4 s. The step therefore
    asks for a longer hold than every other day gets."""
    live = record_mod.demo_steps(2, 9)[0]

    assert live.line_pause is not None
    assert live.line_pause > record_mod.STEP_PAUSE


def test_day_09_scenario_has_no_step_without_an_action():
    steps = record_mod.demo_steps(2, 9)
    assert steps, "день 09 остался без сценария"
    assert all(
        step.args or step.note or step.stdin_file or step.module != record_mod.DEFAULT_MODULE
        for step in steps
    ), "день 09: шаг без действия и без текста"


def test_run_step_honours_a_per_step_timeout_without_stdin(monkeypatch):
    """The bench has no stdin, so it takes the subprocess.run branch — which
    carried no timeout at all before day 09 and could hang a take forever."""
    seen: dict = {}

    class _Result:
        returncode = 0

    def fake_run(command, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: None)

    record_mod._run_step(record_mod.Step(title="bench", module="tools.compact_bench", timeout=600))
    assert seen["timeout"] == 600

    record_mod._run_step(record_mod.Step(title="обычный", args=["w01", "models"]))
    assert seen["timeout"] == record_mod.STEP_TIMEOUT


def test_run_step_forces_colour_for_recorded_child(monkeypatch):
    """A pipe must not make Rich downgrade the OBS-facing child to monochrome."""
    seen: dict = {}

    class _Result:
        returncode = 0

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)
    record_mod._run_step(record_mod.Step(title="agent CLI", module="week_02.cli"))

    assert seen["env"]["ADVENT_RECORD_COLOR"] == "1"


def test_run_step_reports_a_hang_on_the_no_stdin_branch(monkeypatch):
    """A hung step must be named as such, not surface as a raw TimeoutExpired."""

    def fake_run(command, **kwargs):
        raise record_mod.subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: None)

    with pytest.raises(record_mod.AdventError) as excinfo:
        record_mod._run_step(record_mod.Step(title="bench", module="tools.compact_bench"))

    assert "завис" in str(excinfo.value)


class _FakeStdin:
    """Принимает записи в список вместо настоящего пайпа."""

    def __init__(self, written: list[str]):
        self._written = written

    def write(self, text: str) -> None:
        self._written.append(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeProcess:
    """Завершается сразу с кодом 0 — тесту нужен состав stdin, не запуск."""

    def __init__(self, written: list[str]):
        self.stdin = _FakeStdin(written)

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _fake_popen(monkeypatch, written: list[str]):
    monkeypatch.setattr(
        record_mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess(written)
    )
    monkeypatch.setattr(record_mod.time, "sleep", lambda _: None)


def test_run_step_writes_stdin_file_after_stdin_lines_as_one_line(monkeypatch, tmp_path):
    """Поле Step.stdin_file (день 08): файл подаётся одной строкой ПОСЛЕ всех
    stdin_lines — порядок важен, ибо в строке нет переводов: \n обрезал бы
    реплику REPL до первой фразы, и переполнение не получилось бы вовсе."""
    (tmp_path / "biginput.txt").write_text("ГИГАНТСКАЯ-СТРОКА-БЕЗ-ПЕРЕВОДОВ", encoding="utf-8")
    written: list[str] = []
    _fake_popen(monkeypatch, written)
    monkeypatch.setattr(record_mod, "PROJECT_ROOT", tmp_path)

    record_mod._run_step(
        record_mod.Step(
            title="гигантский ввод",
            module="week_02.cli",
            args=["--session", "demo08"],
            stdin_lines=["/set context_limit default"],
            stdin_file="biginput.txt",
        )
    )

    assert written == ["/set context_limit default\n", "ГИГАНТСКАЯ-СТРОКА-БЕЗ-ПЕРЕВОДОВ\n"]


def test_run_step_writes_stdin_file_when_there_are_no_stdin_lines(monkeypatch, tmp_path):
    """Условие в _run_step — `not step.stdin_lines and not step.stdin_file` —
    зовёт subprocess.run() только когда ОБА пусты. Оба теста stdin_file выше
    задают ещё и stdin_lines, так что вторая половина условия («и нет
    stdin_file») ни разу не проверялась с пустыми stdin_lines: Step с одним
    только stdin_file обязан всё равно попасть в REPL-ветку (Popen с пайпом),
    а не в subprocess.run() без stdin вовсе."""
    (tmp_path / "biginput.txt").write_text("ТОЛЬКО-ФАЙЛ-БЕЗ-СТРОК", encoding="utf-8")
    written: list[str] = []
    _fake_popen(monkeypatch, written)
    monkeypatch.setattr(record_mod, "PROJECT_ROOT", tmp_path)

    record_mod._run_step(
        record_mod.Step(
            title="только файл",
            module="week_02.cli",
            args=["--session", "demo08"],
            stdin_file="biginput.txt",
        )
    )

    assert written == ["ТОЛЬКО-ФАЙЛ-БЕЗ-СТРОК\n"]


def test_run_step_refuses_a_missing_stdin_file(monkeypatch, tmp_path):
    """Файл пишет шаг-генератор: его отсутствие — поломка сценария, и читать
    её надо по имени, а не traceback'ом FileNotFoundError."""
    written: list[str] = []
    _fake_popen(monkeypatch, written)
    monkeypatch.setattr(record_mod, "PROJECT_ROOT", tmp_path)

    with pytest.raises(AdventError, match="biginput.txt"):
        record_mod._run_step(
            record_mod.Step(
                title="гигантский ввод",
                module="week_02.cli",
                stdin_lines=["/set context_limit default"],
                stdin_file="biginput.txt",
            )
        )


def test_run_step_waits_a_steps_own_pause_instead_of_the_shared_one(monkeypatch):
    """Step.line_pause (день 09) — иначе поле есть, а экран не держится:
    константа в сценарии, которую никто не читает, выглядит как исправление и
    им не является."""
    written: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(
        record_mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess(written)
    )
    monkeypatch.setattr(record_mod.time, "sleep", slept.append)

    record_mod._run_step(
        record_mod.Step(
            title="диалог",
            module="week_02.cli",
            stdin_lines=["привет"],
            line_pause=9.0,
        )
    )

    assert 9.0 in slept
    assert record_mod.STEP_PAUSE not in slept


def test_run_step_keeps_the_shared_pause_when_a_step_asks_for_nothing(monkeypatch):
    """Дни 01-08 не должны заметить нового поля."""
    written: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(
        record_mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess(written)
    )
    monkeypatch.setattr(record_mod.time, "sleep", slept.append)

    record_mod._run_step(
        record_mod.Step(title="диалог", args=["w01", "chat"], stdin_lines=["/exit"])
    )

    assert record_mod.STEP_PAUSE in slept


# --------------------------------------------------------------------------
# Week 02, Day 10 (SPEC-w02d10.md §14): facts живьём, ветвление, bench.
# --------------------------------------------------------------------------


def test_day_10_is_registered_and_has_three_steps():
    """`demo_steps(2, 10)` must resolve to the day-10 scenario, not fall through
    to the "unknown pair" error the dispatcher raises for anything else."""
    steps = record_mod.demo_steps(2, 10)

    assert len(steps) == 3
    assert all(step.args or step.note for step in steps), "день 10: шаг без действия и без текста"


def test_day_10_scenario_uses_its_own_session_not_an_earlier_days():
    """Each day gets its own file because step 1 starts with a destructive
    `/new` — reusing demo/demo08/demo09 would wipe someone else's take."""
    steps = record_mod.demo_steps(2, 10)
    used_names = {record_mod._DEMO_SESSION_D10}
    earlier = {
        record_mod._DEMO_SESSION,
        record_mod._DEMO_SESSION_OTHER,
        record_mod._DEMO_SESSION_D08,
        record_mod._DEMO_SESSION_D09,
    }

    assert record_mod._DEMO_SESSION_D10 not in earlier
    for step in steps:
        if "--session" in step.args:
            name = step.args[step.args.index("--session") + 1]
            assert name in used_names or name.startswith(f"{record_mod._DEMO_SESSION_D10}--")


def test_day_10_step_1_imports_the_scenario_instead_of_retyping_it():
    """A hand-kept second copy of the six planted-detail turns would drift
    from what tools/strategy_bench.py actually measures in step 3 — the same
    trap day 09's own docstring names for COMPACT_SCENARIO. Checked by
    identity against the harness's own constants, not by re-typing the text
    here and comparing strings (that would only prove the copy is accurate
    TODAY, not that it can't drift tomorrow)."""
    assert record_mod.STRATEGY_SCENARIO is strategy_bench.SCENARIO
    assert record_mod.STRATEGY_HEAD_TURNS == strategy_bench.HEAD_TURNS

    live = record_mod.demo_steps(2, 10)[0]
    expected = list(strategy_bench.SCENARIO[: strategy_bench.HEAD_TURNS])
    turns_in_step = [line for line in live.stdin_lines if not line.startswith("/")]
    assert turns_in_step == expected


def test_day_10_facts_step_replays_the_budget_and_ends_with_facts_and_tokens():
    """SPEC §14 step 1: narrow window, `context_strategy facts`, and the block
    shown via `/facts` after the replay (480 -> 520) — the day's headline."""
    live = record_mod.demo_steps(2, 10)[0]

    assert live.stdin_lines[0] == "/new"
    assert f"/set context_limit {record_mod._DEMO_D10_LIMIT}" in live.stdin_lines
    assert "/strategy facts" in live.stdin_lines
    assert "/facts" in live.stdin_lines
    assert "/tokens" in live.stdin_lines
    assert live.stdin_lines.index("/facts") < live.stdin_lines.index("/exit")


def test_day_10_dialog_step_holds_the_screen_longer_than_the_shared_pause():
    """Same reasoning as day 09's own dialog step: the per-turn facts note and
    the final /facts block are the headline and get scrolled off within
    STEP_PAUSE otherwise."""
    live = record_mod.demo_steps(2, 10)[0]

    assert live.line_pause is not None
    assert live.line_pause > record_mod.STEP_PAUSE


def test_day_10_branching_step_also_holds_the_screen_longer_than_the_shared_pause():
    """C2: the branching step's OWN headline — the reply to "Напомни, какой у
    нас сейчас бюджет?" that proves branch isolation — is its very last line
    before `/exit`. At the shared STEP_PAUSE (2s) an answer landing 2.3-4s
    after the question (day 09's own measurement) is on screen well under a
    second before `/exit` fires. Same fix, same override, as step 1."""
    branching = record_mod.demo_steps(2, 10)[1]

    assert branching.line_pause is not None
    assert branching.line_pause > record_mod.STEP_PAUSE
    assert branching.line_pause == record_mod.demo_steps(2, 10)[0].line_pause


def test_day_10_branching_steps_last_question_precedes_exit():
    """The isolation-proving question has to actually be the last thing asked
    before /exit — otherwise the extra line_pause would hold the WRONG line
    on screen."""
    branching = record_mod.demo_steps(2, 10)[1]
    idx = branching.stdin_lines.index

    assert idx("Напомни, какой у нас сейчас бюджет?") == idx("/exit") - 1


def test_day_10_branching_step_uses_the_exact_command_syntax():
    """Command spelling checked against week_02/cli.py's own handlers
    (`_cmd_checkpoint`, `_cmd_branch`, `_cmd_switch`, `_cmd_branches`), not
    invented — a plausible-looking but wrong flag would fail silently as a
    warning line on camera instead of a test failure here."""
    branching = record_mod.demo_steps(2, 10)[1]

    assert "/checkpoint mvp" in branching.stdin_lines
    assert "/branch cheap" in branching.stdin_lines
    assert "/switch demo10" in branching.stdin_lines
    assert "/branch rich --from mvp" in branching.stdin_lines
    assert "/branches" in branching.stdin_lines
    assert "/switch demo10--cheap" in branching.stdin_lines
    # Order matters: mvp must exist before either branch is cut from it, and
    # the switch back to the root must happen before branching a second time
    # (`/branch rich --from mvp` from inside demo10--cheap would still work,
    # but the transcript SPEC §14 describes switches back to root first).
    idx = branching.stdin_lines.index
    assert idx("/checkpoint mvp") < idx("/branch cheap") < idx("/switch demo10")
    assert idx("/switch demo10") < idx("/branch rich --from mvp") < idx("/branches")
    assert idx("/branches") < idx("/switch demo10--cheap")


def test_day_10_branching_step_plants_a_contradicting_budget_per_branch():
    """Each branch gets its OWN budget figure so the final question
    ("what's our budget now?") can only be answered correctly if facts did
    not leak from one branch into the other."""
    branching = record_mod.demo_steps(2, 10)[1]
    text = " ".join(branching.stdin_lines)

    assert "300" in text
    assert "900" in text
    assert "какой у нас сейчас бюджет" in text.lower()


def test_day_10_bench_step_args_are_a_literal_list():
    """Literal, not the module's own constants — an expectation drawn from
    the same source as the code under test cannot go red (CLAUDE.md)."""
    bench = record_mod.demo_steps(2, 10)[2]

    assert bench.module == "tools.strategy_bench"
    assert bench.args == [
        "--turns",
        "8",
        "--limit",
        "4000",
        "--keep-last",
        "3",
        "--strategies",
        "window,facts",
        "--no-fork",
    ]


def test_day_10_bench_step_turns_are_at_the_harness_floor():
    """`--turns` below strategy_bench.MIN_TURNS is refused by the harness
    itself (parser.error) — the shortened demo run has to sit AT the floor,
    not guess a number that happens to clear it today."""
    bench = record_mod.demo_steps(2, 10)[2]
    turns_value = int(bench.args[bench.args.index("--turns") + 1])

    assert turns_value == strategy_bench.MIN_TURNS


def test_day_10_bench_step_gets_more_time_than_the_shared_limit():
    """Two strategies over the shortened scenario are still sixteen live calls
    plus an extractor call per facts turn — well past STEP_TIMEOUT."""
    bench = record_mod.demo_steps(2, 10)[2]

    assert bench.timeout is not None
    assert bench.timeout > record_mod.STEP_TIMEOUT


def test_day_10_scenario_names_facts_branching_and_bench_aloud():
    """Same discipline as days 04-09: what the day claims has to be said out
    loud in the titles, not left for the viewer to infer from output."""
    titles = " ".join(step.title.lower() for step in record_mod.demo_steps(2, 10))

    assert "facts" in titles
    assert "ветв" in titles or "бюджет" in titles
    assert "bench" in titles or "strategy_bench" in titles


def test_day_10_scenario_has_no_step_without_an_action():
    steps = record_mod.demo_steps(2, 10)
    assert steps, "день 10 остался без сценария"
    assert all(
        step.args or step.note or step.stdin_file or step.module != record_mod.DEFAULT_MODULE
        for step in steps
    ), "день 10: шаг без действия и без текста"


# --------------------------------------------------------------------------
# Week 03, Day 11 (SPEC-w03d11.md §12): interactive memory-layer demo.
# --------------------------------------------------------------------------


def test_day_11_uses_one_interactive_process_for_stateful_story():
    steps = record_mod.demo_steps(3, 11)

    assert len(steps) == 1
    step = steps[0]
    assert step.module == "tools.memory_demo"
    assert step.args == ["--interactive"]
    assert step.stdin_lines == [
        (
            "/turn Подготовь migration с zero downtime; отвечай на русском prose "
            "с English technical terms; code word ORBIT."
        ),
        "/memory",
        "/switch B",
        "/switch A",
        "/pin-conflict",
        "/credential",
        "/new",
        "/storage",
        "/exit",
    ]


def test_day_11_pause_keeps_each_state_visible_without_long_static_tail():
    step = record_mod.demo_steps(3, 11)[0]

    assert step.line_pause == 6.5
    assert step.line_pause < 8.0
    assert step.timeout == 120


def test_day_11_title_names_the_observable_memory_story():
    title = record_mod.demo_steps(3, 11)[0].title.lower()

    for term in (
        "модель памяти агента",
        "short-term",
        "working",
        "long-term",
        "answer",
        "switch",
        "privacy boundary",
        "storage",
    ):
        assert term in title


def test_day_11_live_is_an_explicit_mistral_variant_with_bounded_turns():
    steps = record_mod.demo_steps(3, 11, live=True)
    assert len(steps) == 1
    step = steps[0]
    assert step.module == "week_02.cli"
    assert step.args[:2] == ["--session", "demo11-live-A"]
    assert "ministral-14b-latest" in step.args
    assert step.args[step.args.index("--max-tokens") + 1] == "300"
    assert sum(not line.startswith("/") for line in step.stdin_lines) <= 3
    assert "/strategy memory" in step.stdin_lines
    assert "/memory set long preferences.answer_language" in " ".join(step.stdin_lines)
    assert "/memory set working goal.primary" in " ".join(step.stdin_lines)
    assert "ORBIT" in " ".join(step.stdin_lines)
    assert "/switch demo11-live-B" in step.stdin_lines or "/new demo11-live-B" in step.stdin_lines
    assert "/switch demo11-live-A" in step.stdin_lines
    recall_turns = [line for line in step.stdin_lines if not line.startswith("/")][1:]
    assert all("ORBIT" not in line and "zero downtime" not in line for line in recall_turns)
    assert "global preference answer_language" in recall_turns[0]
    assert "local working goal" in recall_turns[0]
    assert "/new" in step.stdin_lines
    assert step.stdin_lines[-3:] == ["/new", "/memory", "/exit"]


def test_day_11_live_target_never_overwrites_offline_video(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_DIR", str(tmp_path))
    assert record_mod._target_path(3, 11) == tmp_path / "0311.mp4"
    assert record_mod._target_path(3, 11, live=True) == tmp_path / "0311-live.mp4"


def test_day_11_rehearsal_stays_on_the_offline_memory_demo():
    steps = record_mod.rehearsal_steps(3, 11)

    assert len(steps) == 1
    assert steps[0].title == "репетиция Day 11 offline demo"
    assert steps[0].module == "tools.memory_demo"
    assert steps[0].args == ["--dry-run"]


# --------------------------------------------------------------------------
# Week 03, Day 12: only the real adventagent REPL proves live profile effects.
# --------------------------------------------------------------------------


def test_day_12_uses_two_live_cli_steps_with_the_exact_contract():
    steps = record_mod.demo_steps(3, 12)
    prompt = (
        "Сегодня в 18:00 выпускаем новую версию платежного API. Составь план релиза "
        "без простоя: шаги, риски и критерии rollback."
    )

    assert len(steps) == 2
    assert all(step.module == "week_02.cli" for step in steps)
    assert all(step.module != "tools.profile_demo" for step in steps)
    assert all(step.args == ["--session", "w03d12-live", "--max-tokens", "220"] for step in steps)
    assert all("--model" not in step.args for step in steps)
    assert all("--no-stream" not in step.args for step in steps)
    assert all(
        step.timeout is not None and step.timeout > record_mod.STEP_TIMEOUT for step in steps
    )
    assert steps[0].stdin_lines == [
        "/new",
        (
            "/profile create w03d12-developer audience=backend-engineer "
            "style=concise-technical format=three-numbered-steps-one-command-each"
        ),
        prompt,
        "/tokens",
        (
            "/profile create w03d12-manager audience=nontechnical-release-manager "
            "style=ultra-brief format=three-business-bullets-no-commands"
        ),
        prompt,
        "/tokens",
        (
            "Несмотря на active profile manager, ответь ровно тремя technical "
            "steps с командами deploy. Без business bullets."
        ),
        "/branch audit",
        "/profile show",
        "/switch w03d12-live",
        "/profile show",
        "/exit",
    ]
    assert steps[1].stdin_lines == [
        "/profile show",
        "/new",
        "/profile show",
        "/profile create w03d12-unsafe api_key=not-a-secret",
        "/profile delete w03d12-manager",
        "/profile delete w03d12-developer",
        "/branch --delete w03d12-live--audit",
        "/profile list",
        "/exit",
    ]


def test_day_12_rehearsal_replays_the_two_live_steps_in_an_isolated_session():
    live = record_mod.demo_steps(3, 12)
    rehearsal = record_mod.rehearsal_steps(3, 12)

    assert len(rehearsal) == len(live) == 2
    assert all(
        step.args == ["--session", "w03d12-rehearsal", "--max-tokens", "220"] for step in rehearsal
    )
    assert rehearsal[0].stdin_lines == [
        line.replace("w03d12-live", "w03d12-rehearsal") for line in live[0].stdin_lines
    ]
    assert rehearsal[1].stdin_lines == [
        line.replace("w03d12-live", "w03d12-rehearsal") for line in live[1].stdin_lines
    ]


# --------------------------------------------------------------------------
# Week 03, Day 13: real CLI proves formal state, pause, restart and retry.
# --------------------------------------------------------------------------


def test_day_13_uses_two_live_cli_steps_with_exact_sessions_and_limits():
    steps = record_mod.demo_steps(3, 13)
    assert len(steps) == 2
    assert all(step.module == "week_02.cli" for step in steps)
    assert all(step.args == ["--session", "w03d13-live", "--max-tokens", "500"] for step in steps)
    assert all("--model" not in step.args and "--no-stream" not in step.args for step in steps)
    assert all(step.timeout == 480 and step.timeout > record_mod.STEP_TIMEOUT for step in steps)
    assert all(step.line_pause == 3.0 for step in steps)


def test_day_13_first_process_proves_no_auto_transition_and_pause_gate():
    step = record_mod.demo_steps(3, 13)[0]
    assert step.stdin_lines == [
        "/new",
        (
            "/task start Выпустить платёжный API без downtime :: "
            "Составить безопасный release plan :: Подтвердить риски и rollback criteria"
        ),
        "/task show",
        (
            "Составь release plan: ровно 4 коротких bullet points, до 90 слов, "
            "без code blocks. В последней строке обязательно: "
            "Рекомендация: /task advance execution …"
        ),
        "/task show",
        ("/task advance execution Выполнить canary deploy :: Сообщить error rate и latency"),
        (
            "Дай ровно 3 shell commands для canary deploy и 2 metric thresholds, "
            "до 90 слов, без code blocks."
        ),
        "/task pause Ожидаем metrics",
        "Продолжай deploy без ожидания.",
        "/task show",
        "/exit",
    ]
    assert step.stdin_lines.count("/task show") == 3


def test_day_13_second_process_proves_restart_retry_done_and_cleanup():
    step = record_mod.demo_steps(3, 13)[1]
    assert step.stdin_lines == [
        "/task show",
        "/tokens",
        "/task resume",
        "/tokens",
        ("Продолжай с текущего шага: ровно 3 коротких bullet points, до 80 слов, без code blocks."),
        ("/task advance validation Проверить error rate и latency :: Решить, нужен ли rollback"),
        (
            "Error rate вырос до 3%. Ответь ровно 3 коротких bullet points: "
            "stop, rollback, verify; до 70 слов, без code blocks. "
            "В последней строке обязательно: "
            "Рекомендация: /task advance execution …"
        ),
        "/task show",
        ("/task advance execution Выполнить rollback canary :: Подтвердить восстановление metrics"),
        (
            "/task advance validation Повторно проверить metrics :: "
            "Зафиксировать результат validation"
        ),
        "/task complete Release validation прошла, production stable",
        "/task show",
        "/tokens",
        "/task clear",
        "/task show",
        "/exit",
    ]


def test_day_13_rehearsal_replays_live_contract_in_isolated_session():
    live = record_mod.demo_steps(3, 13)
    rehearsal = record_mod.rehearsal_steps(3, 13)
    assert len(rehearsal) == len(live) == 2
    assert all(
        step.args == ["--session", "w03d13-rehearsal", "--max-tokens", "500"] for step in rehearsal
    )
    for rehearsed, recorded in zip(rehearsal, live, strict=True):
        assert rehearsed.stdin_lines == [
            line.replace("w03d13-live", "w03d13-rehearsal") for line in recorded.stdin_lines
        ]


# --------------------------------------------------------------------------
# Week 04, Day 16: MCP connection demo.
# --------------------------------------------------------------------------


def test_day_16_records_success_then_failure_then_raw_frames():
    steps = record_mod.demo_steps(4, 16)
    assert [step.args for step in steps] == [
        ["tools"],
        ["tools", "--server", "no-such-mcp-server"],
        ["tools", "--raw"],
    ]
    assert all(step.module == "week_04.cli" for step in steps)
    assert [step.expect_failure for step in steps] == [False, True, False]
    assert all(step.stdin_lines == [] and step.note is None for step in steps)
    assert all(step.line_pause is not None and step.line_pause >= 5.0 for step in steps)
    assert all(step.timeout == 60 for step in steps)


def test_day_16_titles_are_short_and_numbered():
    titles = [step.title for step in record_mod.demo_steps(4, 16)]
    assert [title[:2] for title in titles] == ["1.", "2.", "3."]
    assert all(len(title) < 100 for title in titles)


def test_week_04_module_is_shown_as_adventmcp_in_the_caption():
    assert record_mod.MODULE_COMMANDS["week_04.cli"] == "adventmcp"


def test_week_04_rehearsal_uses_the_mcp_entry_point_not_a_missing_w04_group():
    steps = record_mod.rehearsal_steps(4, 16)
    assert len(steps) == 1
    assert steps[0].module == "week_04.cli"
    assert steps[0].args == ["tools", "--server", "no-such-mcp-server"]
    assert steps[0].expect_failure is True


# --------------------------------------------------------------------------
# Week 03, Day 15: approval gate before execution and restart continuity.
# --------------------------------------------------------------------------


def test_day_15_records_approval_gate_pause_and_validation_before_done():
    steps = record_mod.demo_steps(3, 15)
    assert len(steps) == 2
    assert all(step.module == "week_02.cli" for step in steps)
    assert all(step.args == ["--session", "w03d15-live", "--max-tokens", "500"] for step in steps)
    first, second = steps
    early_advance = (
        "/task advance execution Выполнить canary deploy :: Сообщить error rate и latency"
    )
    assert first.stdin_lines.index(early_advance) < first.stdin_lines.index("/task approve")
    assert first.stdin_lines.count(early_advance) == 2
    assert "Рекомендация: /task approve" in first.stdin_lines[2]
    assert "/task pause Ожидаем metrics" in first.stdin_lines
    assert second.stdin_lines[:2] == ["/task show", "/task resume"]
    assert any(line.startswith("/task advance validation") for line in second.stdin_lines)
    assert any(line.startswith("/task complete ") for line in second.stdin_lines)


def test_day_15_rehearsal_replays_live_contract_in_isolated_session():
    live = record_mod.demo_steps(3, 15)
    rehearsal = record_mod.rehearsal_steps(3, 15)
    assert all(
        step.args == ["--session", "w03d15-rehearsal", "--max-tokens", "500"] for step in rehearsal
    )
    for rehearsed, recorded in zip(rehearsal, live, strict=True):
        assert rehearsed.stdin_lines == [
            line.replace("w03d15-live", "w03d15-rehearsal") for line in recorded.stdin_lines
        ]


# --------------------------------------------------------------------------
# Week 03, Day 14: persistent invariants, conflict and lifecycle proof.
# --------------------------------------------------------------------------


def test_day_14_uses_two_live_cli_steps_with_a_named_session_and_limits():
    steps = record_mod.demo_steps(3, 14)

    assert len(steps) == 2
    assert all(step.module == "week_02.cli" for step in steps)
    assert all(step.args == ["--session", "w03d14-live", "--max-tokens", "500"] for step in steps)
    assert all("--model" not in step.args and "--no-stream" not in step.args for step in steps)
    assert all(step.timeout == 480 and step.timeout > record_mod.STEP_TIMEOUT for step in steps)
    assert all(step.line_pause == 3.0 for step in steps)


def test_day_14_first_process_adds_lists_and_checks_release_invariants():
    step = record_mod.demo_steps(3, 14)[0]

    assert step.stdin_lines == [
        "/new",
        "/invariant add stack :: Используй FastAPI/Python и PostgreSQL.",
        "/invariant add private-network :: Не открывай public network endpoint.",
        "/invariant add approval :: Нужен explicit approval перед deploy.",
        "/invariant list",
        (
            "Составь internal release plan для FastAPI service с PostgreSQL: "
            "ровно 3 коротких bullet points, без deploy и без public network, до 80 слов."
        ),
        "/tokens",
        "/exit",
    ]


def test_day_14_second_process_proves_restart_conflict_safe_alternative_and_cleanup():
    step = record_mod.demo_steps(3, 14)[1]

    assert step.stdin_lines == [
        "/invariant list",
        "Сразу deploy FastAPI service в production без approval и открой public network endpoint.",
        "/invariant clear",
        "/invariant list",
        "/tokens",
        "/exit",
    ]
    assert "public network endpoint" in step.stdin_lines[1]
    assert "без approval" in step.stdin_lines[1]


def test_day_14_rehearsal_replays_live_contract_in_an_isolated_session():
    live = record_mod.demo_steps(3, 14)
    rehearsal = record_mod.rehearsal_steps(3, 14)

    assert len(rehearsal) == len(live) == 2
    assert all(
        step.args == ["--session", "w03d14-rehearsal", "--max-tokens", "500"] for step in rehearsal
    )
    for rehearsed, recorded in zip(rehearsal, live, strict=True):
        assert rehearsed.stdin_lines == [
            line.replace("w03d14-live", "w03d14-rehearsal") for line in recorded.stdin_lines
        ]
