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
from advent_core.errors import AdventError


@contextmanager
def _noop_context(*args, **kwargs):
    yield


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
    сам по себе он ничего не проверял о дне 7. То же повторилось с (2, 9)
    в день 09; заменено на (2, 10)."""
    for week, day in ((2, 10), (3, 1), (1, 9)):
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
