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
    она читается как пройденная (CLAUDE.md)."""
    for week, day in ((2, 7), (3, 1), (1, 9)):
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
