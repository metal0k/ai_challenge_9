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
