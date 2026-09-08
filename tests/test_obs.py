"""Проверка `verify_capture` — защиты от чёрного видео.

Защита ценна ровно настолько, насколько ей можно верить. Ложное срабатывание
стоит того же, что пропущенный чёрный кадр: сорванного дубля. 2026-09-07
запись сорвалась именно на ложном срабатывании — источник был исправен, а
кадр ещё не приехал.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from advent_cli import obs
from advent_core.errors import AdventError


def _shot(brightness: int) -> str:
    """Кадр 480x270 одного тона в том виде, в каком его отдаёт obs-websocket."""
    buffer = io.BytesIO()
    Image.new("L", (480, 270), brightness).save(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Client:
    """Клиент OBS, отдающий заранее заданную последовательность кадров."""

    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)
        self.calls = 0

    def get_source_screenshot(self, *_args, **_kwargs):
        self.calls += 1
        frame = self.frames.pop(0) if self.frames else self.frames_exhausted()
        if isinstance(frame, Exception):
            raise frame
        return type("Shot", (), {"image_data": frame})()

    def frames_exhausted(self):
        raise AssertionError("verify_capture запросил больше кадров, чем задано")


def test_black_frame_that_becomes_bright_is_accepted():
    """Первые кадры чёрные, потом источник просыпается — это НЕ отказ.

    Ровно этот случай сорвал запись 2026-09-07: ensure_scene() только что
    перепривязала источник, WGC ещё не отдал кадр, а проверка судила по
    одному мгновенному снимку.
    """
    client = _Client([_shot(0), _shot(0), _shot(200)])

    obs.verify_capture(client, sleep=lambda _s: None)

    assert client.calls == 3


def test_frame_black_all_the_way_is_still_rejected():
    """Ретраи не должны превратиться в «всегда разрешаем»."""
    client = _Client([_shot(0)] * 6)

    with pytest.raises(AdventError) as error:
        obs.verify_capture(client, attempts=6, sleep=lambda _s: None)

    assert "пустой кадр" in error.value.message
    assert client.calls == 6


def test_bright_frame_passes_without_extra_requests():
    """Исправный источник не должен стоить лишних секунд ожидания."""
    client = _Client([_shot(200)])

    obs.verify_capture(client, sleep=lambda _s: None)

    assert client.calls == 1


def test_request_error_that_clears_up_is_not_fatal():
    """Источник в момент перепривязки может ответить отказом — это тоже «рано»."""
    client = _Client([RuntimeError("source not ready"), _shot(200)])

    obs.verify_capture(client, sleep=lambda _s: None)

    assert client.calls == 2


def test_request_error_that_never_clears_names_the_error():
    """Если так и не получилось — сообщение про запрос, а не про чёрный кадр.

    Иначе диагноз уводит в другую сторону: человек идёт проверять окно
    терминала, хотя сломан сам запрос к OBS.
    """
    client = _Client([RuntimeError("source not ready")] * 3)

    with pytest.raises(AdventError) as error:
        obs.verify_capture(client, attempts=3, sleep=lambda _s: None)

    assert "пробный кадр" in error.value.message
    assert "source not ready" in error.value.message


def test_verify_capture_waits_between_attempts():
    """Пауза между попытками реальная — без неё ретраи ничего не ждут."""
    slept: list[float] = []
    client = _Client([_shot(0), _shot(0), _shot(200)])

    obs.verify_capture(client, pause=0.7, sleep=slept.append)

    # Пауза только МЕЖДУ попытками: перед первой ждать нечего.
    assert slept == [0.7, 0.7]
