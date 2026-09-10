"""Маппинг ошибок API в человеческие сообщения (errors.translate).

Исключения строятся как утки со `status_code` и текстом — как реальные
SDKError mistralai v2, но без сети и без завязки на классы SDK
(Speakeasy их переименовывает, поэтому translate() и работает утиной
типизацией).
"""

from __future__ import annotations

import json

import httpx
import pytest
from mistralai.client.errors import SDKError

from advent_core.errors import AdventError, RateLimitError, translate

OVERFLOW_BODY = json.dumps(
    {
        "object": "error",
        "message": "Prompt 267060 > 262144 maximum context length",
        "type": "invalid_request_prompt_too_long",
        "param": None,
        "code": "3059",
        "raw_status_code": 400,
    }
)


def _sdk_error(status: int, body: str) -> SDKError:
    """SDKError, как его собирает Speakeasy на реальном ответе сервера."""
    response = httpx.Response(status, headers={"content-type": "application/json"})
    return SDKError("mistral error", response, body=body)


class _DuckError(Exception):
    """Минимальная утка: статус читается из атрибута, текст — из str()."""

    def __init__(self, status: int, text: str):
        super().__init__(text)
        self.status_code = status


def test_overflow_400_by_type_gets_window_hint():
    error = translate(_sdk_error(400, OVERFLOW_BODY))

    assert isinstance(error, AdventError)
    assert "окно модели" in error.message
    assert "267060" in error.message and "262144" in error.message
    assert error.hint is not None
    assert "/new" in error.hint


def test_overflow_400_by_marker_without_type():
    """type может исчезнуть из тела — подстрока message достаточна."""
    body = json.dumps({"message": "Prompt 999999 > 131072 maximum context length"})
    error = translate(_sdk_error(400, body))

    assert "окно модели" in error.message
    assert "999999" in error.message and "131072" in error.message


def test_overflow_400_by_marker_without_numbers():
    """Маркеры есть, числа не разобрались — сырой текст сервера сохраняется."""
    error = translate(_DuckError(400, "maximum context length exceeded"))

    assert "окно модели" in error.message
    assert "maximum context length exceeded" in error.message
    assert error.hint is not None


def test_other_400_keeps_old_message():
    """Прочие 400 — прежнее поведение: hint про response_format."""
    error = translate(_sdk_error(400, json.dumps({"message": "bad format request"})))

    assert isinstance(error, AdventError)
    assert "response_format" in (error.hint or "")
    assert "окно модели" not in error.message


def test_429_is_still_rate_limit_not_overflow():
    body = json.dumps({"message": "Prompt 10 > 5 maximum context length"})
    error = translate(_sdk_error(429, body))

    # 429 сворачивается по статусу раньше любого разбора тела — маркер
    # переполнения не должен перехватить чужой статус.
    assert isinstance(error, RateLimitError)


@pytest.mark.parametrize("status", [401, 403, 404, 422, 500])
def test_non_400_statuses_are_untouched(status):
    error = translate(_DuckError(status, "Prompt 1 > 0 maximum context length"))

    assert "окно модели" not in error.message
