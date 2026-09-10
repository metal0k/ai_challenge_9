"""Маппинг исключений SDK и HTTP-статусов в человеческие сообщения."""

from __future__ import annotations

import re

from advent_core.config import redact

# Дискриминаторы серверного переполнения окна — живой замер 2026-09-09
# (specs/PROBE-w02d08-overflow.md): тело 400 несёт type
# "invalid_request_prompt_too_long" и message вида
# "Prompt 267060 > 262144 maximum context length". Проверяем оба, потому что
# формат тела не задокументирован и любой из двух маркеров может исчезнуть.
_OVERFLOW_TYPE = "invalid_request_prompt_too_long"
_OVERFLOW_MARKER = "maximum context length"
_OVERFLOW_NUMBERS = re.compile(r"(\d+)\s*>\s*(\d+)")


class AdventError(Exception):
    """Ошибка, которую показываем пользователю текстом, а не traceback."""

    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class AuthError(AdventError):
    exit_code = 3


class RateLimitError(AdventError):
    exit_code = 4


class ServerError(AdventError):
    exit_code = 5


class NetworkError(AdventError):
    exit_code = 6


class StreamTruncated(AdventError):
    exit_code = 7


class ConfigurationError(AdventError):
    """Ошибка конфигурации, долетевшая до слоя chat.py (обычно из formats.py).

    exit_code = 2 — тот же смысл, что у advent_core.config.ConfigError
    («настройка неверна», не сбой API), но AdventError-совместимая обёртка:
    week_01/cli.py в REPL ловит именно `except AdventError` вокруг вызовов
    chat.complete()/stream(), поэтому ошибка конфигурации предупреждает и не
    убивает сессию, а advent_core.config.ConfigError таким except'ом не
    поймать — она не наследует AdventError. advent_cli/app.py на верхнем
    уровне читает error.exit_code, а не тип исключения, поэтому именно этот
    exit_code=2 и доезжает до одношотового вызова (`advent w01 chat ...
    --format schema` без --schema-file → exit 2, а не общий 1 у голого
    AdventError).
    """

    exit_code = 2


def _overflow_of(detail: str) -> tuple[int | None, int | None] | None:
    """Вытаскивает из текста ошибки факт переполнения окна и его числа.

    Возвращает (прислано, лимит) — любое из чисел может быть None, если
    маркеры нашлись, а разобрать их не вышло. None на всю кортеж-обёртку
    значит «это не переполнение», и 400 разбирается по обычной ветке.
    """
    if _OVERFLOW_TYPE not in detail and _OVERFLOW_MARKER not in detail:
        return None
    match = _OVERFLOW_NUMBERS.search(detail)
    if match is None:
        return (None, None)
    return (int(match.group(1)), int(match.group(2)))


def _status_of(exc: Exception) -> int | None:
    """Достаёт HTTP-статус из исключения SDK, не завязываясь на его класс.

    В mistralai v2 ошибки приходят как SDKError со `status_code`; у httpx —
    как HTTPStatusError с `response.status_code`. Утиная типизация здесь
    надёжнее, чем импорт конкретного класса, который SDK может переименовать.
    """
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def translate(exc: Exception) -> AdventError:
    """Превращает исключение в AdventError с понятным текстом."""
    if isinstance(exc, AdventError):
        return exc

    status = _status_of(exc)
    detail = redact(str(exc))[:500]

    if status in (401, 403):
        return AuthError(
            f"Ключ Mistral отклонён ({status}).",
            hint=(
                "Проверь MISTRAL_API_KEY в .env — возможно, ключ отозван "
                "или скопирован не полностью."
            ),
        )
    if status == 404:
        return AdventError(
            "Mistral отвечает 404 — скорее всего, неизвестное имя модели.",
            hint="Посмотри доступные модели: advent w01 models",
        )
    if status == 400:
        overflow = _overflow_of(detail)
        if overflow is not None:
            sent, limit = overflow
            if sent is not None and limit is not None:
                numbers = f": прислано {sent}, лимит {limit}"
            else:
                # Числа не разобрались — показываем сырой текст сервера,
                # чтобы диагностика не превратилась в «что-то с окном».
                numbers = f": {detail}"
            return AdventError(
                f"Запрос не влезает в окно модели{numbers}.",
                hint=(
                    "История пересылается целиком, поэтому повтор той же сессии "
                    "снова не влезет: начни новую командой /new или уменьши "
                    "ввод — например, сократи вставленный текст."
                ),
            )
        # Самая частая причина здесь — отказ модели от response_format
        # (format=json/schema): такой capability-флаг не отдаётся списком
        # моделей ни у одной из них (SPEC-w01d02.md §2, §6.4), поэтому
        # response_format уходит всегда и разбирается по факту отказа, а не
        # гейтится заранее по недокументированной догадке.
        return AdventError(
            f"Mistral отклонил запрос (400): {detail}",
            hint=(
                "Часто это отказ модели от response_format (format=json/schema) "
                "— попробуй /set format text или другую модель: advent w01 models"
            ),
        )
    if status == 422:
        return AdventError(f"Mistral отклонил параметры запроса (422): {detail}")
    if status == 429:
        return RateLimitError(
            "Лимит запросов Mistral исчерпан (429).",
            hint=(
                "Повторные попытки уже сделаны. Подожди минуту или возьми модель "
                "полегче: --model ministral-8b-latest. Если 429 повторяется на "
                "КАЖДОЙ попытке — модель, возможно, недоступна на текущем тарифе "
                "(у Mistral это выглядит как x-ratelimit-limit-req-minute: 0 в "
                "заголовке ответа, а не как 403); попробуй другую модель через "
                "--model."
            ),
        )
    if status is not None and 500 <= status < 600:
        return ServerError(
            f"Mistral вернул ошибку сервера ({status}). Повторные попытки не помогли."
        )

    name = type(exc).__name__.lower()
    if any(marker in name for marker in ("timeout", "connect", "network", "ssl")):
        return NetworkError(
            "Нет связи с api.mistral.ai.",
            hint="Проверь интернет, VPN и прокси.",
        )

    return AdventError(f"Неожиданная ошибка при обращении к Mistral: {detail}")
