"""Фабрика клиента Mistral и список моделей.

ВАЖНО: SDK mistralai переехал на v2 — импорты идут из `mistralai.client`,
а не из `mistralai`. Любой сниппет с `from mistralai import Mistral` — от v1.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from mistralai.client import Mistral

from advent_core.config import Config
from advent_core.errors import translate

MODELS_URL = "https://api.mistral.ai/v1/models"

# Атрибут, под которым на клиенте живёт проба заголовков. Своё имя с
# префиксом, чтобы не столкнуться ни с чем в объекте SDK.
_PROBE_ATTR = "_advent_rate_limit_probe"


class _RateLimitProbe:
    """Запоминает заголовки последнего успешного ответа.

    Нужна ровно для лимитов частоты: Mistral отдаёт их только в заголовках
    (`x-ratelimit-limit-req-minute` и остальные), а разобранное тело ответа их
    не содержит — `chat.complete()` возвращает уже распакованный
    `ChatCompletionResponse`, из которого `httpx.Response` не достать.

    Это НЕ утка в приватный SDK: `SDKHooks.after_success()` перебирает
    зарегистрированные хуки и зовёт у каждого `after_success(ctx, response)`
    без всякой проверки типа, поэтому обычного объекта с этим методом
    достаточно — наследоваться от `mistralai.client._hooks.types.AfterSuccessHook`
    (приватный модуль) не требуется.

    Хук ОБЯЗАН вернуть response: `SDKHooks.after_success()` присваивает
    возвращённое значение обратно в цепочку, и `None` сломал бы разбор ответа
    у всех последующих хуков.
    """

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def after_success(self, hook_ctx: object, response: object) -> object:
        headers = getattr(response, "headers", None)
        if headers is not None:
            # Заголовки httpx нечувствительны к регистру, dict — уже нет,
            # поэтому ключи приводятся сразу, а не при каждом чтении.
            self.headers = {str(k).lower(): str(v) for k, v in headers.items()}
        return response


def _attach_rate_limit_probe(client: Mistral) -> None:
    """Вешает пробу на клиент; молча ничего не делает, если SDK изменился.

    Точка регистрации приватная: `Mistral.__init__` кладёт объект хуков в
    `self.sdk_configuration.__dict__["_hooks"]`, публичного способа добавить
    хук после создания клиента SDK не даёт (генератор Speakeasy предполагает
    правку `_hooks/registration.py`, то есть файла внутри пакета).

    Отсюда два следствия, оба сознательные:
      * всё завёрнуто в широкий except — телеметрия не имеет права уронить
        сам вызов, а этот шов может исчезнуть в любом минорном апдейте SDK;
      * когда шов исчезнет, проба останется пустой, и потребитель обязан
        считать лимит НЕизвестным, а не подставлять умолчание. Молчащий
        «ноль» здесь был бы хуже отсутствующего значения.
    """
    try:
        hooks = getattr(client.sdk_configuration, "_hooks", None)
        register = getattr(hooks, "register_after_success_hook", None)
        if register is None:
            return
        probe = _RateLimitProbe()
        register(probe)
        setattr(client, _PROBE_ATTR, probe)
    except Exception:  # noqa: BLE001 — см. docstring: телеметрия не роняет вызов
        return


def rate_limit_headers(client: object) -> dict[str, str]:
    """Заголовки лимита из последнего успешного ответа этого клиента.

    Пустой словарь означает «неизвестно» — либо шов регистрации отвалился,
    либо успешного ответа ещё не было.
    """
    probe = getattr(client, _PROBE_ATTR, None)
    return dict(getattr(probe, "headers", {}) or {})


def _header_int(headers: dict[str, str], name: str) -> int | None:
    """Заголовок как целое; None вместо исключения на любом мусоре."""
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def requests_per_minute(client: object) -> int | None:
    """Лимит запросов в минуту для модели последнего вызова, из заголовка.

    У Mistral он ПОМОДЕЛЬНЫЙ: 2026-09-04 на одном ключе `ministral-3b-latest`
    отдавал 750, `ministral-14b-latest` — 30, а `mistral-small-latest` — 0 и
    429 в те же секунды. Поэтому значение осмысленно только рядом с моделью,
    которой был сделан вызов, и кэшировать его на аккаунт нельзя.
    """
    return _header_int(rate_limit_headers(client), "x-ratelimit-limit-req-minute")


@contextmanager
def mistral_client(config: Config) -> Iterator[Mistral]:
    """Клиент с retry на 429/5xx.

    Backoff настраивается на уровне SDK: initial 1s, max 30s, множитель 1.5,
    общий бюджет 60s. Этого хватает, чтобы пережить всплеск лимита, и не
    настолько много, чтобы демо зависло перед камерой.
    """
    # Локальный сервер (LM Studio) держит один запрос за раз и отвечает
    # секундами, особенно на reasoning-моделях — дефолтный таймаут SDK этого
    # не переживёт. 600000 мс — тот же бюджет, что и у общего retry ниже, с
    # запасом под цепочку рассуждения на ~55 tok/s.
    extra_kwargs: dict = {}
    if config.base_url:
        extra_kwargs["server_url"] = config.base_url
        extra_kwargs["timeout_ms"] = 600_000

    try:
        from mistralai.client.utils import BackoffStrategy, RetryConfig

        retry_config = RetryConfig("backoff", BackoffStrategy(1000, 30000, 1.5, 60000), True)
        client = Mistral(api_key=config.api_key, retry_config=retry_config, **extra_kwargs)
    except ImportError:
        # Утилиты retry лежат в приватном модуле и могут переехать между
        # минорными версиями SDK. Без них клиент всё равно рабочий.
        client = Mistral(api_key=config.api_key, **extra_kwargs)

    _attach_rate_limit_probe(client)

    try:
        with client as mistral:
            yield mistral
    except Exception as exc:
        raise translate(exc) from exc


def list_models(config: Config) -> list[dict]:
    """Список моделей аккаунта через REST.

    Через httpx, а не через SDK: нужен сырой ответ с полями `id` и `aliases`,
    чтобы показать, во что разрешается `-latest`.

    При заданном config.base_url ходим на `{base_url}/v1/models`, а не на
    облачный MODELS_URL — LM Studio держит собственный список моделей.
    Authorization туда не нужен (сервер его не проверяет), но заголовок всё
    равно передаётся с тем же api_key: `config.api_key` в локальном режиме —
    заглушка (см. Config.resolve), лишний заголовок ничего не портит.
    """
    url = f"{config.base_url}/v1/models" if config.base_url else MODELS_URL
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
    except Exception as exc:
        raise translate(exc) from exc

    return response.json().get("data", [])


def resolve_alias(models: list[dict], name: str) -> str | None:
    """Во что разрешается алиас вида `mistral-small-latest`.

    Ищем модель, у которой запрошенное имя числится в aliases; её `id` и есть
    конкретная версия. Если API не отдал aliases — возвращаем None, а не врём.
    """
    for model in models:
        if name in (model.get("aliases") or []):
            return model.get("id")
    return None


def find_model(models: list[dict], name: str) -> dict | None:
    """Карточка модели по id или алиасу."""
    for model in models:
        if model.get("id") == name or name in (model.get("aliases") or []):
            return model
    return None


def capabilities_of(models: list[dict], name: str) -> dict | None:
    """Capabilities модели: чем определяется, какие параметры ей слать.

    LM Studio отдаёт `/v1/models` без поля `capabilities` вовсе — не пустой
    словарь, а отсутствующий ключ. find_model() тогда возвращает карточку без
    "capabilities", и .get() честно вернёт None: «неизвестно», не «ничего не
    умеет». Params.as_payload() именно так и трактует None — параметры уходят
    БЕЗ фильтрации. НЕ подставляй здесь дефолт вида {} или выдуманный набор
    capabilities для локальных моделей: сервер, у которого этого списка нет
    вовсе, не даёт оснований ни разрешать, ни запрещать что-то конкретное —
    гадать про его возможности хуже, чем отправить и узнать по ответу.
    """
    model = find_model(models, name)
    return (model or {}).get("capabilities") if model else None


def chat_models(models: list[dict]) -> list[dict]:
    """Только те, что умеют chat completion — 29 из 48 на этом аккаунте."""
    return [m for m in models if (m.get("capabilities") or {}).get("completion_chat")]


def model_names(models: list[dict]) -> set[str]:
    """Все имена, которые API примет как model: и id, и алиасы."""
    names: set[str] = set()
    for model in models:
        if model_id := model.get("id"):
            names.add(model_id)
        names.update(model.get("aliases") or [])
    return names
