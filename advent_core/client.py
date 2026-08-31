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


@contextmanager
def mistral_client(config: Config) -> Iterator[Mistral]:
    """Клиент с retry на 429/5xx.

    Backoff настраивается на уровне SDK: initial 1s, max 30s, множитель 1.5,
    общий бюджет 60s. Этого хватает, чтобы пережить всплеск лимита, и не
    настолько много, чтобы демо зависло перед камерой.
    """
    try:
        from mistralai.client.utils import BackoffStrategy, RetryConfig

        retry_config = RetryConfig("backoff", BackoffStrategy(1000, 30000, 1.5, 60000), True)
        client = Mistral(api_key=config.api_key, retry_config=retry_config)
    except ImportError:
        # Утилиты retry лежат в приватном модуле и могут переехать между
        # минорными версиями SDK. Без них клиент всё равно рабочий.
        client = Mistral(api_key=config.api_key)

    try:
        with client as mistral:
            yield mistral
    except Exception as exc:
        raise translate(exc) from exc


def list_models(config: Config) -> list[dict]:
    """Список моделей аккаунта через REST.

    Через httpx, а не через SDK: нужен сырой ответ с полями `id` и `aliases`,
    чтобы показать, во что разрешается `-latest`.
    """
    try:
        response = httpx.get(
            MODELS_URL,
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
    """Capabilities модели: чем определяется, какие параметры ей слать."""
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
