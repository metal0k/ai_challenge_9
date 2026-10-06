"""Конфигурация: загрузка .env, приоритет CLI > env > default, redaction секретов."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from advent_core.params import GenerationParams, ParamError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ministral-14b-latest, не mistral-small-latest: живой замер 2026-09-04
# (SPEC-w01d05.md §2) — mistral-small отдаёт 429 с
# x-ratelimit-limit-req-minute=0 на этом аккаунте, то есть прежний дефолт
# ломает `advent w01 chat` у любого, кто склонирует репозиторий. 14b — самая
# сильная модель из трёх реально доступных (§3).
DEFAULT_MODEL = "ministral-14b-latest"
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "advent_core" / "prompts" / "default_system.md"
LOG_DIR = PROJECT_ROOT / "logs"

# Ключи .env, значения которых никогда не должны попасть в stdout, лог или кадр видео.
SECRET_KEYS = (
    "MISTRAL_API_KEY",
    "GITHUB_TOKEN",
    "OBS_WS_PASSWORD",
    "YANDEX_DISK_TOKEN",
)


class ConfigError(Exception):
    """Ошибка конфигурации, показывается пользователю без traceback."""

    exit_code = 2


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOCAL_API_KEY = "lm-studio-local"


def url_host(url: str) -> str | None:
    """Lower-cased host of `url` without brackets/port; None when unparsable."""
    try:
        host = urlsplit(url if "//" in url else f"//{url}").hostname
    except ValueError:
        return None
    return host.lower() if host else None


def _strict_parts(url: str | None):
    """urlsplit result for a well-formed http(s) URL (scheme optional), else None.

    Rejects a bad port, userinfo and a foreign scheme: `http://127.0.0.1:bad` is
    not a URL, and `http://127.0.0.1@evil.com` is not loopback.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url if "//" in url else f"//{url}")
        parts.port  # noqa: B018 - raises ValueError on a malformed/out-of-range port
    except ValueError:
        return None
    if parts.scheme not in ("", "http", "https") or "@" in parts.netloc:
        return None
    return parts


def is_loopback_url(url: str | None) -> bool:
    """True only for a well-formed http(s) URL on 127.0.0.1 / localhost / ::1."""
    parts = _strict_parts(url)
    return parts is not None and (parts.hostname or "").lower() in LOOPBACK_HOSTS


def validate_loopback_url(url: str, what: str = "локальный сервер") -> str:
    """Return `url` or raise ConfigError (exit 2): offline talks to loopback only."""
    if _strict_parts(url) is None:
        raise ConfigError(f"{what}: {url!r} — некорректный URL (нужен http(s)://host:порт).")
    if not is_loopback_url(url):
        raise ConfigError(
            f"{what}: {url!r} не loopback — разрешены только 127.0.0.1, localhost, ::1."
        )
    return url


def offline_env() -> bool:
    return (os.getenv("ADVENT_OFFLINE") or "").strip().lower() in ("1", "true", "yes", "on")


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def configured_secrets() -> tuple[str, ...]:
    """Return configured secret values that are long enough to protect safely."""
    return tuple(
        dict.fromkeys(
            value for key in SECRET_KEYS if (value := _env(key)) is not None and len(value) >= 8
        )
    )


def normalize_base_url(value: str | None) -> str | None:
    """Приводит base_url к корню сервера — без хвостового `/` и без `/v1`.

    SDK сам дописывает `/v1/chat/completions`, а list_models() — `/v1/models`.
    Но документация LM Studio называет «base URL» именно
    `http://127.0.0.1:1234/v1`, и скопированное оттуда значение молча
    превратилось бы в `/v1/v1/chat/completions`: сервер ответит 404, а
    сообщение будет про недоступную модель, а не про URL. Дешевле срезать
    хвост здесь, в единственной точке разбора, чем объяснять это в help.
    """
    if not value:
        return None
    url = value.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or None


def _first(*candidates: object) -> object | None:
    """Первое заданное значение — так работает приоритет CLI > env > дефолт."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def redact(text: str) -> str:
    """Заменяет значения секретов на ***.

    Вызывается перед печатью любого текста ошибки и перед записью в лог:
    traceback SDK или тело HTTP-ответа могут содержать ключ.
    """
    for value in configured_secrets():
        text = text.replace(value, "***")
    return text


@dataclass(slots=True)
class Config:
    """Разрешённая конфигурация одного запуска."""

    api_key: str
    model: str = DEFAULT_MODEL
    system_prompt_path: Path | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    stream: bool = True
    verbose: bool = False
    log_path: Path = field(default_factory=lambda: LOG_DIR / "calls.jsonl")
    # Базовый URL OpenAI-совместимого endpoint (LM Studio и т.п.). None —
    # облачный Mistral API, поведение не меняется ни в чём.
    base_url: str | None = None
    # Process-wide offline mode (advent_core.offline): no cloud, no real key.
    offline: bool = False

    @classmethod
    def resolve(
        cls,
        *,
        model: str | None = None,
        system: Path | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        random_seed: int | None = None,
        stop: str | None = None,
        reasoning_effort: str | None = None,
        stream: bool = True,
        verbose: bool = False,
        base_url: str | None = None,
        offline: bool | None = None,
    ) -> Config:
        """Собирает конфиг по приоритету: аргумент CLI > переменная .env > дефолт."""
        load_env()

        base_url = normalize_base_url(base_url or _env("ADVENT_BASE_URL"))

        offline = bool(offline) or offline_env()
        if offline:
            from advent_core import offline as offline_guard

            if base_url:
                validate_loopback_url(base_url, "offline-режим")
            offline_guard.enable()

        if offline or is_loopback_url(base_url):
            # A local OpenAI-compatible server ignores Authorization, but the SDK
            # wants a non-empty string: a stub. The real key never goes to a loopback
            # base_url, and offline does not even read it. A non-loopback base_url
            # (e.g. https://api.mistral.ai) keeps the real key and the SDK path.
            api_key = LOCAL_API_KEY
        else:
            api_key = _env("MISTRAL_API_KEY")
            if not api_key:
                raise ConfigError(
                    "Не найден MISTRAL_API_KEY.\n"
                    "Скопируй .env.example в .env и вставь ключ из https://console.mistral.ai/api-keys"
                )

        system_path = system or (Path(p) if (p := _env("ADVENT_SYSTEM_PROMPT")) else None)
        if system_path is None and DEFAULT_SYSTEM_PROMPT.exists():
            system_path = DEFAULT_SYSTEM_PROMPT
        if system_path is not None and not system_path.exists():
            raise ConfigError(f"Файл system prompt не найден: {system_path}")

        # Приоритет на каждый параметр по отдельности: флаг CLI > .env > не слать.
        try:
            params = GenerationParams.build(
                temperature=_first(temperature, _env("MISTRAL_TEMPERATURE")),
                top_p=_first(top_p, _env("MISTRAL_TOP_P")),
                max_tokens=_first(max_tokens, _env("MISTRAL_MAX_TOKENS")),
                random_seed=_first(random_seed, _env("MISTRAL_SEED")),
                stop=_first(stop, _env("MISTRAL_STOP")),
                reasoning_effort=_first(reasoning_effort, _env("MISTRAL_REASONING_EFFORT")),
            )
        except ParamError as exc:
            raise ConfigError(str(exc)) from exc

        return cls(
            api_key=api_key,
            model=model or _env("MISTRAL_MODEL") or DEFAULT_MODEL,
            system_prompt_path=system_path,
            params=params,
            stream=stream,
            verbose=verbose,
            base_url=base_url,
            offline=offline,
        )

    @property
    def is_local(self) -> bool:
        """A loopback base_url: stub key, openai_compat routing, LM Studio payload rewrite."""
        return is_loopback_url(self.base_url)

    def system_prompt(self) -> str | None:
        if self.system_prompt_path is None:
            return None
        text = self.system_prompt_path.read_text(encoding="utf-8").strip()
        return text or None
