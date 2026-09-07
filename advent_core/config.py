"""Конфигурация: загрузка .env, приоритет CLI > env > default, redaction секретов."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


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
    for key in SECRET_KEYS:
        value = _env(key)
        if value and len(value) >= 8:
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
    ) -> Config:
        """Собирает конфиг по приоритету: аргумент CLI > переменная .env > дефолт."""
        load_env()

        base_url = normalize_base_url(base_url or _env("ADVENT_BASE_URL"))

        api_key = _env("MISTRAL_API_KEY")
        if not api_key:
            if base_url:
                # Локальный OpenAI-совместимый сервер (LM Studio) не проверяет
                # Authorization вовсе, но SDK требует непустую строку в
                # api_key — заглушка, а не настоящий секрет. Облачный режим
                # (base_url не задан) по-прежнему требует настоящий ключ ниже.
                api_key = "lm-studio-local"
            else:
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
        )

    def system_prompt(self) -> str | None:
        if self.system_prompt_path is None:
            return None
        text = self.system_prompt_path.read_text(encoding="utf-8").strip()
        return text or None
