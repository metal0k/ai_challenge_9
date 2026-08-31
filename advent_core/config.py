"""Конфигурация: загрузка .env, приоритет CLI > env > default, redaction секретов."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from advent_core.params import GenerationParams, ParamError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mistral-small-latest"
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
    ) -> Config:
        """Собирает конфиг по приоритету: аргумент CLI > переменная .env > дефолт."""
        load_env()

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
        )

    def system_prompt(self) -> str | None:
        if self.system_prompt_path is None:
            return None
        text = self.system_prompt_path.read_text(encoding="utf-8").strip()
        return text or None
