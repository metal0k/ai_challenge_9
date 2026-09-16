"""Named, non-secret preference profiles for the agent."""

from __future__ import annotations

import json
import re
from pathlib import Path

from advent_core.config import LOG_DIR, ConfigError

PROFILES_DIR = LOG_DIR / "profiles"
_NAME_RE = re.compile(r"^[\w-]{1,64}$", re.UNICODE)
_CREDENTIAL_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|bearer|authorization|sk-[a-z0-9])"
)


def validate_name(name: str) -> str:
    value = (name or "").strip()
    if not _NAME_RE.fullmatch(value):
        raise ConfigError(
            "Недопустимое имя profile: разрешены буквы, цифры, дефис и "
            "подчёркивание, до 64 символов"
        )
    return value


def validate_value(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ConfigError("Значение preference не может быть пустым")
    if _CREDENTIAL_RE.search(value):
        raise ConfigError("Profile не может содержать credential-like значение")
    return value


def messages(values: dict[str, str]) -> list[dict[str, str]]:
    """Render profile preferences as a counted pseudo exchange."""
    if not values:
        return []
    # Keep profile context bounded: it is protected from ordinary history trim.
    body = "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))[:4000]
    return [
        {"role": "user", "content": "Профиль пользователя (учитывай):\n" + body},
        {
            "role": "assistant",
            "content": (
                "Профиль учтён как preference по умолчанию; текущий запрос "
                "пользователя при конфликте приоритетнее."
            ),
        },
    ]


def path_for(name: str, directory: Path | None = None) -> Path:
    return (directory or PROFILES_DIR) / f"{validate_name(name)}.json"


def load(name: str, directory: Path | None = None) -> dict[str, str]:
    path = path_for(name, directory)
    if not path.is_file():
        raise ConfigError(f"Profile {name!r} не найден")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Не удалось прочитать profile {name!r}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Profile {name!r} имеет неверный формат")
    values = {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}
    safe: dict[str, str] = {}
    for key, value in values.items():
        if _CREDENTIAL_RE.search(key):
            continue
        try:
            validate_value(value)
        except ConfigError:
            continue
        safe[key.strip()] = value.strip()
    return safe


def save(name: str, values: dict[str, str], directory: Path | None = None) -> Path:
    path = path_for(name, directory)
    checked: dict[str, str] = {}
    for key, value in values.items():
        key = str(key).strip()
        if not key:
            raise ConfigError("Ключ preference не может быть пустым")
        if _CREDENTIAL_RE.search(key):
            raise ConfigError("Profile не может содержать credential-like key")
        checked[key] = validate_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def list_names(directory: Path | None = None) -> list[str]:
    root = directory or PROFILES_DIR
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json") if _NAME_RE.fullmatch(p.stem))


def delete(name: str, directory: Path | None = None) -> None:
    path_for(name, directory).unlink(missing_ok=True)
