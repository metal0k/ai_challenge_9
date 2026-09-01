"""Форматы ответа: пресеты, инструкция в system prompt, response_format, вердикт.

Отдельный модуль в advent_core, а не в week_01: недели 4 (MCP) и 5 (RAG) будут
просить структурированный вывод у тех же моделей Mistral, и весь этот механизм
(пресеты, сборка system, вердикт по ответу) им нужен без переезда.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from advent_core.config import ConfigError
from advent_core.params import FORMAT_CHOICES

# КРИТИЧНО и подтверждено докáми Mistral (Chat Completions / Structured
# Outputs): response_format гарантирует, что ответ — валидный JSON, но не то,
# *какой именно* JSON. "When using JSON mode you MUST also instruct the model
# to produce JSON yourself with a system or a user message." Инструкция в
# промпте — не дублирование параметра, а его обязательное дополнение; без неё
# модель честно вернёт валидный, но произвольный JSON. Не выкидывать этот
# текст как «параметр и так всё гарантирует».
#
# Пресет `json` — общего назначения и НЕ должен диктовать доменную форму
# (когда-то тут было "…{"items":[{"name":…,"amount":…}]}" — ошибка исходного
# ТЗ про салат). Конкретные поля задаёт пользователь своим вопросом/system
# prompt'ом, либо пресет `schema` через файл схемы. Иначе в mode=dialog
# инструкция про {"items":…} дописывается поверх dialog-промпта, который
# требует {"done":…, "question"/"result":…} — два противоречащих друг другу
# требования в одном system prompt.
_JSON_INSTRUCTION = (
    "Верни ответ строго в виде одного валидного JSON-объекта — без пояснений и без обёртки в ```."
)

_SCHEMA_INSTRUCTION = (
    "Верни ответ строго в формате JSON, точно соответствующем приведённой ниже "
    "JSON Schema — без пояснений и без обёртки в ```."
)

# Для YAML и Markdown у API нет никакого механизма (ни response_format, ни его
# аналога) — единственная опора это промпт. Это и есть главный контраст дня
# между json/schema (промпт + API-гарантия) и yaml/md (только промпт).
_YAML_INSTRUCTION = "Верни ответ в формате YAML — без пояснений и без обёртки в ```."
_MD_INSTRUCTION = "Верни ответ в виде таблицы Markdown — без пояснений вне таблицы."


@dataclass(slots=True)
class FormatPreset:
    """Один пресет формата: инструкция для system prompt + response_format."""

    name: str
    instruction: str | None

    def response_format(self, schema: dict[str, Any] | None) -> dict[str, Any] | None:
        """Значение параметра `response_format`, либо None, если API-опоры нет."""
        if self.name == "json":
            return {"type": "json_object"}

        if self.name == "schema":
            if schema is None:
                raise ConfigError(
                    "format=schema требует schema_file — задай его: "
                    "/set schema_file <путь к .json> или --schema-file <путь>"
                )
            # Поле JSONSchema в mistralai 2.9.4 называется schema_definition,
            # не schema — так называется в самом SDK (mistralai.client),
            # список полей: name, schema_definition, description, strict.
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title") or "response",
                    "schema_definition": schema,
                    "strict": True,
                },
            }

        return None  # text / yaml / md


PRESETS: dict[str, FormatPreset] = {
    "text": FormatPreset("text", None),
    "json": FormatPreset("json", _JSON_INSTRUCTION),
    "schema": FormatPreset("schema", _SCHEMA_INSTRUCTION),
    "yaml": FormatPreset("yaml", _YAML_INSTRUCTION),
    "md": FormatPreset("md", _MD_INSTRUCTION),
}
# Реестр пресетов и choices параметра `format` в params.py — одна и та же
# истина в двух местах; при расхождении лучше упасть на импорте, чем молча
# отрисовать пустую строку вместо инструкции.
assert set(PRESETS) == set(FORMAT_CHOICES)


def _preset(format_name: str) -> FormatPreset:
    preset = PRESETS.get(format_name)
    if preset is None:
        allowed = ", ".join(PRESETS)
        raise ValueError(f"неизвестный формат {format_name!r}, ожидалось одно из: {allowed}")
    return preset


def build_system(
    format_name: str, base_system: str | None, schema: dict[str, Any] | None = None
) -> str | None:
    """Дописывает инструкцию пресета к пользовательскому system prompt.

    Пользовательский system НЕ затирается: инструкция формата идёт отдельным
    абзацем следом (`/params` показывает результат целиком, ничего не
    подменяется молча — это прямое требование спеки дня).
    """
    preset = _preset(format_name)
    instruction = preset.instruction
    if preset.name == "schema" and instruction is not None and schema is not None:
        instruction = f"{instruction}\n\n{json.dumps(schema, ensure_ascii=False, indent=2)}"

    parts = [part for part in (base_system, instruction) if part]
    return "\n\n".join(parts) if parts else None


def response_format_for(
    format_name: str, schema: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Значение параметра запроса `response_format` для пресета (или None)."""
    return _preset(format_name).response_format(schema)


def load_schema(path: str | Path) -> dict[str, Any]:
    """Читает JSON Schema из файла.

    ConfigError на всех трёх типичных ошибках: файла нет, содержимое не JSON,
    JSON не объект — во всех трёх случаях "используй эту схему" бессмысленно,
    и лучше сказать это словами, чем упасть traceback'ом на первом запросе.
    """
    schema_path = Path(path)
    if not schema_path.is_file():
        raise ConfigError(f"Файл схемы не найден: {schema_path}")

    raw = schema_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Файл схемы {schema_path} — не валидный JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Схема в {schema_path} должна быть JSON-объектом, а не {type(data).__name__}"
        )
    return data


def parse_done(value: str) -> tuple[str, str]:
    """Разбирает значение параметра `done` в (kind, needle).

    kind — "text" или "json", needle — искомая подстрока (text) или имя
    булева поля (json). Значение без распознанного префикса — ConfigError:
    Spec.done в params.py валидирует только то, что это непустая строка,
    семантику префикса знает только эта функция (см. SPEC-w01d02.md §6.1).
    """
    kind, sep, needle = value.partition(":")
    kind = kind.strip().lower()
    needle = needle.strip()
    if not sep or kind not in ("text", "json") or not needle:
        raise ConfigError(
            f'done должен быть вида "text:<строка>" или "json:<поле>", получено {value!r}'
        )
    return kind, needle


def is_done(text: str, kind: str, needle: str) -> bool:
    """Проверяет условие завершения диалога по уже полученному ответу модели."""
    if kind == "text":
        return needle in text

    if kind == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Битый JSON на промежуточном ходу — сигнал «ещё не готово», а не
            # повод уронить диалог: цикл должен продолжиться следующим ходом.
            return False
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get(needle))

    raise ValueError(f'неизвестный вид условия завершения {kind!r}, ожидалось "text" или "json"')


@dataclass(slots=True)
class FormatVerdict:
    """Вердикт по ответу для footer.

    ok=None значит «вердикт не выносится» (осознанно, не пропуск проверки) —
    у text/yaml/md нет ни API-гарантии, ни дешёвой верификации; «✓» здесь
    означало бы больше уверенности, чем есть на самом деле.
    """

    ok: bool | None
    detail: str


def verify(format_name: str, text: str, schema: dict[str, Any] | None = None) -> FormatVerdict:
    """Проверяет ответ на соответствие пресету format_name."""
    if format_name in ("text", "yaml", "md"):
        return FormatVerdict(ok=None, detail="—")

    if format_name not in ("json", "schema"):
        raise ValueError(f"неизвестный формат {format_name!r}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return FormatVerdict(ok=False, detail="JSON ✗")

    if format_name == "json":
        return FormatVerdict(ok=True, detail="JSON ✓")

    # format_name == "schema": валидация — не поверхностная сверка ключей,
    # а настоящий jsonschema.validate (см. обоснование зависимости в pyproject).
    if schema is None:
        return FormatVerdict(ok=False, detail="JSON ✓ · схема не задана")
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError:
        return FormatVerdict(ok=False, detail="JSON ✓ · схема ✗")

    detail = "JSON ✓ · схема ✓"
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if isinstance(items, list):
        detail += f" · items: {len(items)}"
    return FormatVerdict(ok=True, detail=detail)
