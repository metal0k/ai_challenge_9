"""Пресеты формата ответа: system-инструкция, разбор done, вердикт, схема.

Сети не касается — весь модуль advent_core/formats.py работает на строках
и словарях, без обращения к API.
"""

from __future__ import annotations

import json

import pytest

from advent_core.config import ConfigError
from advent_core.formats import build_system, is_done, load_schema, parse_done, verify

SCHEMA = {
    "title": "ingredients",
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "amount": {"type": "string"}},
                "required": ["name", "amount"],
            },
        }
    },
    "required": ["items"],
}


# --- build_system: инструкция пресета дописывается, а не затирает пользовательский system ---


def test_build_system_appends_instruction_to_user_system():
    system = build_system("json", "Ты дружелюбный ассистент.")
    assert system.startswith("Ты дружелюбный ассистент.")
    assert "JSON" in system


def test_build_system_without_user_system_uses_instruction_alone():
    system = build_system("json", None)
    assert "JSON" in system
    assert not system.startswith("\n")


def test_build_system_text_format_leaves_user_system_untouched():
    """format=text не имеет пресета — system должен вернуться как есть."""
    assert build_system("text", "мой system") == "мой system"


def test_build_system_text_format_with_no_system_is_none():
    assert build_system("text", None) is None


def test_build_system_schema_embeds_schema_definition():
    system = build_system("schema", "мой system", SCHEMA)
    assert "мой system" in system
    assert '"items"' in system  # дамп схемы попал в текст следом за инструкцией


# --- parse_done: разбор префикса text:/json:, мусор без префикса — ConfigError ---


def test_parse_done_text_prefix():
    assert parse_done("text:[ГОТОВО]") == ("text", "[ГОТОВО]")


def test_parse_done_json_prefix():
    assert parse_done("json:done") == ("json", "done")


def test_parse_done_strips_whitespace_around_needle():
    assert parse_done("text: готово ") == ("text", "готово")


@pytest.mark.parametrize("value", ["мусор", "готово", "", ":", "text:", "xml:tag"])
def test_parse_done_rejects_values_without_recognised_prefix(value):
    with pytest.raises(ConfigError):
        parse_done(value)


# --- is_done: срабатывает по полю, по подстроке, не срабатывает ложно, битый JSON не роняет ---


def test_is_done_text_matches_substring():
    assert is_done("вот твой рецепт. [ГОТОВО]", "text", "[ГОТОВО]") is True


def test_is_done_text_no_false_positive_on_unrelated_text():
    assert is_done("ещё уточняю детали салата", "text", "[ГОТОВО]") is False


def test_is_done_json_matches_true_field():
    assert is_done('{"done": true, "result": {}}', "json", "done") is True


def test_is_done_json_false_field_is_not_done():
    assert is_done('{"done": false, "question": "какой салат?"}', "json", "done") is False


def test_is_done_broken_json_does_not_raise():
    """Промежуточный ход диалога может прислать битый JSON — это не готово, а не ошибка."""
    assert is_done("{совсем не json", "json", "done") is False


def test_is_done_json_array_instead_of_object_is_not_done():
    assert is_done("[1, 2, 3]", "json", "done") is False


# --- verify: вердикт по формату для footer ---


def test_verify_valid_json_is_ok():
    verdict = verify("json", '{"items": []}')
    assert verdict.ok is True
    assert verdict.detail == "JSON ✓"


def test_verify_broken_json_is_not_ok():
    verdict = verify("json", "{незакрытая скобка")
    assert verdict.ok is False
    assert "✗" in verdict.detail


def test_verify_schema_valid_reports_item_count():
    payload = json.dumps({"items": [{"name": "огурец", "amount": "2 шт"}]})
    verdict = verify("schema", payload, SCHEMA)
    assert verdict.ok is True
    assert "схема ✓" in verdict.detail
    assert "items: 1" in verdict.detail


def test_verify_schema_mismatch_is_not_ok():
    """Валидный JSON, но без обязательного поля amount — схема не проходит."""
    payload = json.dumps({"items": [{"name": "огурец"}]})
    verdict = verify("schema", payload, SCHEMA)
    assert verdict.ok is False
    assert "схема ✗" in verdict.detail


def test_verify_truncated_by_length_reads_as_broken_json():
    """Обрыв на max_tokens режет JSON посередине — это и есть finish=length ломает format."""
    truncated = '{"items": [{"name": "огурец", "amount": "2'
    verdict = verify("json", truncated)
    assert verdict.ok is False


def test_verify_text_yaml_md_have_no_verdict():
    """Нет ни API-гарантии, ни дешёвой проверки — прочерк, а не False."""
    for fmt in ("text", "yaml", "md"):
        verdict = verify(fmt, "что угодно, хоть невалидный YAML")
        assert verdict.ok is None
        assert verdict.detail == "—"


# --- load_schema: файла нет / не JSON / не объект ---


def test_load_schema_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_schema(tmp_path / "нет-такого.json")


def test_load_schema_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_schema(path)


def test_load_schema_rejects_non_object_json(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_schema(path)


def test_load_schema_reads_valid_object(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(SCHEMA, ensure_ascii=False), encoding="utf-8")
    assert load_schema(path) == SCHEMA
