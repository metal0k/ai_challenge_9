"""Pure facts functions: delta application, stable rendering, note diffing.

No network — pure functions over a facts dict. Tests assert literals and
counts, never a value derived from the code under test (CLAUDE.md rule).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jsonschema
import pytest

from advent_core.facts import (
    FACT_CATEGORIES,
    FACTS_ACK,
    apply_delta,
    facts_messages,
    format_facts,
    render_note,
    validate_key,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "advent_core" / "schemas" / "facts_delta.json"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "advent_core" / "prompts" / "facts.md"


# --- FACT_CATEGORIES / validate_key ----------------------------------------


def test_fact_categories_are_fixed():
    assert FACT_CATEGORIES == ("цель", "ограничения", "предпочтения", "решения", "договорённости")


def test_validate_key_accepts_known_category():
    assert validate_key("ограничения.бюджет") == "ограничения.бюджет"


def test_validate_key_rejects_unknown_category():
    with pytest.raises(ValueError):
        validate_key("погода.сегодня")


@pytest.mark.parametrize("bad", ["бюджет", "ограничения.", ".бюджет", ""])
def test_validate_key_rejects_malformed_key(bad):
    with pytest.raises(ValueError):
        validate_key(bad)


# --- apply_delta -------------------------------------------------------


def test_apply_delta_unmentioned_key_is_untouched():
    facts = {"ограничения.бюджет": "480"}
    delta = {"set": [{"key": "ограничения.срок", "value": "1 декабря"}], "delete": []}
    result = apply_delta(facts, (), delta)
    assert result.facts["ограничения.бюджет"] == "480"
    assert result.added == ("ограничения.срок",)


def test_apply_delta_delete_removes_key():
    facts = {"договорённости.логотип": "не делаем"}
    delta = {"set": [], "delete": ["договорённости.логотип"]}
    result = apply_delta(facts, (), delta)
    assert "договорённости.логотип" not in result.facts
    assert result.removed == ("договорённости.логотип",)


def test_apply_delta_delete_of_missing_key_is_noop():
    result = apply_delta({}, (), {"set": [], "delete": ["цель.основное"]})
    assert result.facts == {}
    assert result.removed == ()


def test_apply_delta_set_on_pinned_key_is_blocked_not_applied():
    facts = {"ограничения.бюджет": "480"}
    delta = {"set": [{"key": "ограничения.бюджет", "value": "999"}], "delete": []}
    result = apply_delta(facts, ("ограничения.бюджет",), delta)
    assert result.facts["ограничения.бюджет"] == "480"
    assert result.blocked == ("ограничения.бюджет",)
    assert result.updated == ()


def test_apply_delta_delete_on_pinned_key_is_blocked_not_applied():
    facts = {"ограничения.бюджет": "480"}
    delta = {"set": [], "delete": ["ограничения.бюджет"]}
    result = apply_delta(facts, ("ограничения.бюджет",), delta)
    assert result.facts == {"ограничения.бюджет": "480"}
    assert result.blocked == ("ограничения.бюджет",)
    assert result.removed == ()


def test_apply_delta_key_outside_categories_is_rejected_not_silently_applied():
    delta = {"set": [{"key": "погода.сегодня", "value": "дождь"}], "delete": []}
    result = apply_delta({}, (), delta)
    assert result.facts == {}
    assert result.rejected == ("погода.сегодня",)
    assert result.added == ()


def test_apply_delta_malformed_delete_key_is_rejected():
    result = apply_delta({}, (), {"set": [], "delete": ["бюджет"]})
    assert result.rejected == ("бюджет",)
    assert result.removed == ()


def test_apply_delta_distinguishes_added_from_updated():
    facts = {"цель.основное": "MVP"}
    delta = {
        "set": [
            {"key": "цель.основное", "value": "MVP v2"},
            {"key": "решения.платформа", "value": "Android"},
        ],
        "delete": [],
    }
    result = apply_delta(facts, (), delta)
    assert result.updated == ("цель.основное",)
    assert result.added == ("решения.платформа",)
    assert result.facts["цель.основное"] == "MVP v2"


def test_apply_delta_same_value_again_is_not_reported():
    facts = {"цель.основное": "MVP"}
    delta = {"set": [{"key": "цель.основное", "value": "MVP"}], "delete": []}
    result = apply_delta(facts, (), delta)
    assert result.updated == ()
    assert result.added == ()


def test_apply_delta_does_not_mutate_input_facts():
    facts = {"цель.основное": "MVP"}
    apply_delta(facts, (), {"set": [{"key": "цель.основное", "value": "v2"}], "delete": []})
    assert facts == {"цель.основное": "MVP"}


def test_apply_delta_missing_set_or_delete_keys_are_tolerated():
    result = apply_delta({}, (), {})
    assert result.facts == {}
    empty = ()
    assert result.added == result.updated == result.removed == result.blocked == empty
    assert result.rejected == empty


def test_delta_result_is_frozen():
    result = apply_delta({}, (), {"set": [], "delete": []})
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.added = ("x",)


# --- format_facts --------------------------------------------------------


def test_format_facts_empty_is_empty_string():
    assert format_facts({}, ()) == ""


def test_format_facts_orders_by_category_then_alphabetically():
    facts = {
        "ограничения.срок": "1 декабря",
        "ограничения.бюджет": "480",
        "цель.основное": "MVP",
    }
    text = format_facts(facts, ())
    assert text.index("цель") < text.index("ограничения")
    assert text.index("бюджет") < text.index("срок")


def test_format_facts_stable_across_calls():
    facts = {
        "договорённости.платёж": "СБП",
        "цель.основное": "MVP кофейни",
        "ограничения.бюджет": "480",
    }
    first = format_facts(facts, ())
    second = format_facts(dict(sorted(facts.items())), ())  # different input dict order
    assert first == second


def test_format_facts_marks_pinned_key():
    facts = {"ограничения.бюджет": "480"}
    marked = format_facts(facts, ("ограничения.бюджет",))
    unmarked = format_facts(facts, ())
    assert "\U0001f4cc" in marked
    assert "\U0001f4cc" not in unmarked


# --- render_note -----------------------------------------------------------


def test_render_note_none_when_nothing_changed():
    facts = {"цель.основное": "MVP"}
    assert render_note(facts, dict(facts)) is None


def test_render_note_none_on_two_empty_dicts():
    assert render_note({}, {}) is None


def test_render_note_matches_spec_example_shape():
    before = {"ограничения.бюджет": "480", "договорённости.платформа": "iOS"}
    after = {"ограничения.бюджет": "520", "ограничения.срок": "1 декабря"}
    note = render_note(before, after)
    assert note == "facts: ~бюджет, +срок, −платформа"


# --- facts_messages --------------------------------------------------------


def test_facts_messages_empty_when_no_facts():
    assert facts_messages({}, ()) == []


def test_facts_messages_shape_and_ack():
    facts = {"цель.основное": "MVP"}
    messages = facts_messages(facts, ())
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "MVP" in messages[0]["content"]
    assert messages[-1] == {"role": "assistant", "content": FACTS_ACK}


# --- schema file -------------------------------------------------------


def test_schema_file_is_pairs_shape_and_strict():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"set", "delete"}
    assert schema["properties"]["set"]["items"]["additionalProperties"] is False
    assert set(schema["properties"]["set"]["items"]["required"]) == {"key", "value"}


def test_schema_file_accepts_valid_delta():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    valid = {
        "set": [{"key": "цель.основное", "value": "MVP"}],
        "delete": ["договорённости.логотип"],
    }
    jsonschema.validate(valid, schema)  # must not raise


def test_schema_file_rejects_missing_value_field():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"set": [{"key": "цель.основное"}], "delete": []}, schema)


def test_schema_file_rejects_map_shape_for_set():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"set": {"цель.основное": "MVP"}, "delete": []}, schema)


def test_schema_file_rejects_unknown_top_level_field():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"set": [], "delete": [], "extra": 1}, schema)


# --- prompt file -------------------------------------------------------


def test_prompt_mentions_all_categories_and_key_rules():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for category in FACT_CATEGORIES:
        assert category in text
    assert "delete" in text
    assert "\U0001f4cc" in text  # pinned marker rule
    assert "additionalProperties" in text  # schema duplicated as text
