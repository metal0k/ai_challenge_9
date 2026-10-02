"""Day 25: task state in working memory (SPEC-w05d25.md §2 and §9a.5-7)."""

import pytest

from advent_core import memory
from advent_core.memory import (
    MemoryDelta,
    MemoryOperation,
    MemorySnapshot,
    MemoryStore,
    StructuredMemory,
    apply_delta,
    render_delta,
)


def _user(text: str) -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": text},)


def _msgs(*pairs: tuple[str, str]) -> tuple[dict[str, str], ...]:
    return tuple({"role": role, "content": text} for role, text in pairs)


def _set(field: str, key: str, value: str, evidence: str) -> MemoryOperation:
    return MemoryOperation("set", field, key, value, evidence)


def _working(entries: dict[str, str], pinned: frozenset[str] = frozenset()) -> MemorySnapshot:
    return MemorySnapshot(working=StructuredMemory(entries, pinned))


def test_working_fields_are_the_six_in_order() -> None:
    assert memory.WORKING_FIELDS == (
        "goal",
        "clarified",
        "constraints",
        "terms",
        "decisions",
        "open_items",
    )


def test_new_fields_are_accepted_by_apply_delta() -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(
            working=(
                _set("clarified", "os", "Windows Terminal", "я на Windows Terminal"),
                _set("terms", "dup", "дубль = запись", "дубль = запись"),
            )
        ),
        user_messages=_user("я на Windows Terminal, дубль = запись"),
        memory_upto=1,
    )
    assert update.rejected == ()
    assert set(update.snapshot.working.entries) == {"clarified.os", "terms.dup"}


def test_old_file_without_new_fields_loads(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    store.save_working("s", StructuredMemory({"goal.primary": "g"}, frozenset()), 2)
    loaded = store.load_working("s", turns_count=2)
    assert loaded.value.entries == {"goal.primary": "g"}
    assert loaded.warnings == ()


@pytest.mark.parametrize(
    "evidence",
    ["user: я на Windows Terminal", "Пользователь:  я на Windows Terminal\n  "],
)
def test_role_prefix_is_stripped_from_evidence(evidence: str) -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("clarified", "os", "Windows Terminal", evidence),)),
        user_messages=_user("привет. я на Windows Terminal"),
        memory_upto=1,
    )
    assert update.rejected == ()
    assert update.snapshot.working.entries == {"clarified.os": "Windows Terminal"}


def test_multiline_evidence_inside_one_message_is_accepted() -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("clarified", "x", "v", "user: первая строка\nвторая строка"),)),
        user_messages=_user("первая строка\nвторая строка\nтретья"),
        memory_upto=1,
    )
    assert update.rejected == ()


def test_evidence_spliced_from_two_messages_is_rejected() -> None:
    messages = _msgs(("user", "цель: записать демо"), ("assistant", "ок"), ("user", "только 14b"))
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(
            working=(_set("clarified", "x", "v", "user: цель: записать демо\nuser: только 14b"),)
        ),
        user_messages=messages,
        memory_upto=1,
    )
    assert update.snapshot.working.entries == {}
    assert len(update.rejected) == 1


@pytest.mark.parametrize("evidence", ["user:", "user:   \n  ", "   "])
def test_evidence_empty_after_normalisation_is_rejected(evidence: str) -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("clarified", "x", "v", evidence),)),
        user_messages=_user("user: что-то"),
        memory_upto=1,
    )
    assert update.snapshot.working.entries == {}
    assert len(update.rejected) == 1


def test_evidence_found_only_in_an_assistant_message_is_rejected() -> None:
    messages = _msgs(("user", "вопрос"), ("assistant", "Не знаю: нет в документации"))
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("clarified", "x", "v", "Не знаю: нет в документации"),)),
        user_messages=messages,
        memory_upto=1,
    )
    assert update.snapshot.working.entries == {}
    assert len(update.rejected) == 1


def test_mixed_delta_applies_valid_and_rejects_invalid_operations() -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(
            working=(
                _set("terms", "dup", "дубль = запись", "дубль = запись record"),
                _set("terms", "bad", "v", "нет такого в сообщении"),
                _set("nofield", "k", "v", "дубль"),
            )
        ),
        user_messages=_user("дубль = запись record"),
        memory_upto=5,
    )
    assert update.snapshot.working.entries == {"terms.dup": "дубль = запись"}
    assert [op.canonical_key for _, op, _ in update.rejected] == ["terms.bad", "nofield.k"]
    assert update.memory_upto == 5


def test_first_automatic_goal_is_applied_and_pinned() -> None:
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("goal", "main", "записать демо", "записать демо"),)),
        user_messages=_user("хочу записать демо"),
        memory_upto=1,
    )
    assert update.snapshot.working.entries == {"goal.main": "записать демо"}
    assert update.snapshot.working.pinned == frozenset({"goal.main"})
    assert update.blocked == ()


def test_pinned_goal_blocks_another_goal_key_and_a_delete() -> None:
    first = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set("goal", "main", "G", "хочу демо"),)),
        user_messages=_user("хочу демо"),
        memory_upto=1,
    )
    update = apply_delta(
        first.snapshot,
        MemoryDelta(
            working=(
                _set("goal", "other", "H", "про бюджет"),
                MemoryOperation("delete", "goal", "main", None, "про бюджет"),
            )
        ),
        user_messages=_user("кстати про бюджет"),
        memory_upto=2,
    )
    assert update.snapshot.working.entries == {"goal.main": "G"}
    assert [(op.canonical_key, why) for _, op, why in update.blocked] == [
        ("goal.other", "goal pinned"),
        ("goal.main", "goal pinned"),
    ]
    assert render_delta(update) == "memory: working !goal.other, !goal.main"


def test_manual_set_replaces_a_pinned_goal() -> None:
    snapshot = _working({"goal.main": "G"}, frozenset({"goal.main"}))
    replaced = memory.manual_set(snapshot, "working", "goal.new", "H")
    assert replaced.working.entries == {"goal.new": "H"}
    assert replaced.working.pinned == frozenset({"goal.new"})


def test_manual_set_with_several_legacy_goals_leaves_exactly_one() -> None:
    snapshot = _working({"goal.a": "1", "goal.b": "2", "goal.c": "3"}, frozenset({"goal.a"}))
    after = memory.manual_set(snapshot, "working", "goal.b", "новая")
    assert after.working.entries == {"goal.b": "новая"}
    assert after.working.pinned == frozenset({"goal.b"})


def test_legacy_unpinned_goals_are_migrated_on_the_first_goal_set() -> None:
    snapshot = _working({"goal.a": "старая", "goal.b": "ещё", "constraints.c": "коротко"})
    update = apply_delta(
        snapshot,
        MemoryDelta(working=(_set("goal", "main", "новая", "новая цель"),)),
        user_messages=_user("новая цель"),
        memory_upto=1,
    )
    assert update.snapshot.working.entries == {"goal.main": "новая", "constraints.c": "коротко"}
    assert update.snapshot.working.pinned == frozenset({"goal.main"})


def test_task_context_empty_working_is_none() -> None:
    assert memory.task_context(MemorySnapshot()) is None
    assert memory.task_context(_working({"decisions.a": "x", "open_items.b": "y"})) is None


def test_task_context_order_and_excluded_fields() -> None:
    snapshot = _working(
        {
            "terms.dup": "дубль = запись",
            "decisions.d": "не попадёт",
            "constraints.len": "коротко",
            "open_items.o": "не попадёт",
            "clarified.os": "Windows Terminal",
            "goal.main": "записать демо",
        }
    )
    assert memory.task_context(snapshot) == (
        "Цель: записать демо\n"
        "Уточнено: Windows Terminal\n"
        "Ограничение: коротко\n"
        "Термин: дубль = запись"
    )


def test_task_context_is_cut_by_whole_entry_within_1200_chars() -> None:
    entries = {"goal.main": "цель"}
    entries.update({f"clarified.k{i:02d}": "я" * 200 for i in range(10)})
    text = memory.task_context(_working(entries))
    assert text is not None
    assert len(text) <= 1200
    lines = text.split("\n")
    assert lines[0] == "Цель: цель"
    assert len(lines) == 6  # goal + five whole 213-char entries; a sixth would overflow
    assert all(line.endswith("я" * 200) for line in lines[1:])


def test_task_context_single_oversized_goal_is_truncated_not_dropped() -> None:
    text = memory.task_context(_working({"goal.main": "ж" * 5000}))
    assert text is not None
    assert len(text) == 1200
    assert text.endswith("…")


def test_task_context_fields_argument_narrows_the_text() -> None:
    snapshot = _working(
        {
            "goal.main": "записать демо",
            "terms.dup": "дубль = запись",
            "clarified.os": "Windows Terminal",
            "constraints.len": "без select",
        }
    )
    assert memory.task_context(snapshot, memory.RETRIEVAL_CONTEXT_FIELDS) == (
        "Цель: записать демо\nТермин: дубль = запись"
    )
    assert memory.task_context(snapshot, ("clarified",)) == "Уточнено: Windows Terminal"
    assert memory.task_context(_working({"clarified.os": "x"}), ("goal", "terms")) is None


@pytest.mark.parametrize("field", ["clarified", "constraints", "terms", "open_items"])
def test_automatic_op_with_a_question_as_evidence_is_rejected(field: str) -> None:
    question = "Теперь про простой в дубле: чем вырезать мёртвое время?"
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(working=(_set(field, "dead_space", "вырезать мёртвое время", question),)),
        user_messages=_user(question),
        memory_upto=1,
    )

    assert update.applied == ()
    assert update.snapshot.working.entries == {}
    assert [reason for _, _, reason in update.rejected] == ["evidence is a question"]


def test_question_evidence_is_fine_for_goal_and_for_a_statement_ending_before_the_question() -> (
    None
):
    text = "Хочу записать демо. Как начать?"
    update = apply_delta(
        MemorySnapshot(),
        MemoryDelta(
            working=(
                _set("goal", "main", "записать демо", "Как начать?"),
                _set("clarified", "x", "записать демо", "Хочу записать демо."),
            )
        ),
        user_messages=_user(text),
        memory_upto=1,
    )

    assert update.rejected == ()
    assert set(update.snapshot.working.entries) == {"goal.main", "clarified.x"}
