import json

import pytest

from advent_core import memory
from advent_core.config import ConfigError
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


def test_apply_delta_validates_both_sections_before_mutating() -> None:
    snapshot = MemorySnapshot()
    delta = MemoryDelta(
        working=(MemoryOperation("set", "goal", "primary", "ship", "ship"),),
        long_term=(MemoryOperation("set", "profile", "name", "Denis", "not said"),),
    )

    with pytest.raises(ValueError, match="evidence"):
        apply_delta(snapshot, delta, user_messages=_user("ship"), memory_upto=9)

    assert snapshot == MemorySnapshot()


@pytest.mark.parametrize(
    "operation, message, pattern",
    [
        (MemoryOperation("set", "goal", "primary", "x", "x"), "x", "duplicate"),
        (MemoryOperation("set", "preferences", "token", "abc", "abc"), "abc", "credential"),
    ],
)
def test_apply_delta_invalid_delta_raises_and_keeps_cursor(
    operation: MemoryOperation, message: str, pattern: str
) -> None:
    if pattern == "duplicate":
        delta = MemoryDelta(working=(operation, operation))
    else:
        delta = MemoryDelta(long_term=(operation,))

    with pytest.raises(ValueError, match=pattern):
        apply_delta(MemorySnapshot(), delta, user_messages=_user(message), memory_upto=12)


def test_apply_delta_pinned_conflict_blocks_only_that_operation() -> None:
    snapshot = MemorySnapshot(
        working=StructuredMemory({"goal.primary": "old"}, frozenset({"goal.primary"}))
    )
    delta = MemoryDelta(
        working=(
            MemoryOperation("set", "goal", "primary", "new", "new"),
            MemoryOperation("set", "decisions", "storage", "JSON", "JSON"),
        ),
        long_term=(MemoryOperation("set", "profile", "name", "Denis", "Denis"),),
    )

    update = apply_delta(snapshot, delta, user_messages=_user("new JSON Denis"), memory_upto=4)

    assert update.memory_upto == 4
    assert update.snapshot.working.entries == {
        "goal.primary": "old",
        "decisions.storage": "JSON",
    }
    assert update.snapshot.long_term.entries == {"profile.name": "Denis"}
    assert len(update.blocked) == 1
    assert update.blocked[0][1].canonical_key == "goal.primary"


def test_render_delta_uses_prior_layer_state_for_all_markers() -> None:
    snapshot = MemorySnapshot(
        working=StructuredMemory({"goal.primary": "old", "decisions.storage": "old"}),
        long_term=StructuredMemory({"profile.name": "old", "knowledge.stack": "Python"}),
    )
    delta = MemoryDelta(
        working=(
            MemoryOperation("set", "goal", "primary", "new", "new"),
            MemoryOperation("set", "constraints", "deadline", "today", "today"),
            MemoryOperation("delete", "decisions", "storage", None, "old"),
        ),
        long_term=(
            MemoryOperation("set", "profile", "name", "new", "new"),
            MemoryOperation("set", "preferences", "language", "Russian", "Russian"),
            MemoryOperation("delete", "knowledge", "stack", None, "Python"),
        ),
    )

    update = apply_delta(
        snapshot,
        delta,
        user_messages=_user("new today old new Russian Python"),
    )

    assert render_delta(update) == (
        "memory: working ~goal.primary, +constraints.deadline, −decisions.storage; "
        "long-term ~profile.name, +preferences.language, −knowledge.stack"
    )


def test_store_roundtrip_and_safe_session_name(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    value = StructuredMemory({"goal.primary": "ship"}, frozenset({"goal.primary"}))

    working_path = store.save_working(" demo ", value, 3)
    store.save_long_term(StructuredMemory({"profile.name": "Denis"}))

    loaded = store.load_working("demo", turns_count=5)
    assert working_path == tmp_path / "memory" / "working" / "demo.json"
    assert loaded.value == value
    assert loaded.upto == 3
    assert store.load_long_term().value.entries == {"profile.name": "Denis"}
    with pytest.raises(ConfigError):
        store.working_path("../../outside")


@pytest.mark.parametrize("field, value", [("entries", []), ("pinned", {})])
def test_load_rejects_wrong_root_shapes(tmp_path, field: str, value: object) -> None:
    store = MemoryStore(tmp_path)
    path = store.long_term_path
    path.parent.mkdir(parents=True)
    payload = {
        "version": 1,
        "layer": "long_term",
        "entries": {},
        "pinned": [],
    }
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load_long_term()

    assert loaded.value == StructuredMemory()
    assert loaded.warnings


def test_load_rejects_boolean_version_and_keeps_valid_file_untouched(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    path = store.long_term_path
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"version": True, "layer": "long_term", "entries": {}, "pinned": []}),
        encoding="utf-8",
    )

    loaded = store.load_long_term()

    assert loaded.value == StructuredMemory()
    assert "wrong version" in loaded.warnings[0]
    assert json.loads(path.read_text(encoding="utf-8"))["version"] is True


def test_load_warns_per_invalid_entry_and_keeps_the_rest(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    path = store.long_term_path
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "layer": "long_term",
                "entries": {
                    "profile.name": "Denis",
                    "profile.empty": " ",
                    "not_allowed.key": "ignored",
                },
                "pinned": ["profile.name", "profile.missing", 4],
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load_long_term()

    assert loaded.value.entries == {"profile.name": "Denis"}
    assert loaded.value.pinned == frozenset({"profile.name"})
    assert len(loaded.warnings) == 4


def test_invalid_working_upto_falls_back_to_transcript_end(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    path = store.working_path("demo")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "layer": "working",
                "session": "demo",
                "entries": {},
                "pinned": [],
                "upto": 99,
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load_working("demo", turns_count=7)

    assert loaded.upto == 7
    assert loaded.warnings


def test_save_rejects_noncanonical_keys_before_writing(tmp_path) -> None:
    store = MemoryStore(tmp_path)

    with pytest.raises(ValueError, match="memory key"):
        store.save_long_term(StructuredMemory({"profile.bad key": "x"}))

    assert not store.long_term_path.exists()


def test_atomic_write_removes_temp_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path)

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(memory.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save_long_term(StructuredMemory({"profile.name": "Denis"}))

    assert list(store.long_term_path.parent.glob(".long_term.json.*.tmp")) == []
