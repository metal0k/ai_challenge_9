"""Branching core: naming, checkpoint/branch snapshots, tree listing.

Pure file/dict work, no network — same posture as test_session.py.
"""

from __future__ import annotations

import pytest

from advent_core.config import ConfigError
from advent_core.session import (
    CONTENT_STATE_KEYS,
    Session,
    branch_file_name,
    build_tree,
    checkpoint_file_name,
    delete_branch,
    make_branch,
    make_checkpoint,
    parse_name,
    root_of,
    validate_name,
)

# --- naming -----------------------------------------------------------------


def test_double_dash_combined_name_already_passes_validate_name():
    # SPEC-w02d10.md §7.2: the branch separator needs no regex change.
    assert validate_name("root--mvp") == "root--mvp"


def test_branch_file_name_shape():
    assert branch_file_name("demo10", "cheap") == "demo10--cheap"


def test_checkpoint_file_name_shape():
    assert checkpoint_file_name("demo10", "mvp") == "demo10--cp-mvp"


def test_branch_segment_starting_with_cp_prefix_is_rejected():
    with pytest.raises(ConfigError):
        branch_file_name("demo10", "cp-mvp")


def test_branch_segment_cannot_contain_the_tree_separator():
    with pytest.raises(ConfigError):
        branch_file_name("demo10", "mvp--lite")


def test_over_long_combined_name_raises_instead_of_truncating():
    long_parent = "p" * 60  # + "--cheap" = 67 chars, over the 64 limit
    with pytest.raises(ConfigError):
        branch_file_name(long_parent, "cheap")


def test_parse_name_root_branch_and_checkpoint():
    assert parse_name("demo10") == (None, "demo10", False)
    assert parse_name("demo10--cheap") == ("demo10", "cheap", False)
    assert parse_name("demo10--cp-mvp") == ("demo10", "cp-mvp", True)
    # Immediate parent, not the topmost root.
    assert parse_name("root--mvp--lite") == ("root--mvp", "lite", False)


def test_parse_name_does_not_mistake_a_root_for_a_checkpoint():
    # No "--" at all: a root literally named "cp-foo" is still just a root.
    assert parse_name("cp-foo") == (None, "cp-foo", False)


def test_root_of_walks_up_by_name_alone():
    assert root_of("root--mvp--lite") == "root"
    assert root_of("demo10--cp-mvp") == "demo10"
    assert root_of("demo10") == "demo10"


# --- make_checkpoint ---------------------------------------------------------


def test_make_checkpoint_writes_a_full_snapshot_with_kind(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.record("первый", "ответ", model="m")
    parent.save()

    checkpoint = make_checkpoint(parent, "mvp")

    assert checkpoint.name == "demo10--cp-mvp"
    assert checkpoint.path == tmp_path / "demo10--cp-mvp.json"
    assert checkpoint.path.is_file()
    assert checkpoint.state["kind"] == "checkpoint"
    assert len(checkpoint.turns) == 2


def test_checkpoint_snapshot_is_independent_of_later_parent_mutation(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.record("первый", "ответ", model="m")
    parent.save()

    checkpoint = make_checkpoint(parent, "mvp")

    parent.record("второй", "ответ", model="m")
    parent.state["context_limit"] = 999
    parent.save()

    assert len(checkpoint.turns) == 2
    assert "context_limit" not in checkpoint.state

    reloaded = Session.load("demo10--cp-mvp", directory=tmp_path)
    assert len(reloaded.turns) == 2
    assert "context_limit" not in reloaded.state


# --- make_branch --------------------------------------------------------------


def test_branch_write_does_not_modify_the_parent_file(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.record("в", "о", model="m")
    parent.save()
    before = parent.path.read_text(encoding="utf-8")

    branch = make_branch(parent, "cheap")
    branch.record("новый вопрос", "новый ответ", model="m")
    branch.save()

    assert parent.path.read_text(encoding="utf-8") == before
    assert branch.path == tmp_path / "demo10--cheap.json"


def test_branch_from_checkpoint_attaches_to_the_checkpoints_root(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.save()
    checkpoint = make_checkpoint(parent, "mvp")

    branch = make_branch(checkpoint, "cheap")

    # NOT "demo10--cp-mvp--cheap" — a checkpoint is a snapshot to fork FROM.
    assert branch.name == "demo10--cheap"
    assert branch.state["parent"] == "demo10"
    assert branch.state["fork_at"] == "demo10--cp-mvp"
    assert "kind" not in branch.state


def test_branch_directly_off_a_session_has_no_fork_at(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.save()

    branch = make_branch(parent, "cheap")

    assert branch.state["parent"] == "demo10"
    assert branch.state["fork_at"] is None


def test_nested_branch_round_trips(tmp_path):
    root = Session.new("root", directory=tmp_path)
    root.save()

    mvp = make_branch(root, "mvp")
    assert mvp.name == "root--mvp"

    lite = make_branch(mvp, "lite")
    assert lite.name == "root--mvp--lite"
    assert lite.state["parent"] == "root--mvp"

    reloaded = Session.load("root--mvp--lite", directory=tmp_path)
    assert reloaded.warnings == []
    assert reloaded.state["parent"] == "root--mvp"


def test_branch_copies_turns_and_settings_from_its_source(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.record("в", "о", model="m")
    parent.state["mode"] = "dialog"
    parent.state["context_strategy"] = "facts"
    parent.save()

    branch = make_branch(parent, "cheap")

    assert len(branch.turns) == 2
    assert branch.state["mode"] == "dialog"
    assert branch.state["context_strategy"] == "facts"


# --- delete_branch ------------------------------------------------------------


def test_delete_branch_removes_the_file(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.save()
    branch = make_branch(parent, "cheap")
    assert branch.path.is_file()

    delete_branch("demo10--cheap", directory=tmp_path)

    assert not branch.path.is_file()


def test_delete_branch_refuses_on_a_checkpoint(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.save()
    checkpoint = make_checkpoint(parent, "mvp")

    with pytest.raises(ConfigError):
        delete_branch(checkpoint.name, directory=tmp_path)
    assert checkpoint.path.is_file()


def test_delete_branch_refuses_on_a_root(tmp_path):
    parent = Session.new("demo10", directory=tmp_path)
    parent.save()

    with pytest.raises(ConfigError):
        delete_branch("demo10", directory=tmp_path)
    assert parent.path.is_file()


def test_delete_branch_refuses_on_a_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        delete_branch("no-such-branch", directory=tmp_path)


# --- tree listing ---------------------------------------------------------


def test_build_tree_groups_roots_branches_and_checkpoints():
    entries = build_tree(["demo10", "demo10--cheap", "demo10--cp-mvp"])
    by_name = {e.name: e for e in entries}

    assert by_name["demo10"].parent is None
    assert by_name["demo10"].is_checkpoint is False
    assert by_name["demo10"].depth == 0

    assert by_name["demo10--cheap"].parent == "demo10"
    assert by_name["demo10--cheap"].is_checkpoint is False
    assert by_name["demo10--cheap"].root == "demo10"
    assert by_name["demo10--cheap"].depth == 1

    assert by_name["demo10--cp-mvp"].is_checkpoint is True
    assert by_name["demo10--cp-mvp"].orphaned is False


def test_orphaned_branch_is_reported_not_crashed():
    # "root--mvp" itself is missing, only "root" and its grandchild exist.
    entries = build_tree(["root", "root--mvp--lite"])
    by_name = {e.name: e for e in entries}

    assert by_name["root"].orphaned is False
    orphan = by_name["root--mvp--lite"]
    assert orphan.orphaned is True
    assert orphan.parent is None
    assert orphan.root == "root"
    assert orphan.depth == 2


def test_build_tree_on_empty_input():
    assert build_tree([]) == []


# --- CONTENT_STATE_KEYS / clear() -------------------------------------------


def test_content_state_keys_includes_facts_triplet():
    assert set(("facts", "facts_pinned", "facts_upto")) <= set(CONTENT_STATE_KEYS)


def test_clear_drops_facts_but_keeps_settings_including_branch_pointers(tmp_path):
    session = Session.new("demo10--cheap", directory=tmp_path)
    session.state.update(
        {
            "mode": "dialog",
            "context_limit": 2000,
            "context_strategy": "facts",
            "parent": "demo10",
            "fork_at": "demo10--cp-mvp",
            "facts": {"цель.проект": "кофейня"},
            "facts_pinned": ["цель.проект"],
            "facts_upto": 3,
            "summary": "пересказ",
            "summary_upto": 5,
        }
    )
    session.record("в", "о", model="m")

    session.clear()

    assert session.turns == []
    for key in ("facts", "facts_pinned", "facts_upto", "summary", "summary_upto"):
        assert key not in session.state
    assert session.state["mode"] == "dialog"
    assert session.state["context_limit"] == 2000
    assert session.state["context_strategy"] == "facts"
    assert session.state["parent"] == "demo10"
    assert session.state["fork_at"] == "demo10--cp-mvp"
