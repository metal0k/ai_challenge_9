"""ShipState: atomic writes, corrupt/empty reads, hashes and dependent invalidation."""

import json
import os
from pathlib import Path

import pytest

from adventship.ctx import ShipError
from adventship.state import DEPENDENTS, STAGES, UPSTREAM, ShipState


def make(tmp_path):
    return ShipState(tmp_path / "ship" / "state.json")


def test_missing_file_is_empty(tmp_path):
    state = ShipState.load(tmp_path / "state.json")
    assert state.task_hash is None
    assert state.status("commit") == "pending"


def test_roundtrip(tmp_path):
    state = make(tmp_path)
    state.set_task_hash("abc")
    state.mark("commit", "done", at="t1", data={"sha": "1" * 40})
    loaded = ShipState.load(state.path)
    assert loaded.task_hash == "abc"
    assert loaded.data("commit") == {"sha": "1" * 40}
    assert loaded.status("commit") == "done"
    assert loaded.is_valid("commit")


def test_corrupt_json_is_backed_up_and_reported(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ShipError) as caught:
        ShipState.load(path)
    assert caught.value.exit_code == 10
    assert not path.exists()
    assert (tmp_path / "state.json.bak").read_text(encoding="utf-8") == "{not json"


def test_wrong_version_counts_as_corrupt(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "stages": {}}), encoding="utf-8")
    with pytest.raises(ShipError):
        ShipState.load(path)
    assert (tmp_path / "state.json.bak").exists()


def test_interrupted_write_keeps_previous_state(tmp_path, monkeypatch):
    state = make(tmp_path)
    state.mark("checks", "done", at="t1", data={"n": 1})

    def boom(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        state.mark("describe", "done", at="t2", data={"text": "x"})
    monkeypatch.undo()

    reloaded = ShipState.load(state.path)
    assert reloaded.status("checks") == "done"
    assert reloaded.status("describe") == "pending"
    assert not state.path.with_name("state.json.tmp").exists()


def test_replace_retries_on_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr("adventship.state._RETRY_DELAY", 0)
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    state = make(tmp_path)
    state.mark("checks", "done", at="t", data={})
    monkeypatch.undo()
    assert len(calls) == 3
    assert ShipState.load(state.path).status("checks") == "done"


def test_read_retries_on_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr("adventship.state._RETRY_DELAY", 0)
    state = make(tmp_path)
    state.mark("checks", "done", at="t", data={})
    real = Path.read_text
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("locked")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert ShipState.load(state.path).status("checks") == "done"
    assert len(calls) == 2


def test_dependency_graph_matches_spec_chain():
    chain = ["commit", "tag", "record", "submit", "sheet"]
    for index, stage in enumerate(chain):
        assert set(chain[index + 1 :]) <= set(DEPENDENTS[stage])
    assert set(UPSTREAM["sheet"]) >= {"intake", "describe", "commit", "tag", "record", "submit"}
    assert STAGES[0] == "intake" and STAGES[-1] == "report"


def test_changed_output_invalidates_dependents(tmp_path):
    state = make(tmp_path)
    for stage in ("commit", "tag", "record", "submit", "sheet"):
        state.mark(stage, "done", at="t", data={"v": stage})
    assert state.is_valid("sheet")

    state.mark("commit", "done", at="t2", data={"v": "other-sha"})
    assert state.status("commit") == "done"
    for stage in ("tag", "record", "submit", "sheet"):
        assert state.status(stage) == "pending"


def test_identical_output_keeps_dependents(tmp_path):
    state = make(tmp_path)
    state.mark("commit", "done", at="t", data={"sha": "a"})
    state.mark("tag", "done", at="t", data={"tag": "x"})
    state.mark("commit", "done", at="t2", data={"sha": "a"})
    assert state.status("tag") == "done"
    assert state.is_valid("tag")


def test_volatile_keys_do_not_change_hashes(tmp_path):
    state = make(tmp_path)
    state.mark("commit", "done", at="t", data={"sha": "a", "_ms": 1})
    state.mark("tag", "done", at="t", data={})
    state.mark("commit", "done", at="t", data={"sha": "a", "_ms": 999})
    assert state.is_valid("tag")


def test_upstream_data_change_makes_downstream_stale(tmp_path):
    state = make(tmp_path)
    state.mark("describe", "done", at="t", data={"text": "one"})
    state.mark("commit", "done", at="t", data={"sha": "a"})
    # a change that bypassed mark() (state edited by hand)
    state.doc["stages"]["describe"]["data"]["text"] = "two"
    assert not state.is_valid("commit")


def test_task_hash_change_wipes_stages(tmp_path):
    state = make(tmp_path)
    state.set_task_hash("one")
    state.mark("checks", "done", at="t", data={})
    assert state.set_task_hash("one") is False
    assert state.status("checks") == "done"
    assert state.set_task_hash("two") is True
    assert state.status("checks") == "pending"


def test_invalidate_from_covers_later_stages_only(tmp_path):
    state = make(tmp_path)
    for stage in STAGES:
        state.mark(stage, "done", at="t", data={})
    removed = state.invalidate_from("review")
    assert removed == STAGES[STAGES.index("review") :]
    assert state.status("describe") == "done"
    assert state.status("intake") == "done"


def test_failed_stage_is_not_valid(tmp_path):
    state = make(tmp_path)
    state.mark("checks", "failed", at="t", data={"error": "x"})
    assert not state.is_valid("checks")
