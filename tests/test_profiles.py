from pathlib import Path

import pytest

from advent_core.config import ConfigError
from advent_core.profiles import (
    delete,
    list_names,
    load,
    messages,
    save,
    validate_name,
    validate_value,
)
from advent_core.session import Session, make_branch


def test_profile_round_trip_and_listing(tmp_path: Path):
    save("formal", {"style": "formal", "format": "bullets"}, tmp_path)
    assert list_names(tmp_path) == ["formal"]
    assert load("formal", tmp_path)["style"] == "formal"
    assert messages(load("formal", tmp_path))[0]["role"] == "user"


def test_profile_rejects_unsafe_names_and_credentials(tmp_path: Path):
    with pytest.raises(ConfigError):
        validate_name("../secrets")
    with pytest.raises(ConfigError):
        validate_value("Bearer abc")
    with pytest.raises(ConfigError):
        save("safe", {"style": "ok", "token": "secret"}, tmp_path)
    with pytest.raises(ConfigError):
        save("empty", {"style": ""}, tmp_path)
    assert not (tmp_path / "safe.json").exists()


def test_profile_save_rejects_unsafe_update_without_overwriting_existing_file(
    tmp_path: Path,
):
    save("developer", {"style": "technical", "format": "bullets"}, tmp_path)
    path = tmp_path / "developer.json"
    before = path.read_bytes()

    with pytest.raises(ConfigError):
        save("developer", {"style": "technical", "api_key": "secret"}, tmp_path)

    assert path.read_bytes() == before


def test_legacy_credential_fields_are_omitted(tmp_path: Path):
    (tmp_path / "old.json").write_text(
        '{"style": "brief", "api_key": "do-not-show", "format": "plain"}',
        encoding="utf-8",
    )
    assert load("old", tmp_path) == {"style": "brief", "format": "plain"}


def test_profile_delete(tmp_path: Path):
    save("brief", {"style": "brief"}, tmp_path)
    delete("brief", tmp_path)
    assert list_names(tmp_path) == []


def test_active_profile_state_is_additive_and_survives_clear_and_branch(tmp_path: Path):
    session = Session.new("root", directory=tmp_path)
    session.state["active_profile"] = "formal"
    session.record("q", "a")
    session.clear()
    session.save()
    loaded = Session.load("root", directory=tmp_path)
    assert loaded.state["active_profile"] == "formal"
    branch = make_branch(loaded, "brief", directory=tmp_path)
    assert branch.state["active_profile"] == "formal"
