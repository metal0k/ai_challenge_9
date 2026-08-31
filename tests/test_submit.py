"""Построение ссылок и путей для сдачи — без обращения к сети."""

from __future__ import annotations

import pytest

from advent_cli.links import disk_path, find_sync_root, parse_repo, tag_url
from advent_cli.submit import day_tag, video_name
from advent_core.errors import AdventError

REPO = "https://github.com/metal0k/ai_challenge_9.git"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/metal0k/ai_challenge_9.git",
        "https://github.com/metal0k/ai_challenge_9",
        "https://github.com/metal0k/ai_challenge_9/",
        "git@github.com:metal0k/ai_challenge_9.git",
    ],
)
def test_repo_is_parsed_from_every_url_form(url):
    assert parse_repo(url) == ("metal0k", "ai_challenge_9")


def test_broken_url_is_reported():
    with pytest.raises(AdventError):
        parse_repo("https://gitlab.com/who/what")


def test_tag_url_points_at_tree():
    assert tag_url(REPO, "w01d01") == "https://github.com/metal0k/ai_challenge_9/tree/w01d01"


def test_tag_and_video_naming_are_zero_padded():
    assert day_tag(1, 1) == "w01d01"
    assert day_tag(9, 7) == "w09d07"
    assert video_name(1, 1) == "0101.mp4"
    assert video_name(12, 3) == "1203.mp4"


def test_sync_root_is_found_by_marker(tmp_path):
    root = tmp_path / "YaDisk"
    (root / ".sync").mkdir(parents=True)
    deep = root / "Media" / "Study" / "AI_Challenge_9"
    deep.mkdir(parents=True)
    assert find_sync_root(deep) == root


def test_sync_root_is_none_without_marker(tmp_path):
    assert find_sync_root(tmp_path) is None


def test_disk_path_is_posix_relative_to_root(tmp_path, monkeypatch):
    monkeypatch.delenv("YANDEX_DISK_LOCAL_ROOT", raising=False)
    root = tmp_path / "YaDisk"
    (root / ".sync").mkdir(parents=True)
    video = root / "Media" / "Study" / "AI_Challenge_9" / "0101.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    assert disk_path(video) == "/Media/Study/AI_Challenge_9/0101.mp4"


def test_disk_path_respects_explicit_root(tmp_path, monkeypatch):
    root = tmp_path / "Custom"
    video = root / "sub" / "0101.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    monkeypatch.setenv("YANDEX_DISK_LOCAL_ROOT", str(root))

    assert disk_path(video) == "/sub/0101.mp4"


def test_file_outside_disk_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "Root"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "0101.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    monkeypatch.setenv("YANDEX_DISK_LOCAL_ROOT", str(root))

    with pytest.raises(AdventError):
        disk_path(outside)
