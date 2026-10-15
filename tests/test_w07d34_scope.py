"""Day 34: path canonicalisation and policy (pure, no model, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adventassistant import files_scope as scope
from adventassistant.files_scope import ScopeError
from tests.w07d34_repo import make_dir_link, make_repo


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "..\\x.md",
        "a\\..\\..\\x.md",
        "a/../x.md",
        "./x.md",
        "/etc/x.md",
        "C:/x.md",
        "C:x.md",
        "\\\\server\\share\\x.md",
        "//server/x.md",
        "\\\\?\\C:\\x.md",
        "\\\\.\\pipe\\x",
        "a.md:stream",
        "a//b.md",
        "a/b.md/",
        "a/b.",
        "a/b.md ",
        "con.md",
        "docs/NUL.txt",
        "docs/Com1.tar.gz",
        "lpt9",
        "a\x00b.md",
        "a\nb.md",
        "a*.md",
        "GIT~1/config",
        "x" * 261,
    ],
)
def test_canonical_rejects(bad: str) -> None:
    with pytest.raises(ScopeError):
        scope.canonical(bad)


def test_canonical_rejects_non_strings() -> None:
    for bad in (None, 5, b"a.md", ["a.md"]):
        with pytest.raises(ScopeError):
            scope.canonical(bad)


def test_canonical_accepts_and_normalises_separators() -> None:
    assert scope.canonical("docs\\a.md") == "docs/a.md"
    assert scope.canonical("console.md") == "console.md"
    assert scope.canonical("x" * 260) == "x" * 260


def test_dot_git_is_denied_case_insensitively() -> None:
    assert scope.is_denied(".GIT/config")
    assert scope.is_denied("sub/.git/HEAD")
    assert not scope.is_denied(".gitignore")


@pytest.mark.parametrize(
    "name", [".env", ".ENV.local", "a/.env.example", "k/x.PEM", "x.key", "d/a.sqlite3", "UV.LOCK"]
)
def test_deny_list(name: str) -> None:
    assert scope.is_denied(name)
    with pytest.raises(ScopeError):
        scope.check_readable_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "src/app.py",
        "notes.docx",
        ".github/x.md",
        ".GITHUB/x.md",
        "specs/x.md",
        "SPECS/x.md",
        "CLAUDE.md",
        "sub/claude.md",
        "AGENTS.md",
        "README.md",
        "readme.MD",
        "logs/x.txt",
        "data/x.md",
        ".env.md",
    ],
)
def test_write_name_refusals(name: str) -> None:
    with pytest.raises(ScopeError):
        scope.check_write_name(name)


@pytest.mark.parametrize("name", ["week_07/README.md", "docs/n.TXT", "a.md", "sub/readme.md"])
def test_write_name_allowed(name: str) -> None:
    scope.check_write_name(name)


def test_write_size_and_nul() -> None:
    scope.check_write_size(b"x" * scope.MAX_WRITE_BYTES)
    with pytest.raises(ScopeError):
        scope.check_write_size(b"x" * (scope.MAX_WRITE_BYTES + 1))
    with pytest.raises(ScopeError):
        scope.check_write_size(b"a\0b")


def test_text_gate_binary_utf8_and_size() -> None:
    assert scope.check_text_bytes("привет".encode(), "x.md") == "привет"
    with pytest.raises(ScopeError, match="Бинарный"):
        scope.check_text_bytes(b"abc\0def", "x.bin")
    with pytest.raises(ScopeError, match="UTF-8"):
        scope.check_text_bytes(b"\xff\xfe", "x.md")
    with pytest.raises(ScopeError, match="1 MiB"):
        scope.check_text_bytes(b"x" * (scope.MAX_READ_BYTES + 1), "x.txt")
    scope.check_text_bytes(b"x" * scope.MAX_READ_BYTES, "x.txt")


def test_tracked_snapshot_modes_and_casefold(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "repo")
    tracked = scope.load_tracked(root)
    assert tracked.lookup("DOCS/A.MD") == "docs/a.md"
    assert tracked.lookup("nope.md") is None
    assert tracked.is_symlink("Linked.md")
    assert "linked.md" not in {p.casefold() for p in tracked.paths()}
    assert "docs" in tracked.parent_dirs and "" in tracked.parent_dirs
    assert "newdir" not in tracked.parent_dirs


def test_load_tracked_on_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScopeError):
        scope.load_tracked(tmp_path / "missing")


def test_inside_root_rejects_symlink_parent_and_leaf(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("s", encoding="utf-8")
    assert scope.assert_inside_root(root, "docs/a.md") == root / "docs" / "a.md"
    if not make_dir_link(root / "docs" / "lnk", outside):
        pytest.skip("OS refuses to create symlinks/junctions")
    with pytest.raises(ScopeError, match="ссылк"):
        scope.assert_inside_root(root, "docs/lnk/secret.md")
    with pytest.raises(ScopeError, match="ссылк"):
        scope.assert_inside_root(root, "docs/lnk")
    assert scope.is_link(root / "docs" / "lnk")
    assert not scope.is_link(root / "docs")
