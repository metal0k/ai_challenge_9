"""week_05/corpus.py — corpus collection from a real (isolated) git repo (SPEC-w05d21.md §1, §9).

Uses a throwaway repo built in tmp_path, never this project's own history —
same isolation approach as tests/test_check_staged.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from advent_core.errors import AdventError
from week_05.corpus import collect_corpus


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    # Disable the clean/smudge CRLF filter so a file staged with literal CRLF
    # keeps it in the blob — otherwise `git add` would normalize it for us
    # and the test would not exercise collect_corpus's own conversion.
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    return repo


def _write(repo: Path, rel: str, content: str, *, newline: str = "\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline=newline)


def _commit(repo: Path, message: str = "commit") -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_collect_corpus_applies_inclusion_rules_and_sorts_by_source(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Hi\n")
    _write(repo, ".agents/notes.md", "secret course chat notes")
    _write(repo, "sub/dir/notes.md", "nested md is included")
    _write(repo, "advent_core/config.py", "# config")
    _write(repo, "advent_core/sub/deep.py", "# not top-level, excluded")
    _write(repo, "week_01/cli.py", "# outside advent_core, excluded")
    _write(repo, "advent_core/notes.txt", "not .md or .py, excluded")
    head_sha = _commit(repo)

    sha, docs = collect_corpus(repo)

    assert sha == head_sha
    assert len(sha) == 40
    assert [d.source for d in docs] == sorted(
        ["README.md", "sub/dir/notes.md", "advent_core/config.py"]
    )


def test_collect_corpus_normalizes_crlf_to_lf(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "line one\r\nline two\r\n", newline="")
    _commit(repo)

    _, docs = collect_corpus(repo)

    assert docs[0].text == "line one\nline two\n"
    assert "\r" not in docs[0].text


def test_collect_corpus_reads_from_the_commit_not_the_working_tree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "committed content\n")
    _commit(repo)
    # Dirty the working tree after the commit — collect_corpus must ignore this.
    _write(repo, "README.md", "UNCOMMITTED EDIT\n")

    _, docs = collect_corpus(repo)

    assert docs[0].text == "committed content\n"


def test_collect_corpus_paths_are_posix(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "sub/dir/notes.md", "x\n")
    _commit(repo)

    _, docs = collect_corpus(repo)

    assert docs[0].source == "sub/dir/notes.md"
    assert "\\" not in docs[0].source


def test_collect_corpus_rev_pinned_to_an_explicit_sha_matches_default(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Hi\n")
    head_sha = _commit(repo)

    sha_default, docs_default = collect_corpus(repo)
    sha_explicit, docs_explicit = collect_corpus(repo, rev=head_sha)

    assert sha_default == sha_explicit == head_sha
    assert [d.source for d in docs_default] == [d.source for d in docs_explicit]
    assert [d.text for d in docs_default] == [d.text for d in docs_explicit]


def test_collect_corpus_unknown_rev_raises_advent_error(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Hi\n")
    _commit(repo)

    with pytest.raises(AdventError):
        collect_corpus(repo, rev="not-a-real-revision")


def test_collect_corpus_not_a_git_repo_raises_advent_error(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()

    with pytest.raises(AdventError):
        collect_corpus(not_a_repo)


def test_collect_corpus_enumerates_files_from_the_pinned_rev_not_the_index(tmp_path: Path) -> None:
    """A file added or deleted AFTER the target rev must not affect it (finding 1)."""
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Hi\n")
    _write(repo, "gone.md", "will be deleted after the pin\n")
    old_sha = _commit(repo, "old")

    (repo / "gone.md").unlink()
    _write(repo, "added.md", "added after the pin\n")
    _commit(repo, "new")

    _, docs = collect_corpus(repo, rev=old_sha)

    assert [d.source for d in docs] == ["README.md", "gone.md"]


def test_collect_corpus_empty_repo_returns_no_documents(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, ".gitkeep", "")
    _commit(repo)

    sha, docs = collect_corpus(repo)

    assert len(sha) == 40
    assert docs == []
