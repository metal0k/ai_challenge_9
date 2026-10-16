"""Day change set: commit range `base..HEAD` in a real temp repo; untracked files never count."""

import subprocess

import pytest

from adventship import gitutil
from adventship.ctx import ShipContext, ShipError
from adventship.state import ShipState


def sh(repo, *argv):
    return subprocess.run(
        ["git", *argv], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def commit(repo, name, subject, text="x\n"):
    (repo / name).write_text(text, encoding="utf-8")
    sh(repo, "add", name)
    sh(repo, "commit", "-q", "-m", subject)


def make_ctx(repo, tag="w07d35"):
    def run(argv, *, timeout, input=None, env=None):
        return subprocess.run(
            argv, cwd=repo, capture_output=True, text=True, encoding="utf-8", input=input
        )

    workdir = repo.parent / "ship"
    return ShipContext(
        week=7,
        day=35,
        tag=tag,
        root=repo,
        workdir=workdir,
        env={},
        state=ShipState.load(workdir / "state.json"),
        run=run,
        complete=lambda *a, **k: None,
        gate=lambda q, d: True,
        driver_factory=lambda: None,
        now=lambda: "2026-10-16T10:00:00+03:00",
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "t@example.com")
    sh(root, "config", "user.name", "t")
    commit(root, "base.py", "init")
    sh(root, "tag", "w07d33")
    commit(root, "d34.py", "feat(w07d34): day 34")
    sh(root, "tag", "w07d34")
    sh(root, "tag", "release-1")
    sh(root, "tag", "w7d36")
    commit(root, "d35.py", "feat(w07d35): day 35 work")
    return root


def test_day_range_picks_w07d34_over_w07d33_and_ignores_other_tags(repo):
    assert gitutil.day_range(make_ctx(repo), "w07d35") == ("w07d34", "w07d34..HEAD")


def test_day_range_compares_week_and_day_numerically(repo):
    sh(repo, "tag", "w06d99")
    sh(repo, "tag", "w07d36")  # higher than the current tag: ignored
    assert gitutil.day_range(make_ctx(repo))[0] == "w07d34"


def test_day_range_falls_back_to_the_root_commit_without_tags(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "t@example.com")
    sh(root, "config", "user.name", "t")
    commit(root, "a.py", "first")
    first = sh(root, "rev-parse", "HEAD")
    commit(root, "b.py", "second")
    base, rng = gitutil.day_range(make_ctx(root))
    assert (base, rng) == (first, f"{first}..HEAD")
    assert gitutil.day_changes(make_ctx(root)).commits == ["second"]


def test_day_changes_lists_range_commits_files_and_stat(repo):
    changes = gitutil.day_changes(make_ctx(repo))
    assert changes.commits == ["feat(w07d35): day 35 work"]
    assert changes.files == ["d35.py"]
    assert "d35.py" in changes.stat and "1 file changed" in changes.stat
    assert changes.dirty_tracked == []


def test_untracked_file_is_absent_from_the_change_set(repo):
    (repo / "AGENTS.md").write_text("owner\n", encoding="utf-8")
    changes = gitutil.day_changes(make_ctx(repo))
    assert changes.files == ["d35.py"]
    assert "AGENTS.md" not in changes.stat
    assert changes.dirty_tracked == []


def test_dirty_tracked_file_is_reported_but_not_in_files(repo):
    (repo / "d34.py").write_text("changed\n", encoding="utf-8")
    changes = gitutil.day_changes(make_ctx(repo))
    assert changes.dirty_tracked == ["d34.py"]
    assert changes.files == ["d35.py"]


def test_empty_repository_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    with pytest.raises(ShipError):
        gitutil.day_range(make_ctx(root))
