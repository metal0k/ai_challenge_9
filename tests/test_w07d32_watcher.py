"""Week 07 day 32: local PR watcher (SPEC-w07d32-watcher.md). Fake client, git, build, review.

No network, no real env or journal: every seam is injected, state lives under tmp_path.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from advent_core.config import ConfigError
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult
from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    ServerError,
)
from adventassistant import cli, review, review_index, watcher
from adventassistant.github import PullRequest
from adventassistant.watcher import (
    REVIEW_MARKER,
    FileLock,
    WatcherFatal,
    basic_header,
    parse_github_url,
    revision_marker,
    run_watch,
)
from week_05.chunking import Chunk
from week_05.index import ENDPOINT_CLOUD, write_index

BASE, HEAD = "a" * 40, "b" * 40
BASE2, HEAD2 = "d" * 40, "c" * 40
REPO = "o/r"
TOKEN = "synthetic-token-123"
COMPLETE = object()


class _Stop(Exception):
    pass


def _pull(number=1, base=BASE, head=HEAD, state="open", head_repo=REPO) -> PullRequest:
    return PullRequest(
        number=number,
        state=state,
        base_sha=base,
        head_sha=head,
        base_repo=REPO,
        head_repo=head_repo,
    )


def _err(cls, text: str, retry_after: float | None = None):
    exc = cls(text)
    if retry_after is not None:
        exc.retry_after = retry_after
    return exc


def _flat(text: str) -> str:
    return " ".join(text.split())


def _write_index(path: Path, corpus_rev: str) -> None:
    chunk = Chunk(
        chunk_id="review:a.md#0",
        strategy="review",
        source="a.md",
        title="t",
        section="s",
        ordinal=0,
        char_start=0,
        char_end=1,
        line_start=1,
        line_end=1,
        text="x",
    )
    vectors = np.ones((1, 4), dtype=np.float32)
    embedded = EmbedResult(
        vectors=vectors, model=DEFAULT_EMBED_MODEL, prompt_tokens=1, requests=1, latency_ms=1
    )
    write_index(
        path,
        {"review": ([chunk], embedded)},
        corpus_rev=corpus_rev,
        corpus_files=1,
        corpus_chars=1,
        endpoint=ENDPOINT_CLOUD,
        meta={"purpose": "w07d32"},
    )


class FakeClient:
    def __init__(self, pulls=()) -> None:
        self.pulls = {p.number: p for p in pulls}
        self.comments: list[tuple[int, str, int]] = []
        self.me = 1
        self.fail_post: list[Exception] = []
        self.lose_response = 0
        self.list_errors: list[Exception] = []
        self.get_override: dict[int, PullRequest] = {}
        self.find_error: Exception | None = None
        self.list_calls = 0
        self.closed = False

    @property
    def posts(self):
        return [c for c in self.comments if c[2] == self.me]

    def list_open_pulls(self, pr=None):
        self.list_calls += 1
        if self.list_errors:
            raise self.list_errors.pop(0)
        return [p for p in self.pulls.values() if pr in (None, p.number)]

    def get_pull(self, number):
        return self.get_override.get(number, self.pulls[number])

    def find_comment(self, number, marker):
        if self.find_error is not None:
            raise self.find_error
        for index, (n, body, user) in enumerate(self.comments):
            if n == number and marker in body and user == self.me:
                return f"https://github.com/o/r/issues/{n}#c{index}"
        return None

    def post_comment(self, number, body):
        if self.fail_post:
            raise self.fail_post.pop(0)
        self.comments.append((number, body, self.me))
        if self.lose_response:
            self.lose_response -= 1
            raise NetworkError("ответ потерян")
        return f"https://github.com/o/r/issues/{number}#c{len(self.comments) - 1}"

    def close(self):
        self.closed = True


class FakeGit:
    def __init__(self) -> None:
        self.origin = "https://github.com/o/r.git"
        self.have = False
        self.empty = False
        self.fetch_rc = 0
        self.fetch_err = ""
        self.calls: list[tuple[list[str], dict, float]] = []

    def __call__(self, args, *, cwd, env, timeout):
        args = list(args)
        self.calls.append((args, dict(env), timeout))
        out, rc, err = "", 0, ""
        if args[0] == "remote":
            out = self.origin + "\n"
        elif args[0] == "cat-file":
            rc = 0 if self.have else 1
        elif args[0] == "fetch":
            rc, err = self.fetch_rc, self.fetch_err
            if rc == 0:
                self.have = True
        elif args[0] == "diff":
            out = "" if self.empty else "M\0a.py\0"
        return subprocess.CompletedProcess(args, rc, out, err)

    def commands(self, name):
        return [c for c in self.calls if c[0][0] == name]


class FakeBuild:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, root, *, rev, db_path):
        self.calls.append((rev, Path(db_path)))
        _write_index(Path(db_path), rev)
        return "индекс"


class FakeReview:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, root, *, base, head, db_path, complete, pr=None):
        self.calls.append(
            {"base": base, "head": head, "db": db_path, "complete": complete, "pr": pr}
        )
        text = f"Потенциальные баги\n- ревью {len(self.calls)}"
        print(text)
        return text


class Env:
    def __init__(self, tmp_path: Path, pulls) -> None:
        self.root = tmp_path / "root"
        self.root.mkdir()
        (self.root / ".env").write_text(
            f"GITHUB_URL=https://github.com/o/r.git\nGITHUB_TOKEN={TOKEN}\n", encoding="utf-8"
        )
        self.state = tmp_path / "data" / "watcher.json"
        self.reports = self.state.parent / "reports"
        self.cache = tmp_path / "data" / "indexes"
        self.client = FakeClient(pulls)
        self.git = FakeGit()
        self.build = FakeBuild()
        self.review = FakeReview()
        self.sleeps: list[float] = []

    def kwargs(self) -> dict:
        return {
            "complete": COMPLETE,
            "once": True,
            "state_path": self.state,
            "cache_dir": self.cache,
            "client": self.client,
            "build_index": self.build,
            "review_runner": self.review,
            "git_runner": self.git,
            "sleep": self.sleeps.append,
        }

    def run(self, **over) -> int:
        return run_watch(self.root, **{**self.kwargs(), **over})

    def status(self, number=1, base=BASE, head=HEAD):
        entries = json.loads(self.state.read_text(encoding="utf-8"))["entries"]
        entry = entries.get(f"{number}:{base}:{head}")
        return None if entry is None else entry["status"]

    def result_path(self, number=1, base=BASE, head=HEAD) -> Path:
        return self.reports / f"pr-{number}-{base}-{head}.json"


@pytest.fixture
def env(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real seam used in a test")

    no_client = type("NoClient", (), {"from_project": staticmethod(boom)})
    monkeypatch.setattr(watcher, "GithubClient", no_client)
    monkeypatch.setattr(watcher, "run_git", boom)
    monkeypatch.setattr(review, "run_review", boom)
    monkeypatch.setattr(review_index, "build_review_index", boom)
    return Env(tmp_path, [_pull()])


# --- review, report, comment, state --------------------------------------------------------


def test_first_review_reports_comments_and_checkpoints(env, capsys):
    assert env.run() == 0
    assert len(env.build.calls) == 1
    rev, db = env.build.calls[0]
    assert rev == BASE and db.parent == env.cache
    assert len(env.review.calls) == 1
    call = env.review.calls[0]
    assert call["pr"] is None and call["complete"] is COMPLETE and call["base"] == BASE
    assert len(env.client.posts) == 1
    body = env.client.posts[0][1]
    assert REVIEW_MARKER in body and revision_marker(BASE, HEAD) in body
    assert f"{BASE[:12]}..{HEAD[:12]}" in body and "Потенциальные баги" in body
    assert env.status() == "posted"
    assert (env.reports / f"pr-1-{BASE}-{HEAD}.md").exists() and env.result_path().exists()
    out, err = capsys.readouterr()
    assert out.count("PR #1") == 1 and out.count("Потенциальные баги") == 1
    assert "проход" in err and "Потенциальные баги" not in err


def test_same_revision_again_calls_nothing(env, capsys):
    env.run()
    capsys.readouterr()
    assert env.run() == 0
    assert len(env.build.calls) == len(env.review.calls) == len(env.client.posts) == 1
    assert capsys.readouterr().out == ""


def test_new_head_and_new_base_with_same_head_are_reviewed(env):
    env.run()
    env.client.pulls[1] = _pull(head=HEAD2)
    env.run()
    assert (len(env.review.calls), len(env.build.calls), len(env.client.posts)) == (2, 1, 2)
    env.client.pulls[1] = _pull(base=BASE2, head=HEAD2)
    env.run()
    assert (len(env.review.calls), len(env.build.calls), len(env.client.posts)) == (3, 2, 3)
    assert env.status(base=BASE2, head=HEAD2) == "posted"


def test_no_post_then_post_calls_the_model_once(env, capsys):
    assert env.run(post=False) == 0
    assert env.client.posts == [] and env.status() == "reviewed"
    first = capsys.readouterr().out
    assert first.count("Потенциальные баги") == 1
    assert env.run(post=True) == 0
    assert len(env.review.calls) == 1 and len(env.client.posts) == 1
    assert env.status() == "posted"
    assert capsys.readouterr().out == ""


def test_post_failure_keeps_the_paid_review(env):
    env.client.fail_post = [_err(ServerError, "500")]
    assert env.run() == 1
    assert env.status() == "reviewed" and env.result_path().exists()
    assert env.run() == 0
    assert len(env.review.calls) == 1 and len(env.client.posts) == 1
    assert env.status() == "posted"


def test_lost_post_response_does_not_duplicate_the_comment(env):
    env.client.lose_response = 1
    assert env.run() == 1
    assert len(env.client.comments) == 1 and env.status() == "reviewed"
    assert env.run() == 0
    assert len(env.client.comments) == 1 and env.status() == "posted"


def test_foreign_author_marker_is_not_completion(env):
    env.client.comments.append((1, f"{REVIEW_MARKER}\n{revision_marker(BASE, HEAD)}\n", 99))
    assert env.run() == 0
    assert len(env.client.posts) == 1 and env.status() == "posted"


def test_find_comment_failure_forbids_the_post(env):
    env.client.find_error = _err(NetworkError, "сеть")
    assert env.run() == 1
    assert env.client.comments == [] and env.status() == "reviewed"


def test_checkpoint_failure_recovers_from_result_json(env, monkeypatch, capsys):
    real_set = watcher.State.set

    def broken(self, key, status):
        if status == "reviewed":
            raise WatcherFatal("диск")
        real_set(self, key, status)

    monkeypatch.setattr(watcher.State, "set", broken)
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.result_path().exists() and len(env.review.calls) == 1
    monkeypatch.setattr(watcher.State, "set", real_set)
    capsys.readouterr()
    assert env.run() == 0
    assert len(env.review.calls) == 1 and len(env.client.posts) == 1
    out = capsys.readouterr().out
    assert out.count("PR #1") == 1 and "ревью 1" in out


def test_recovered_result_with_existing_marker_posts_nothing(env):
    env.client.comments.append((1, f"x\n{revision_marker(BASE, HEAD)}\n", 1))
    env.reports.mkdir(parents=True)
    watcher.Store(env.reports, REPO).save_result(
        watcher.Key(1, BASE, HEAD), watcher.Result(False, "Потенциальные баги\n- старое")
    )
    assert env.run() == 0
    assert env.review.calls == [] and len(env.client.comments) == 1
    assert env.status() == "posted"


def _reviewed_state(env) -> None:
    env.state.parent.mkdir(parents=True)
    env.state.write_text(
        json.dumps(
            {
                "version": 1,
                "repo": REPO,
                "entries": {f"1:{BASE}:{HEAD}": {"status": "reviewed", "updated": "t"}},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("post", [True, False])
def test_missing_result_for_reviewed_is_fatal_not_a_second_review(env, post):
    _reviewed_state(env)
    with pytest.raises(WatcherFatal, match="Модель не вызвана"):
        env.run(post=post)
    assert env.review.calls == [] and env.build.calls == [] and env.client.comments == []


@pytest.mark.parametrize("post", [True, False])
def test_corrupt_result_for_reviewed_is_fatal_in_both_modes(env, post):
    _reviewed_state(env)
    env.reports.mkdir(parents=True)
    env.result_path().write_text("{oops", encoding="utf-8")
    with pytest.raises(WatcherFatal, match="повреждён"):
        env.run(post=post)
    assert env.review.calls == [] and env.build.calls == [] and env.client.comments == []
    assert env.result_path().read_text(encoding="utf-8") == "{oops"


@pytest.mark.parametrize("variant", ["closed", "fork", "moved"])
def test_stale_pr_is_not_commented(env, variant):
    changed = {
        "closed": _pull(state="closed"),
        "fork": _pull(head_repo="x/y"),
        "moved": _pull(head=HEAD2),
    }[variant]
    env.client.get_override[1] = changed
    assert env.run() == 0
    assert env.client.comments == [] and env.status() == "reviewed"
    assert env.result_path().exists()


def test_forks_and_closed_prs_are_skipped(env):
    env.client.pulls = {
        1: _pull(1, head_repo="fork/r"),
        2: _pull(2, state="closed"),
        3: _pull(3, head=HEAD2),
    }
    assert env.run() == 0
    assert [c["head"] for c in env.review.calls] == [HEAD2]
    assert env.status(3, head=HEAD2) == "posted"


def test_pr_filter_limits_the_pass(env):
    env.client.pulls = {1: _pull(1), 2: _pull(2, head=HEAD2)}
    assert env.run(pr=2) == 0
    assert [c["head"] for c in env.review.calls] == [HEAD2]


def test_empty_diff_is_done_in_both_modes(env, capsys):
    env.git.empty = True
    for post in (True, False):
        assert env.run(post=post) == 0
    assert env.build.calls == [] and env.review.calls == [] and env.client.comments == []
    assert env.status() == "empty"
    assert json.loads(env.result_path().read_text(encoding="utf-8"))["empty"] is True
    assert capsys.readouterr().out == ""


# --- index cache ---------------------------------------------------------------------------


def test_index_is_reused_for_the_same_base(env):
    env.client.pulls = {1: _pull(1), 2: _pull(2, head=HEAD2)}
    assert env.run() == 0
    assert len(env.build.calls) == 1 and len(env.review.calls) == 2
    for _rev, db in env.build.calls:
        assert db != review_index.REVIEW_INDEX_PATH and db.parent == env.cache


@pytest.mark.parametrize("once", [True, False])
def test_cached_index_with_wrong_corpus_rev_is_fatal(env, once):
    env.cache.mkdir(parents=True)
    name = f"review-{BASE}-v{review_index.REVIEW_INDEX_VERSION}.sqlite3"
    _write_index(env.cache / name, "e" * 40)
    with pytest.raises(WatcherFatal, match="corpus_rev"):
        env.run(once=once)
    assert env.review.calls == [] and env.build.calls == []
    assert env.sleeps == []


def test_cache_lock_busy_fails_the_pr_not_the_process(env, monkeypatch):
    monkeypatch.setattr(watcher, "CACHE_LOCK_WAIT", 0.0)
    name = f"review-{BASE}-v{review_index.REVIEW_INDEX_VERSION}.lock"
    with FileLock(env.cache / name):
        assert env.run() == 1
    assert env.build.calls == []


# --- errors and the loop -------------------------------------------------------------------


def test_auth_error_stops_and_releases_the_lock(env):
    env.client.list_errors = [_err(AuthError, "401")]
    with pytest.raises(AuthError):
        env.run()
    with FileLock(env.state.with_name("watcher.json.lock")):
        pass


def test_auth_error_on_post_stops_but_keeps_the_review(env):
    env.client.fail_post = [_err(AuthError, "403")]
    with pytest.raises(AuthError):
        env.run()
    assert env.status() == "reviewed" and env.result_path().exists()


def test_transient_error_survives_the_continuous_loop(env):
    env.client.list_errors = [_err(ServerError, "502", retry_after=90)]

    def sleep(seconds):
        env.sleeps.append(seconds)
        if len(env.sleeps) == 2:
            raise _Stop

    with pytest.raises(_Stop):
        env.run(once=False, sleep=sleep)
    assert env.sleeps == [90.0, 60.0]
    assert env.status() == "posted"
    with FileLock(env.state.with_name("watcher.json.lock")):
        pass


def test_interval_is_the_sleep_between_passes(env):
    def sleep(seconds):
        env.sleeps.append(seconds)
        raise _Stop

    with pytest.raises(_Stop):
        env.run(once=False, interval=10, sleep=sleep)
    assert env.sleeps == [10.0]


@pytest.mark.parametrize("error", [ConfigurationError("конфиг"), ConfigError("конфиг")])
@pytest.mark.parametrize("once", [True, False])
def test_config_errors_from_a_callback_stop_the_watcher(env, error, once):
    def broken(*args, **kwargs):
        raise error

    with pytest.raises(type(error)):
        env.run(once=once, review_runner=broken)
    assert env.sleeps == [] and not env.state.exists()


def test_transient_review_error_is_still_retried(env):
    failures = [_err(ServerError, "502")]

    def flaky(*args, **kwargs):
        if failures:
            raise failures.pop(0)
        return env.review(*args, **kwargs)

    def sleep(seconds):
        env.sleeps.append(seconds)
        if len(env.sleeps) == 2:
            raise _Stop

    with pytest.raises(_Stop):
        env.run(once=False, review_runner=flaky, sleep=sleep)
    assert env.sleeps == [60.0, 60.0]
    assert len(env.review.calls) == 1 and env.status() == "posted"


def test_generic_builder_error_is_fatal_in_the_continuous_loop(env):
    calls = []

    def broken(root, *, rev, db_path):
        calls.append(rev)
        raise AdventError("Не удалось записать индекс: disk I/O error", hint="Проверь диск.")

    def sleep(seconds):
        env.sleeps.append(seconds)
        raise _Stop

    with pytest.raises(WatcherFatal, match="disk I/O error") as info:
        env.run(once=False, build_index=broken, sleep=sleep)
    assert info.value.hint == "Проверь диск."
    assert len(calls) == 1 and env.sleeps == [] and env.review.calls == []
    assert not env.state.exists()


def test_transient_builder_error_is_still_retried(env):
    failures = [_err(ServerError, "502")]

    def flaky(root, *, rev, db_path):
        if failures:
            raise failures.pop(0)
        return env.build(root, rev=rev, db_path=db_path)

    def sleep(seconds):
        env.sleeps.append(seconds)
        if len(env.sleeps) == 2:
            raise _Stop

    with pytest.raises(_Stop):
        env.run(once=False, build_index=flaky, sleep=sleep)
    assert env.sleeps == [60.0, 60.0]
    assert len(env.build.calls) == 1 and env.status() == "posted"


@pytest.mark.parametrize("interval", [4, 0, -1])
def test_interval_below_five_is_refused(env, interval):
    with pytest.raises(ConfigurationError):
        env.run(interval=interval)
    assert env.client.list_calls == 0


def test_pr_number_must_be_positive(env):
    with pytest.raises(ConfigurationError):
        env.run(pr=0)


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2, "repo": REPO, "entries": {}}),
        json.dumps({"version": True, "repo": REPO, "entries": {}}),
        json.dumps({"version": 1, "repo": "x/y", "entries": {}}),
        json.dumps({"version": 1, "repo": REPO, "entries": {}, "extra": 1}),
        json.dumps({"version": 1, "repo": REPO, "entries": {"1:abc:def": {}}}),
        json.dumps(
            {
                "version": 1,
                "repo": REPO,
                "entries": {f"1:{BASE}:{HEAD}": {"status": "reviewed"}},
            }
        ),
        json.dumps(
            {
                "version": 1,
                "repo": REPO,
                "entries": {f"1:{BASE}:{HEAD}": {"status": "wat", "updated": "t"}},
            }
        ),
    ],
)
def test_invalid_state_is_refused_and_left_alone(env, content):
    env.state.parent.mkdir(parents=True)
    env.state.write_text(content, encoding="utf-8")
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.state.read_text(encoding="utf-8") == content
    assert env.client.list_calls == 0


def test_permission_error_on_state_read_is_not_an_empty_state(env, monkeypatch):
    env.state.parent.mkdir(parents=True)
    env.state.write_text(json.dumps({"version": 1, "repo": REPO, "entries": {}}), encoding="utf-8")
    before = env.state.read_text(encoding="utf-8")
    real = Path.read_text

    def read_text(self, *a, **k):
        if self.name == "watcher.json":
            raise PermissionError("busy")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(watcher.time, "sleep", lambda s: None)
    with pytest.raises(WatcherFatal):
        env.run()
    monkeypatch.setattr(Path, "read_text", real)
    assert env.state.read_text(encoding="utf-8") == before


def test_permission_error_on_state_replace_keeps_the_old_file(env, monkeypatch):
    env.state.parent.mkdir(parents=True)
    env.state.write_text(json.dumps({"version": 1, "repo": REPO, "entries": {}}), encoding="utf-8")
    before = env.state.read_text(encoding="utf-8")
    real = watcher.os.replace

    def replace(src, dst):
        if Path(dst).name == "watcher.json":
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(watcher.os, "replace", replace)
    monkeypatch.setattr(watcher.time, "sleep", lambda s: None)
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.state.read_text(encoding="utf-8") == before
    assert not list(env.state.parent.glob("*.tmp"))


def test_tmp_cleanup_failure_does_not_mask_the_state_write_error(env, monkeypatch):
    env.state.parent.mkdir(parents=True)
    env.state.write_text(json.dumps({"version": 1, "repo": REPO, "entries": {}}), encoding="utf-8")
    before = env.state.read_text(encoding="utf-8")
    real_replace, real_unlink = watcher.os.replace, Path.unlink

    def replace(src, dst):
        if Path(dst).name == "watcher.json":
            raise PermissionError("locked")
        return real_replace(src, dst)

    def unlink(self, *a, **k):
        if self.suffix == ".tmp":
            raise PermissionError("busy")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(watcher.os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(watcher.time, "sleep", lambda s: None)
    with pytest.raises(WatcherFatal, match="Не удалось записать"):
        env.run()
    assert env.state.read_text(encoding="utf-8") == before


def test_index_part_cleanup_failure_does_not_mask_the_save_error(env, monkeypatch):
    real_replace, real_unlink = watcher.os.replace, Path.unlink

    def replace(src, dst):
        if Path(src).name.endswith(".part.sqlite3"):
            raise PermissionError("locked")
        return real_replace(src, dst)

    def unlink(self, *a, **k):
        if self.name.endswith(".part.sqlite3") and self.exists():
            raise PermissionError("busy")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(watcher.os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(WatcherFatal, match="Не удалось сохранить индекс"):
        env.run()
    assert env.review.calls == []


def test_stale_index_part_that_cannot_be_removed_is_fatal(env, monkeypatch):
    env.cache.mkdir(parents=True)
    stale = env.cache / f"review-{BASE}-v{review_index.REVIEW_INDEX_VERSION}.part.sqlite3"
    stale.write_bytes(b"x")
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self == stale:
            raise PermissionError("busy")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(WatcherFatal, match="Не удалось удалить"):
        env.run()
    assert env.build.calls == [] and env.review.calls == []


@pytest.mark.parametrize("field", ["version", "number"])
def test_result_json_with_bool_in_an_int_field_is_refused(env, field):
    env.reports.mkdir(parents=True)
    watcher.Store(env.reports, REPO).save_result(
        watcher.Key(1, BASE, HEAD), watcher.Result(False, "Потенциальные баги\n- старое")
    )
    doc = json.loads(env.result_path().read_text(encoding="utf-8"))
    doc[field] = True
    env.result_path().write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.review.calls == [] and env.client.comments == []


def test_corrupt_result_json_is_refused(env):
    env.reports.mkdir(parents=True)
    env.result_path().write_text("{oops", encoding="utf-8")
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.review.calls == []
    assert env.result_path().read_text(encoding="utf-8") == "{oops"


def test_malformed_pull_is_fatal(env):
    env.client.pulls = {1: _pull(head="XYZ")}
    with pytest.raises(WatcherFatal):
        env.run()
    env.client.pulls = {0: _pull(number=0)}
    with pytest.raises(WatcherFatal):
        env.run()


# --- git -----------------------------------------------------------------------------------


def test_fetch_header_is_ephemeral_and_prompts_are_off(env):
    assert env.run() == 0
    ((args, genv, timeout),) = env.git.commands("fetch")
    assert args == [
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-recurse-submodules",
        "origin",
        BASE,
        HEAD,
    ]
    assert all(TOKEN not in a and "AUTHORIZATION" not in a for a in args)
    assert timeout == 60
    assert genv["GIT_TERMINAL_PROMPT"] == "0" and genv["GCM_INTERACTIVE"] == "Never"
    assert genv["ALL_PROXY"] == ""
    pairs = {
        genv[f"GIT_CONFIG_KEY_{i}"]: genv[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(genv["GIT_CONFIG_COUNT"]))
    }
    assert pairs["credential.helper"] == ""
    assert pairs["http.https://github.com/.extraheader"] == basic_header(TOKEN)
    for args, local_env, _ in env.git.calls:
        if args[0] != "fetch":
            assert basic_header(TOKEN) not in local_env.values()


def test_git_touches_only_remote_cat_file_fetch_diff(env):
    env.run()
    assert {c[0][0] for c in env.git.calls} <= {"remote", "cat-file", "fetch", "diff"}


def test_present_objects_skip_the_fetch(env):
    env.git.have = True
    assert env.run() == 0
    assert env.git.commands("fetch") == [] and env.git.commands("remote") == []


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/o/r",
        "https://github.com/o/r.git",
        "https://github.com/o/r/",
        "HTTPS://GitHub.com/o/r.git",
    ],
)
def test_github_url_accepts_plain_https(url):
    assert parse_github_url(url) == "o/r"


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/o/r.git",
        "https://user@github.com/o/r.git",
        "https://user:pw@github.com/o/r.git",
        "https://github.com:444/o/r.git",
        "https://evil.example/o/r.git",
        "https://github.com/o",
        "https://github.com/o/r/extra",
        "https://github.com/o/r.git?x=1",
        "git@github.com:o/r.git",
        "",
    ],
)
def test_github_url_rejects_everything_else(url):
    with pytest.raises(ConfigurationError):
        parse_github_url(url)


@pytest.mark.parametrize(
    "origin", ["https://github.com/other/repo.git", "https://tok@github.com/o/r.git", "ssh://x/y"]
)
def test_origin_must_match_the_configured_repo(env, origin):
    env.git.origin = origin
    with pytest.raises(WatcherFatal):
        env.run()
    assert env.git.commands("fetch") == []


def test_fetch_auth_failure_stops_without_leaking_the_header(env, capsys):
    env.git.fetch_rc = 128
    env.git.fetch_err = (
        f"fatal: Authentication failed for 'https://github.com/' {TOKEN} {basic_header(TOKEN)}"
    )
    with pytest.raises(AuthError) as caught:
        env.run()
    shown = caught.value.message + (caught.value.hint or "")
    out, err = capsys.readouterr()
    for secret in (TOKEN, basic_header(TOKEN), basic_header(TOKEN).split()[-1]):
        assert secret not in shown + out + err
    assert len(env.git.commands("fetch")) == 1


def test_fetch_network_failure_is_transient_and_has_no_fallback(env, capsys):
    env.git.fetch_rc = 1
    env.git.fetch_err = f"fatal: unable to access: Could not resolve host {basic_header(TOKEN)}"
    assert env.run() == 1
    assert len(env.git.commands("fetch")) == 1
    out, err = capsys.readouterr()
    assert basic_header(TOKEN).split()[-1] not in out + err
    assert env.review.calls == [] and not env.state.exists()


def test_default_git_runner_passes_env_and_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(watcher.subprocess, "run", fake_run)
    watcher.run_git(["fetch"], cwd=Path("."), env={"A": "1"}, timeout=60)
    assert seen["timeout"] == 60 and seen["env"] == {"A": "1"}
    assert seen["cmd"] == ["git", "-c", "credential.helper=", "fetch"]
    assert seen["stdin"] == subprocess.DEVNULL

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(watcher.subprocess, "run", hang)
    with pytest.raises(NetworkError):
        watcher.run_git(["fetch"], cwd=Path("."), env={}, timeout=60)


def test_real_git_argv_drops_the_helper_and_keeps_the_header_in_env(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(watcher.subprocess, "run", fake_run)
    watcher.run_git(["fetch", "origin"], cwd=Path("."), env=watcher.git_env(TOKEN), timeout=60)
    cmd = seen["cmd"]
    assert cmd[:3] == ["git", "-c", "credential.helper="]
    for part in cmd:
        assert TOKEN not in part and "AUTHORIZATION" not in part and "extraheader" not in part
    assert basic_header(TOKEN) in seen["env"].values()


# --- locks, defaults, secrets --------------------------------------------------------------


def test_file_lock_is_exclusive_and_released_on_exit(tmp_path):
    path = tmp_path / "x.lock"
    with FileLock(path), pytest.raises(AdventError), FileLock(path):
        pass
    with FileLock(path):
        pass


def test_second_watcher_on_the_same_state_is_refused(env):
    with (
        FileLock(env.state.with_name("watcher.json.lock")),
        pytest.raises(AdventError, match="другой watcher"),
    ):
        env.run()
    assert env.client.list_calls == 0


def test_defaults_are_none_in_the_signature():
    params = inspect.signature(run_watch).parameters
    for name in ("client", "build_index", "review_runner", "git_runner", "sleep", "state_path"):
        assert params[name].default is None
    assert params["cache_dir"].default is None


def test_defaults_resolve_at_call_time(env, monkeypatch):
    monkeypatch.setattr(review, "run_review", env.review)
    monkeypatch.setattr(review_index, "build_review_index", env.build)
    monkeypatch.setattr(watcher, "run_git", env.git)

    def sleep(seconds):
        raise _Stop

    monkeypatch.setattr(watcher.time, "sleep", sleep)
    kwargs = {
        k: v
        for k, v in env.kwargs().items()
        if k not in ("build_index", "review_runner", "git_runner", "sleep")
    }
    with pytest.raises(_Stop):
        run_watch(env.root, **{**kwargs, "once": False})
    assert len(env.review.calls) == 1 and len(env.build.calls) == 1


def test_no_secret_reaches_state_or_reports(env):
    env.run()
    for path in env.state.parent.rglob("*"):
        if path.is_file() and path.suffix != ".sqlite3":
            text = path.read_text(encoding="utf-8", errors="replace")
            assert TOKEN not in text and basic_header(TOKEN).split()[-1] not in text


def test_missing_github_url_is_a_config_error(env):
    (env.root / ".env").write_text("GITHUB_TOKEN=x\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        env.run()


# --- CLI -----------------------------------------------------------------------------------


def _invoke(monkeypatch, *argv, result=0):
    seen = {}

    def fake(root, **kwargs):
        seen.update(kwargs, root=root)
        return result

    monkeypatch.setattr(cli, "run_watch", fake)
    monkeypatch.setattr(sys, "argv", ["adventassistant", "watch-pr", *argv])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    return seen, caught.value.code


def test_cli_defaults(monkeypatch):
    seen, code = _invoke(monkeypatch)
    assert code == 0
    assert seen["root"] == cli.PROJECT_ROOT and seen["complete"] is cli.chat_core.complete
    assert (seen["interval"], seen["once"], seen["pr"], seen["post"]) == (60, False, None, True)
    assert seen["state_path"] is None and seen["cache_dir"] is None


def test_cli_passes_every_option(monkeypatch, tmp_path):
    state, cache = tmp_path / "s.json", tmp_path / "c"
    seen, code = _invoke(
        monkeypatch,
        "--once", "--pr", "7", "--no-post", "--interval", "10",
        "--state", str(state), "--cache-dir", str(cache),
    )  # fmt: skip
    assert code == 0
    assert (seen["interval"], seen["once"], seen["pr"], seen["post"]) == (10, True, 7, False)
    assert seen["state_path"] == state and seen["cache_dir"] == cache


def test_cli_exit_code_follows_the_pass_result(monkeypatch):
    _seen, code = _invoke(monkeypatch, "--once", result=1)
    assert code == 1


def test_cli_rejects_a_short_interval_with_a_message(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["adventassistant", "watch-pr", "--interval", "4"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert "не меньше 5" in _flat(capsys.readouterr().err)
