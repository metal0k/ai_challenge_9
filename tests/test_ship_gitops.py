"""commit/tag stages: real git in a temp repo, fake push and fake GitHub API."""

import base64
import subprocess

import pytest

from advent_core.telemetry import CallResult
from adventship import gitops, llm
from adventship.ctx import (
    EXIT_AUTH,
    EXIT_CHECKS_FAIL,
    EXIT_GATE_DENIED,
    EXIT_NO_TTY,
    ShipContext,
    ShipError,
    TaskSpec,
)
from adventship.state import ShipState

TAG = "w07d35"
TOKEN = "tok-" + "k" * 24
GITHUB_URL = "https://github.com/owner/repo"
MESSAGE = "feat(w07d35): add ship pipeline"


def sh(repo, *argv):
    return subprocess.run(
        ["git", *argv], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


class Harness:
    """Real git for everything except `push` and `check_staged.py`, which are faked."""

    def __init__(self, repo):
        self.repo = repo
        self.argvs: list[list[str]] = []
        self.push_envs: list[dict] = []
        self.scan_rc = 0
        self.push_rc = 0
        self.push_stderr = ""
        self.push_raises: Exception | None = None
        self.config_extraheader = ""
        self.remote: dict[str, tuple[str, str]] = {}
        self.pushed_sha: str | None = None
        self.gates: list[tuple[str, str]] = []
        self.gate_answer = True
        self.llm_texts = [MESSAGE]

    def run(self, argv, *, timeout, input=None, env=None):
        self.argvs.append(list(argv))
        if argv[-1].endswith("check_staged.py"):
            return subprocess.CompletedProcess(argv, self.scan_rc, "", "")
        if "push" in argv:
            self.push_envs.append(dict(env or {}))
            if self.push_raises:
                raise self.push_raises
            if self.push_rc == 0:
                self.pushed_sha = sh(self.repo, "rev-parse", "HEAD")
                self.remote["heads/main"] = ("commit", self.pushed_sha)
                self.remote[f"tags/{TAG}"] = ("commit", self.pushed_sha)
            return subprocess.CompletedProcess(argv, self.push_rc, "", self.push_stderr)
        if argv[1:4] == ["config", "--get-all", "http.extraheader"]:
            rc = 0 if self.config_extraheader else 1
            return subprocess.CompletedProcess(argv, rc, self.config_extraheader, "")
        return subprocess.run(
            argv, cwd=self.repo, capture_output=True, text=True, encoding="utf-8", input=input
        )

    def gate(self, question, details):
        self.gates.append((question, details))
        return self.gate_answer

    def complete(self, config, messages, *args, **kwargs):
        text = self.llm_texts.pop(0) if len(self.llm_texts) > 1 else self.llm_texts[0]
        return CallResult(text=text, model_requested="m")

    def fetch(self, github_url, token, ref):
        assert token == TOKEN
        return self.remote.get(ref)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "config", "user.email", "t@example.com")
    sh(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("print(1)\n", encoding="utf-8")
    sh(repo, "add", ".")
    sh(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(llm, "_config", lambda max_tokens: "cfg")
    h = Harness(repo)
    monkeypatch.setattr(gitops, "fetch_remote_ref", h.fetch)
    return h


def make_ctx(h, tmp_path):
    workdir = tmp_path / "ship"
    return ShipContext(
        week=7,
        day=35,
        tag=TAG,
        root=h.repo,
        workdir=workdir,
        env={"GITHUB_TOKEN": TOKEN, "GITHUB_URL": GITHUB_URL},
        state=ShipState.load(workdir / "state.json"),
        run=h.run,
        complete=h.complete,
        gate=h.gate,
        driver_factory=lambda: None,
        now=lambda: "2026-10-16T10:00:00+03:00",
        task=TaskSpec(week=7, day=35, source="task_07.md", text="## Day 35\nСдача дня."),
        journal_path=tmp_path / "journal.jsonl",
    )


def dirty_repo(h):
    (h.repo / "a.py").write_text("print(2)\n", encoding="utf-8")
    (h.repo / "b.py").write_text("x = 1\n", encoding="utf-8")


# ------------------------------------------------------------------ commit


def test_commit_creates_one_commit_with_validated_message(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    result = gitops.run_commit(ctx)
    head = sh(harness.repo, "rev-parse", "HEAD")
    assert result.data["sha"] == head
    assert result.data["files"] == ["a.py", "b.py"]
    assert sh(harness.repo, "log", "-1", "--format=%B") == MESSAGE
    assert sh(harness.repo, "status", "--porcelain") == ""


def test_commit_message_has_no_trailers(harness, tmp_path):
    harness.llm_texts = [f"{MESSAGE}\n\nCo-Authored-By: X <x@y.z>\nClaude-Session: https://s"]
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    gitops.run_commit(ctx)
    body = sh(harness.repo, "log", "-1", "--format=%B").lower()
    assert "co-authored-by" not in body and "claude-session" not in body


def test_commit_add_list_is_explicit_and_scan_precedes_commit(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    gitops.run_commit(ctx)
    adds = [a for a in harness.argvs if a[:2] == ["git", "add"]]
    assert adds == [["git", "add", "--", "a.py", "b.py"]]
    flat = [" ".join(a) for a in harness.argvs]
    scan = next(i for i, a in enumerate(flat) if a.endswith("check_staged.py"))
    commit = next(i for i, a in enumerate(flat) if a.startswith("git commit"))
    assert flat.index(" ".join(adds[0])) < scan < commit
    commit_argv = harness.argvs[commit]
    assert "-a" not in commit_argv and "--no-verify" not in commit_argv and "-am" not in commit_argv


def test_commit_gate_shows_message_and_stat(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    gitops.run_commit(ctx)
    ((question, details),) = harness.gates
    assert MESSAGE in details and "b.py" in details and "commit" in question.lower()


def test_commit_gate_denied_leaves_head_untouched(harness, tmp_path):
    harness.gate_answer = False
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    before = sh(harness.repo, "rev-parse", "HEAD")
    with pytest.raises(ShipError) as caught:
        gitops.run_commit(ctx)
    assert caught.value.exit_code == EXIT_GATE_DENIED
    assert sh(harness.repo, "rev-parse", "HEAD") == before


def test_commit_refuses_when_staged_scan_fails(harness, tmp_path):
    harness.scan_rc = 1
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    before = sh(harness.repo, "rev-parse", "HEAD")
    with pytest.raises(ShipError) as caught:
        gitops.run_commit(ctx)
    assert caught.value.exit_code == EXIT_CHECKS_FAIL
    assert sh(harness.repo, "rev-parse", "HEAD") == before
    assert harness.gates == []


def test_commit_refuses_junk_files_before_calling_the_model(harness, tmp_path):
    (harness.repo / "debug.log").write_text("noise", encoding="utf-8")
    dirty_repo(harness)
    ctx = make_ctx(harness, tmp_path)
    harness.llm_texts = ["not a valid message"]
    with pytest.raises(ShipError, match="debug.log"):
        gitops.run_commit(ctx)
    assert not any(a[:2] == ["git", "add"] for a in harness.argvs)


def test_commit_includes_files_that_were_already_staged(harness, tmp_path):
    (harness.repo / "a.py").write_text("print(2)\n", encoding="utf-8")
    (harness.repo / "c.py").write_text("c = 1\n", encoding="utf-8")
    sh(harness.repo, "add", "a.py")
    result = gitops.run_commit(make_ctx(harness, tmp_path))
    assert result.data["files"] == ["a.py", "c.py"]
    assert sh(harness.repo, "status", "--porcelain") == ""


def test_commit_rejects_invalid_model_message_twice(harness, tmp_path):
    harness.llm_texts = ["added stuff"]
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    before = sh(harness.repo, "rev-parse", "HEAD")
    with pytest.raises(ShipError, match="отклонён"):
        gitops.run_commit(ctx)
    assert sh(harness.repo, "rev-parse", "HEAD") == before


def test_commit_on_clean_tree_reuses_the_days_commit(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    first = gitops.run_commit(ctx)
    ctx.state.mark("commit", "done", at="t", data=first.data)
    again = gitops.run_commit(ctx)
    assert again.data["sha"] == first.data["sha"]
    assert again.data["reused"] is True


def test_commit_on_clean_tree_without_a_days_commit_fails(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    with pytest.raises(ShipError, match="нечего коммитить"):
        gitops.run_commit(ctx)


def test_recheck_commit_detects_moved_head_and_dirty_tree(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    ctx.state.mark("commit", "done", at="t", data=gitops.run_commit(ctx).data)
    assert gitops.recheck_commit(ctx) is True
    (harness.repo / "a.py").write_text("z", encoding="utf-8")
    assert gitops.recheck_commit(ctx) is False
    sh(harness.repo, "checkout", "--", "a.py")
    sh(harness.repo, "commit", "-q", "--allow-empty", "-m", "later")
    assert gitops.recheck_commit(ctx) is False


# ------------------------------------------------------------------ tag


def committed_ctx(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    ctx.state.mark("commit", "done", at="t", data=gitops.run_commit(ctx).data)
    harness.gates.clear()
    harness.argvs.clear()
    return ctx


def test_tag_pushes_with_ephemeral_header_and_verifies_remote(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    result = gitops.run_tag(ctx)
    sha = sh(harness.repo, "rev-parse", "HEAD")
    assert sh(harness.repo, "rev-parse", f"refs/tags/{TAG}^{{commit}}") == sha
    assert result.data == {
        "tag": TAG,
        "sha": sha,
        "code_url": f"{GITHUB_URL}/tree/{TAG}",
        "remote_verified": True,
    }
    (argv,) = [a for a in harness.argvs if "push" in a]
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert argv[:5] == [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        f"http.extraheader=AUTHORIZATION: basic {basic}",
    ]
    assert argv[5:] == ["push", "origin", "main", f"refs/tags/{TAG}"]
    assert not any(TOKEN in part for part in argv)  # only the encoded header carries it
    assert not any("--force" in part or part.startswith("+") for part in argv)
    assert harness.push_envs == [{"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}]
    assert TOKEN not in sh(harness.repo, "config", "--list", "--local")


def test_tag_ignores_untracked_files_and_warns_with_a_count(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    (harness.repo / "AGENTS.md").write_text("n\n", encoding="utf-8")
    (harness.repo / "secret_draft.txt").write_text("n\n", encoding="utf-8")
    result = gitops.run_tag(ctx)
    assert result.data["warning"] == "untracked: 2"
    assert "AGENTS" not in str(result.data) and "secret_draft" not in str(result.data)


def test_tag_without_untracked_files_has_no_warning(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    assert "warning" not in gitops.run_tag(ctx).data


def test_tag_blocked_by_modified_tracked_file_even_with_untracked(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    (harness.repo / "AGENTS.md").write_text("n\n", encoding="utf-8")
    (harness.repo / "a.py").write_text("print(3)\n", encoding="utf-8")
    with pytest.raises(ShipError, match="рабочее дерево не чистое: a.py"):
        gitops.run_tag(ctx)


def test_recheck_commit_ignores_untracked_files(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    (harness.repo / "AGENTS.md").write_text("n\n", encoding="utf-8")
    assert gitops.recheck_commit(ctx) is True
    (harness.repo / "a.py").write_text("print(4)\n", encoding="utf-8")
    assert gitops.recheck_commit(ctx) is False


def test_tag_checks_extraheader_after_push(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    gitops.run_tag(ctx)
    flat = [a for a in harness.argvs]
    push_at = next(i for i, a in enumerate(flat) if "push" in a)
    assert any(a[1:4] == ["config", "--get-all", "http.extraheader"] for a in flat[push_at + 1 :])


def test_tag_fails_when_extraheader_survives(harness, tmp_path):
    harness.config_extraheader = "AUTHORIZATION: basic xxx"
    ctx = committed_ctx(harness, tmp_path)
    with pytest.raises(ShipError, match="extraheader"):
        gitops.run_tag(ctx)


def test_tag_checks_extraheader_even_when_push_blows_up(harness, tmp_path):
    harness.push_raises = ShipError("команда превысила таймаут 300 с: git")
    ctx = committed_ctx(harness, tmp_path)
    with pytest.raises(ShipError, match="таймаут"):
        gitops.run_tag(ctx)
    assert any(a[1:4] == ["config", "--get-all", "http.extraheader"] for a in harness.argvs)


def test_push_failure_hides_token_and_header(harness, tmp_path):
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    harness.push_rc = 1
    harness.push_stderr = f"fatal: header {basic} rejected, token {TOKEN}, Authentication failed"
    ctx = committed_ctx(harness, tmp_path)
    with pytest.raises(ShipError) as caught:
        gitops.run_tag(ctx)
    text = f"{caught.value} {caught.value.hint}"
    assert TOKEN not in text and basic not in text
    assert caught.value.exit_code == EXIT_AUTH


def test_tag_gate_denied_does_not_push(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    harness.gate_answer = False
    with pytest.raises(ShipError) as caught:
        gitops.run_tag(ctx)
    assert caught.value.exit_code == EXIT_GATE_DENIED
    assert not any("push" in a for a in harness.argvs)


def test_pushed_tag_on_another_commit_is_never_moved(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    harness.remote[f"tags/{TAG}"] = ("commit", "f" * 40)
    with pytest.raises(ShipError, match="уже запушен"):
        gitops.run_tag(ctx)
    assert not any("push" in a for a in harness.argvs)
    assert harness.gates == []


def test_unpushed_local_tag_on_old_commit_is_moved(harness, tmp_path):
    old = sh(harness.repo, "rev-parse", "HEAD")
    sh(harness.repo, "tag", TAG, old)
    ctx = committed_ctx(harness, tmp_path)
    gitops.run_tag(ctx)
    assert sh(harness.repo, "rev-parse", f"refs/tags/{TAG}^{{commit}}") == sh(
        harness.repo, "rev-parse", "HEAD"
    )


def test_already_pushed_tag_and_branch_skip_gate_and_push(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    sha = sh(harness.repo, "rev-parse", "HEAD")
    harness.remote[f"tags/{TAG}"] = ("commit", sha)
    harness.remote["heads/main"] = ("commit", sha)
    result = gitops.run_tag(ctx)
    assert result.data["remote_verified"] is True
    assert harness.gates == []
    assert not any("push" in a for a in harness.argvs)


def test_tag_fails_when_remote_does_not_show_the_commit_after_push(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    harness.push_rc = 0

    original = harness.run

    def lying_run(argv, **kwargs):
        result = original(argv, **kwargs)
        if "push" in argv:
            harness.remote.pop(f"tags/{TAG}", None)
        return result

    ctx.run = lying_run
    with pytest.raises(ShipError, match="не указывает"):
        gitops.run_tag(ctx)


def test_tag_requires_head_at_recorded_commit(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    sh(harness.repo, "commit", "-q", "--allow-empty", "-m", "later")
    with pytest.raises(ShipError, match="HEAD ушёл"):
        gitops.run_tag(ctx)


def test_tag_requires_clean_tree_and_main_branch(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    (harness.repo / "a.py").write_text("s", encoding="utf-8")
    with pytest.raises(ShipError, match="не чистое"):
        gitops.run_tag(ctx)
    sh(harness.repo, "checkout", "--", "a.py")
    sh(harness.repo, "checkout", "-q", "-b", "side")
    with pytest.raises(ShipError, match="main"):
        gitops.run_tag(ctx)


def test_tag_needs_token_and_url(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    ctx.env.pop("GITHUB_TOKEN")
    with pytest.raises(ShipError, match="GITHUB_TOKEN"):
        gitops.run_tag(ctx)


def test_recheck_tag_compares_remote_sha(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    ctx.state.mark("tag", "done", at="t", data=gitops.run_tag(ctx).data)
    assert gitops.recheck_tag(ctx) is True
    harness.remote[f"tags/{TAG}"] = ("commit", "e" * 40)
    assert gitops.recheck_tag(ctx) is False
    harness.remote.pop(f"tags/{TAG}")
    assert gitops.recheck_tag(ctx) is False


def test_recheck_tag_false_when_main_moved(harness, tmp_path):
    ctx = committed_ctx(harness, tmp_path)
    ctx.state.mark("tag", "done", at="t", data=gitops.run_tag(ctx).data)
    harness.remote["heads/main"] = ("commit", "f" * 40)
    assert gitops.recheck_tag(ctx) is False
    harness.remote.pop("heads/main")
    assert gitops.recheck_tag(ctx) is False


def test_commit_reuses_head_when_work_is_committed_and_only_untracked_files_remain(
    harness, tmp_path
):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    first = gitops.run_commit(ctx)
    (harness.repo / "AGENTS.md").write_text("owner file\n", encoding="utf-8")  # untracked
    head = sh(harness.repo, "rev-parse", "HEAD")
    again = gitops.run_commit(ctx)
    assert again.data["sha"] == first.data["sha"] == head
    assert again.data["reused"] is True
    assert sh(harness.repo, "rev-parse", "HEAD") == head
    assert sh(harness.repo, "ls-files", "AGENTS.md") == ""


def test_commit_does_not_reuse_head_when_a_tracked_file_is_dirty(harness, tmp_path):
    ctx = make_ctx(harness, tmp_path)
    dirty_repo(harness)
    first = gitops.run_commit(ctx)
    (harness.repo / "a.py").write_text("print(3)\n", encoding="utf-8")
    second = gitops.run_commit(ctx)
    assert second.data["sha"] != first.data["sha"]
    assert "reused" not in second.data


def _no_tty_gate(question, details):
    raise ShipError("нужен tty для подтверждения", exit_code=EXIT_NO_TTY)


@pytest.mark.parametrize("stale_local_tag", [False, True])
def test_tag_gate_without_tty_changes_nothing(harness, tmp_path, stale_local_tag):
    old = sh(harness.repo, "rev-parse", "HEAD")
    if stale_local_tag:
        sh(harness.repo, "tag", TAG, old)
    ctx = committed_ctx(harness, tmp_path)
    ctx.gate = _no_tty_gate
    tags_before = sh(harness.repo, "tag", "--list")
    refs_before = sh(harness.repo, "for-each-ref", "refs/tags")
    harness.argvs.clear()
    with pytest.raises(ShipError) as caught:
        gitops.run_tag(ctx)
    assert caught.value.exit_code == 8
    # no write of any kind was issued: no `git tag`, no push
    assert not any(a[:2] == ["git", "tag"] or "push" in a for a in harness.argvs)
    assert sh(harness.repo, "tag", "--list") == tags_before == (TAG if stale_local_tag else "")
    assert sh(harness.repo, "for-each-ref", "refs/tags") == refs_before
    assert harness.pushed_sha is None
