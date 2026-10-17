"""Codex review stage with a fake `ctx.run`: argv contract, verdict parsing, failure exits."""

import subprocess

import pytest

from adventship import review
from adventship.ctx import (
    EXIT_CONFIG,
    EXIT_REVIEW_FAIL,
    EXIT_STAGE_FAILED,
    ShipContext,
    ShipError,
    TaskSpec,
)
from adventship.state import ShipState


class FakeCodex:
    def __init__(self, text="", *, returncode=0, via_file=True, stderr="", raises=None):
        self.text = text
        self.returncode = returncode
        self.via_file = via_file
        self.stderr = stderr
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, argv, *, timeout, input=None, env=None):
        self.calls.append({"argv": list(argv), "timeout": timeout, "input": input})
        if self.raises:
            raise self.raises
        if self.via_file:
            path = argv[argv.index("-o") + 1]
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.text)
            stdout = "progress noise"
        else:
            stdout = self.text
        return subprocess.CompletedProcess(argv, self.returncode, stdout, self.stderr)


def _git(repo, *argv):
    subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True)


def git_repo(tmp_path, *, dirty=False):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "tag", "w07d34")
    (repo / "b.py").write_text("2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feat(w07d35): add b")
    (repo / "AGENTS.md").write_text("owner file\n", encoding="utf-8")  # untracked
    if dirty:
        (repo / "a.py").write_text("changed\n", encoding="utf-8")
    return repo


def make_ctx(tmp_path, codex, env=None, *, dirty=False):
    repo = git_repo(tmp_path, dirty=dirty)

    def run(argv, *, timeout, input=None, env=None):
        if argv[0] == "git":
            return subprocess.run(argv, cwd=repo, capture_output=True, text=True, encoding="utf-8")
        return codex(argv, timeout=timeout, input=input, env=env)

    return _ctx(tmp_path, repo, run, env)


def _ctx(tmp_path, repo, run, env=None):
    workdir = repo / "logs" / "ship" / "w07d35"
    return ShipContext(
        week=7,
        day=35,
        tag="w07d35",
        root=repo,
        workdir=workdir,
        env=env or {"CODEX_BIN": "codex"},
        state=ShipState.load(workdir / "state.json"),
        run=run,
        complete=lambda *a, **k: None,
        gate=lambda q, d: True,
        driver_factory=lambda: None,
        now=lambda: "2026-10-16T10:00:00+03:00",
        task=TaskSpec(
            week=7, day=35, source="task_07.md", text="## Day 35\nАвтоматизировать сдачу."
        ),
    )


def test_pass_verdict_writes_review_file(tmp_path):
    codex = FakeCodex("NOTE: мелочь\nVERDICT: PASS\n")
    ctx = make_ctx(tmp_path, codex)
    result = review.run_review(ctx)
    assert result.data == {
        "verdict": "PASS",
        "blockers": 0,
        "review_file": "logs/ship/w07d35/review.md",
        "range": "w07d34..HEAD",
    }
    assert "VERDICT: PASS" in (ctx.workdir / "review.md").read_text(encoding="utf-8")


def test_argv_is_read_only_and_prompt_goes_through_stdin(tmp_path):
    codex = FakeCodex("VERDICT: PASS")
    review.run_review(make_ctx(tmp_path, codex))
    (call,) = codex.calls
    argv = call["argv"]
    assert argv[:2] == ["codex", "exec"]
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[-1] == "-"
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "VERDICT: PASS" in call["input"] and "VERDICT: FAIL" in call["input"]
    assert "Автоматизировать сдачу" in call["input"]
    assert call["timeout"] == review.REVIEW_TIMEOUT


def test_fail_verdict_exits_11(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("BLOCKER: x.py токен в URL\nVERDICT: FAIL"))
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_REVIEW_FAIL
    assert (ctx.workdir / "review.md").exists()


def test_pass_with_blocker_lines_still_fails(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("- BLOCKER: пропущен gate\nVERDICT: PASS"))
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_REVIEW_FAIL


def test_missing_verdict_is_a_stage_failure(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("Всё выглядит хорошо."))
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_STAGE_FAILED


def test_nonzero_exit_without_verdict_reports_stderr_tail(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("", returncode=1, stderr="boom: auth required"))
    with pytest.raises(ShipError, match="кодом 1") as caught:
        review.run_review(ctx)
    assert "auth required" in caught.value.hint


def test_nonzero_exit_fails_even_when_the_text_says_pass(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: PASS", returncode=1, stderr="crashed: oom"))
    with pytest.raises(ShipError, match="кодом 1") as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == 10
    assert "oom" in caught.value.hint


def test_zero_exit_with_pass_is_ok(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: PASS", returncode=0))
    assert review.run_review(ctx).data["verdict"] == "PASS"


def test_zero_exit_with_fail_exits_11(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: FAIL", returncode=0))
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == 11


def test_falls_back_to_stdout_when_no_output_file(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: PASS", via_file=False))
    assert review.run_review(ctx).data["verdict"] == "PASS"


def test_stale_output_file_is_not_reused(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: FAIL", via_file=False))
    ctx.workdir.mkdir(parents=True)
    (ctx.workdir / "review.last.md").write_text("VERDICT: PASS", encoding="utf-8")
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_REVIEW_FAIL


def test_timeout_error_propagates(tmp_path):
    boom = ShipError("команда превысила таймаут 1500 с: codex")
    ctx = make_ctx(tmp_path, FakeCodex(raises=boom))
    with pytest.raises(ShipError, match="таймаут"):
        review.run_review(ctx)


def test_missing_codex_is_a_config_error(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex(raises=FileNotFoundError("codex")))
    with pytest.raises(ShipError) as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_CONFIG


def test_review_text_is_scrubbed_of_env_secrets(tmp_path):
    secret = "tok-" + "q" * 20
    codex = FakeCodex(f"NOTE: токен {secret}\nVERDICT: PASS")
    ctx = make_ctx(tmp_path, codex, env={"CODEX_BIN": "codex", "GITHUB_TOKEN": secret})
    review.run_review(ctx)
    assert secret not in (ctx.workdir / "review.md").read_text(encoding="utf-8")


def test_last_verdict_line_wins():
    assert review.parse_verdict("VERDICT: FAIL\nисправлено\nVERDICT: PASS") == ("PASS", 0)
    assert review.parse_verdict("**VERDICT: FAIL**") == ("FAIL", 0)


def test_review_prompt_names_the_commit_range_and_never_untracked_files(tmp_path):
    codex = FakeCodex("VERDICT: PASS")
    review.run_review(make_ctx(tmp_path, codex))
    prompt = codex.calls[0]["input"]
    assert "ДИАПАЗОН: w07d34..HEAD" in prompt
    assert "git diff <диапазон>" in prompt
    assert "AGENTS.md" not in prompt


def test_review_records_dirty_tracked_files_as_a_warning(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: PASS"), dirty=True)
    data = review.run_review(ctx).data
    assert data["range"] == "w07d34..HEAD"
    assert data["warning"] == "незакоммиченные отслеживаемые файлы: a.py"


def test_review_has_no_warning_when_only_untracked_files_exist(tmp_path):
    data = review.run_review(make_ctx(tmp_path, FakeCodex("VERDICT: PASS"))).data
    assert "warning" not in data


def test_review_refuses_an_empty_range(tmp_path):
    ctx = make_ctx(tmp_path, FakeCodex("VERDICT: PASS"))
    _git(ctx.root, "tag", "w07d36")
    ctx.tag = "w07d37"  # base becomes w07d36 == HEAD: nothing to review
    with pytest.raises(ShipError, match="нет коммитов") as caught:
        review.run_review(ctx)
    assert caught.value.exit_code == EXIT_STAGE_FAILED
