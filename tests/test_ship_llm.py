"""LLM stages with a fake `ctx.complete`: validators, retry rules, journal rows, report file."""

import json
import os
import subprocess

import pytest

from advent_core.telemetry import CallResult
from adventship import llm
from adventship.ctx import EXIT_TRUNCATED, ShipContext, ShipError, TaskSpec
from adventship.state import ShipState

TAG = "w07d35"
# Built by concatenation: this test file is committed, and the staged scan greps for these.
LOCAL_PATH = "D" + ":\\Dev\\secret"
TOKEN = "gh" + "p_" + "a" * 30
DESCRIPTION = (
    "Автоматизируем сдачу учебного дня: пайплайн проводит проверки, review, commit и тег. "
    "Модель пишет тексты, а решения принимает код. Код лежит в пакете adventship."
)
GENERIC = "Автоматизация проверок релиза в магазине приложений: пайплайн сам готовит сдачу работы."


class FakeComplete:
    def __init__(self, *texts, finish_reason="stop"):
        self.texts = list(texts)
        self.finish_reason = finish_reason
        self.calls: list[list[dict]] = []

    def __call__(self, config, messages, *args, **kwargs):
        self.calls.append(list(messages))
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return CallResult(text=text, model_requested="m", finish_reason=self.finish_reason)


def sh(repo, *argv):
    subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "t@example.com")
    sh(root, "config", "user.name", "t")
    (root / "a.py").write_text("print(1)\n", encoding="utf-8")
    sh(root, "add", ".")
    sh(root, "commit", "-q", "-m", "init")
    sh(root, "tag", "w07d33")
    (root / "old.py").write_text("y = 1\n", encoding="utf-8")
    sh(root, "add", ".")
    sh(root, "commit", "-q", "-m", "feat(w07d34): previous day")
    sh(root, "tag", "w07d34")
    (root / "a.py").write_text("print(2)  # SECRETBODY\n", encoding="utf-8")
    (root / "adventship_core.py").write_text("x = 1\n", encoding="utf-8")
    sh(root, "add", ".")
    sh(root, "commit", "-q", "-m", "feat(w07d35): add adventship pipeline core")
    (root / "owner_notes.md").write_text("private\n", encoding="utf-8")  # untracked
    return root


def make_ctx(repo, complete, env=None):
    def run(argv, *, timeout, input=None, env=None):
        return subprocess.run(
            argv, cwd=repo, capture_output=True, text=True, encoding="utf-8", input=input
        )

    workdir = repo.parent / "ship"
    return ShipContext(
        week=7,
        day=35,
        tag=TAG,
        root=repo,
        workdir=workdir,
        env=env or {},
        state=ShipState.load(workdir / "state.json"),
        run=run,
        complete=complete,
        gate=lambda q, d: True,
        driver_factory=lambda: None,
        now=lambda: "2026-10-16T10:00:00+03:00",
        task=TaskSpec(
            week=7, day=35, source="task_07.md", text="## Day 35\nАвтоматизировать сдачу."
        ),
        journal_path=repo.parent / "journal.jsonl",
    )


@pytest.fixture(autouse=True)
def no_real_config(monkeypatch):
    monkeypatch.setattr(llm, "_config", lambda max_tokens: f"cfg-{max_tokens}")


def journal_rows(ctx):
    return [json.loads(x) for x in ctx.journal_path.read_text(encoding="utf-8").splitlines()]


def test_config_targets_default_model_in_the_cloud(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-1234567")
    built = llm._config(300)
    assert built.model == llm.DEFAULT_MODEL
    assert built.base_url is None
    assert built.stream is False


def test_describe_returns_description_and_comment_line(repo):
    complete = FakeComplete(DESCRIPTION)
    ctx = make_ctx(repo, complete)
    result = llm.run_describe(ctx)
    assert result.data["description"] == DESCRIPTION
    assert result.data["comment_line"].startswith("Автоматизируем сдачу учебного дня")
    assert len(result.data["comment_line"]) <= 200
    assert "\n" not in result.data["comment_line"]


def test_describe_prompt_has_only_task_stat_and_names(repo):
    complete = FakeComplete(DESCRIPTION)
    ctx = make_ctx(repo, complete)
    llm.run_describe(ctx)
    sent = "\n".join(m["content"] for m in complete.calls[0])
    assert "Автоматизировать сдачу" in sent
    assert "a.py" in sent and "adventship_core.py" in sent
    assert "SECRETBODY" not in sent  # file contents never leave the machine


def test_describe_input_is_the_commit_range_not_the_working_tree(repo):
    complete = FakeComplete(DESCRIPTION)
    llm.run_describe(make_ctx(repo, complete))
    sent = complete.calls[0][1]["content"]
    assert "КОММИТЫ ДНЯ (w07d34..HEAD)" in sent
    assert "feat(w07d35): add adventship pipeline core" in sent
    assert "feat(w07d34): previous day" not in sent
    assert "old.py" not in sent
    assert "owner_notes.md" not in sent
    assert "ЕЩЁ НЕ ЗАКОММИЧЕНО" not in sent


def test_describe_input_lists_dirty_tracked_files_separately(repo):
    (repo / "a.py").write_text("print(3)\n", encoding="utf-8")
    complete = FakeComplete(DESCRIPTION)
    llm.run_describe(make_ctx(repo, complete))
    sent = complete.calls[0][1]["content"]
    assert "ЕЩЁ НЕ ЗАКОММИЧЕНО (отслеживаемые файлы):\na.py" in sent
    assert "owner_notes.md" not in sent


def test_describe_rejects_generic_paraphrase_of_the_task(repo):
    ctx = make_ctx(repo, FakeComplete(GENERIC))
    with pytest.raises(ShipError, match="описание не опирается на изменения") as caught:
        llm.run_describe(ctx)
    assert caught.value.exit_code == EXIT_TRUNCATED


def test_describe_accepts_description_naming_a_changed_stem(repo):
    text = "Собран конвейер сдачи: ядро и проверки лежат в adventship_core, модель пишет текст."
    assert llm.run_describe(make_ctx(repo, FakeComplete(text))).data["description"] == text


def test_grounding_tokens_come_from_names_and_subjects():
    tokens = llm.grounding_tokens(
        ["adventship/gitutil.py", "tests/test_ship_x.py"], ["feat(w07d35): add range helper"]
    )
    assert {"adventship", "gitutil", "range", "helper"} <= tokens
    assert "tests" not in tokens and "py" not in tokens


def test_describe_is_journaled_with_kind_and_day(repo):
    ctx = make_ctx(repo, FakeComplete(DESCRIPTION))
    llm.run_describe(ctx)
    (row,) = journal_rows(ctx)
    assert row["kind"] == "ship_describe"
    assert row["day"] == 35
    assert row["tag"] == TAG


def test_describe_retries_once_on_too_long_answer(repo):
    complete = FakeComplete("слово " * 300, DESCRIPTION)
    ctx = make_ctx(repo, complete)
    result = llm.run_describe(ctx)
    assert result.data["description"] == DESCRIPTION
    assert len(complete.calls) == 2
    assert "отклонён" in complete.calls[1][-1]["content"]
    assert len(journal_rows(ctx)) == 2


def test_describe_rejects_english_after_one_retry(repo):
    complete = FakeComplete("This is an English answer about shipping a day.")
    ctx = make_ctx(repo, complete)
    with pytest.raises(ShipError, match="на русском"):
        llm.run_describe(ctx)
    assert len(complete.calls) == 2


def test_describe_leak_is_fatal_without_retry(repo):
    complete = FakeComplete(f"Файл лежит в {LOCAL_PATH} и содержит настройки.", DESCRIPTION)
    ctx = make_ctx(repo, complete)
    with pytest.raises(ShipError, match="локальный путь"):
        llm.run_describe(ctx)
    assert len(complete.calls) == 1


def test_truncated_answer_exits_7(repo):
    ctx = make_ctx(repo, FakeComplete(DESCRIPTION, finish_reason="length"))
    with pytest.raises(ShipError) as caught:
        llm.run_describe(ctx)
    assert caught.value.exit_code == EXIT_TRUNCATED


def test_describe_without_task_is_a_config_error(repo):
    ctx = make_ctx(repo, FakeComplete(DESCRIPTION))
    ctx.task = None
    with pytest.raises(ShipError):
        llm.run_describe(ctx)


def test_comment_line_cuts_long_first_sentence_on_a_word():
    line = llm.comment_line("слово " * 80 + "конец.")
    assert len(line) <= 200
    assert line.endswith("…")


@pytest.mark.parametrize(
    "message",
    [
        "feat(w07d35): add ship pipeline",
        "fix(w07d35): handle empty diff",
        "docs(w07d35): publish demo links",
    ],
)
def test_commit_message_accepts_valid_lines(message):
    assert llm.validate_commit_message(message, TAG) == message


@pytest.mark.parametrize(
    "message",
    [
        "add ship pipeline",
        "feat(w07d34): wrong day",
        "feat(w07d35):no space",
        "feat(w07d35): " + "x" * 70,
        "feat(w07d35): two\nlines",
        "feat(w07d35): ok\n\nCo-Authored-By: Someone <a@b.c>",
        "feat(w07d35): ok\nClaude-Session: https://example.com/s",
        "feat(w07d35): " + LOCAL_PATH,
    ],
)
def test_commit_message_rejects_invalid_lines(message):
    with pytest.raises(ShipError):
        llm.validate_commit_message(message, TAG)


def test_trailer_check_is_case_insensitive():
    with pytest.raises(ShipError, match="Co-Authored-By"):
        llm.validate_commit_message("feat(w07d35): ok\nco-authored-by: x", TAG)


def test_generate_commit_message_uses_first_line_only(repo):
    complete = FakeComplete("`feat(w07d35): add ship pipeline`\n\nCo-Authored-By: Claude <x@y.z>")
    ctx = make_ctx(repo, complete)
    assert llm.generate_commit_message(ctx) == "feat(w07d35): add ship pipeline"
    (row,) = journal_rows(ctx)
    assert row["kind"] == "ship_commit"
    assert "w07d35" in complete.calls[0][1]["content"]


def test_generate_commit_message_retries_on_bad_format(repo):
    complete = FakeComplete("Added a pipeline", "feat(w07d35): add ship pipeline")
    ctx = make_ctx(repo, complete)
    assert llm.generate_commit_message(ctx).startswith("feat(w07d35)")
    assert len(complete.calls) == 2


def test_find_leaks_covers_env_secrets_and_sheet_link():
    sheet = "https://docs.google" + ".com/spreadsheets/d/" + "A" * 30 + "/edit"
    env = {"GITHUB_TOKEN": "tok-" + "z" * 20, "COURSE_SHEET_URL": sheet}
    assert llm.find_leaks("всё чисто", env) == []
    assert llm.find_leaks("токен tok-" + "z" * 20, env)
    assert llm.find_leaks("ссылка " + sheet, env)
    assert llm.find_leaks("идентификатор " + "A" * 30, env)
    assert llm.find_leaks("см. " + LOCAL_PATH)
    assert llm.find_leaks("ключ " + TOKEN)


def test_scrub_masks_env_secrets():
    env = {"GITHUB_TOKEN": "tok-" + "z" * 20}
    assert "z" * 20 not in llm.scrub("see tok-" + "z" * 20, env)


def test_report_is_written_as_marked_draft(repo):
    complete = FakeComplete("## Задача\nСдача дня.\n\n## Что сделано\nВсё готово.")
    ctx = make_ctx(repo, complete)
    ctx.state.mark("commit", "done", at="t", data={"sha": "abc123", "_volatile": "hidden"})
    result = llm.run_report(ctx)
    path = repo / "specs" / "reports" / f"{TAG}.md"
    text = path.read_text(encoding="utf-8")
    assert llm.DRAFT_MARKER in text and "черновик" in text.lower()
    assert result.data["report"] == f"specs/reports/{TAG}.md"
    sent = complete.calls[0][1]["content"]
    assert "abc123" in sent and "hidden" not in sent
    assert journal_rows(ctx)[0]["kind"] == "ship_report"


def test_report_rewrites_a_draft_but_keeps_a_hand_written_file(repo):
    complete = FakeComplete("## Задача\nПервая версия отчёта.", "## Задача\nВторая версия отчёта.")
    ctx = make_ctx(repo, complete)
    llm.run_report(ctx)
    llm.run_report(ctx)
    path = repo / "specs" / "reports" / f"{TAG}.md"
    assert "Вторая версия" in path.read_text(encoding="utf-8")
    path.write_text("# Мой отчёт\nнаписан руками\n", encoding="utf-8")
    result = llm.run_report(ctx)
    assert result.data["kept"] is True
    assert path.read_text(encoding="utf-8").startswith("# Мой отчёт")
    assert len(complete.calls) == 2


def test_report_refuses_facts_with_local_paths(repo):
    ctx = make_ctx(repo, FakeComplete("## Задача\nтекст"))
    ctx.state.mark("commit", "done", at="t", data={"note": LOCAL_PATH})
    with pytest.raises(ShipError, match="локальные пути"):
        llm.run_report(ctx)


def test_real_journal_is_untouched(repo):
    # conftest fails the suite if the real journal changes; the ctx journal is a temp file.
    ctx = make_ctx(repo, FakeComplete(DESCRIPTION))
    llm.run_describe(ctx)
    assert os.path.dirname(ctx.journal_path) != "logs"


@pytest.mark.parametrize(
    "task_text",
    [
        "## Day 35"
        + chr(10)
        + "Смотри "
        + "D:"
        + chr(92)
        + "Dev"
        + chr(92)
        + "PROJECTS"
        + chr(92)
        + "secret.txt",
        "## Day 35" + chr(10) + "ключ " + "sk-" + "abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_gather_inputs_refuses_leaky_task_before_calling_model(repo, task_text):
    complete = FakeComplete(DESCRIPTION)
    ctx = make_ctx(repo, complete)
    ctx.task = TaskSpec(week=7, day=35, source="task_07.md", text=task_text)
    with pytest.raises(ShipError, match="во входе для модели"):
        llm.run_describe(ctx)
    assert complete.calls == []


def test_gather_inputs_refuses_leaky_file_name(repo):
    ctx = make_ctx(repo, FakeComplete(DESCRIPTION))
    bad = "C:" + chr(92) + "Users" + chr(92) + "bob" + chr(92) + "x.py"
    with pytest.raises(ShipError, match="во входе для модели"):
        llm.gather_inputs(ctx, [bad])
