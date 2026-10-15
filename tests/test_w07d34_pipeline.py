"""Day 34 pipeline: scripted model through the REAL files MCP server on a tmp git repo."""

from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from advent_core import console, journal
from advent_core.config import Config
from advent_core.errors import AdventError, MCPError
from advent_core.mcp_client import ToolCallOutcome
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, RawToolCall, Usage
from adventassistant import cli, files_pipeline, files_store
from tests.w07d34_repo import git, make_repo, write

MODEL = "ministral-14b-latest"
CONFIG = Config(api_key="k", model=MODEL, params=GenerationParams())
GIT_ID = ("-c", "user.name=t", "-c", "user.email=t@t")


def tool_round(*calls: tuple[str, dict]) -> CallResult:
    return CallResult(
        text="",
        model_requested=MODEL,
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        tool_calls=tuple(
            RawToolCall(id=f"c{i}", name=name, arguments=json.dumps(args, ensure_ascii=False))
            for i, (name, args) in enumerate(calls)
        ),
    )


def final(text: str) -> CallResult:
    return CallResult(
        text=text,
        model_requested=MODEL,
        finish_reason="stop",
        usage=Usage(prompt_tokens=200, completion_tokens=20, total_tokens=220),
    )


class Model:
    """Scripted rounds; sent_messages aliases the live list on purpose (day-17 trap)."""

    def __init__(self, *steps, expect: dict[int, tuple[str, ...]] | None = None) -> None:
        self.steps = list(steps)
        # call index -> substrings the tool messages sent with that call must contain
        self.expect = expect or {}
        self.n = 0
        self.tools_seen: list = []
        self.on_call = None

    def complete(self, config, messages, capabilities=None, *, tools=None, tool_choice=None):
        self.tools_seen.append(tools)
        if self.on_call:
            self.on_call(self.n)
        tool_text = chr(10).join(str(m.get("content")) for m in messages if m.get("role") == "tool")
        for needle in self.expect.get(self.n, ()):
            assert needle in tool_text, (
                f"call {self.n}: tool results never reached the model, missing {needle!r}"
            )
        step = self.steps[min(self.n, len(self.steps) - 1)]
        self.n += 1
        if isinstance(step, Exception):
            raise step
        return replace(step, sent_messages=messages)

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("tool rounds use complete()")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = make_repo(tmp_path / "repo")
    write(root, "week_07/task_07.md", "# Week 7\nDay 34 files\n")
    write(
        root,
        "adventassistant/cli.py",
        '@app.command("files")\ndef a(): ...\n\n@app.command("files-apply")\ndef b(): ...\n',
    )
    write(root, ".gitignore", "logs/\n")
    git(root, "add", "-A", "-f")
    git(root, *GIT_ID, "commit", "-qm", "init")
    return root


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(files_pipeline, "cloud_config", lambda _model: CONFIG)


def run(repo: Path, model: Model, goal: str = "цель", **kwargs):
    return files_pipeline.run_files(
        goal, complete=model.complete, root=repo, config=CONFIG, **kwargs
    )


def journal_rows() -> list[dict]:
    path = journal.LOG_DIR / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def scenario1() -> Model:
    return Model(
        tool_round(("search_text", {"query": "call_tool_once"})),
        tool_round(
            ("read_file", {"path": "src/app.py"}),
            ("read_file", {"path": "src/other.py"}),
        ),
        final("Вызовы: src/app.py:5 и src/other.py:2."),
        expect={
            1: ("src/app.py:1: def call_tool_once", "src/other.py:2: value = call_tool_once()"),
            2: ("def call_tool_once", "from app import call_tool_once", "совпадений: 4"),
        },
    )


def test_scenario_search_then_reads_empty_diff_and_split_streams(repo, capsys) -> None:
    model = scenario1()
    result = run(repo, model)
    out, err = capsys.readouterr()

    assert out.startswith("Вызовы: src/app.py:5 и src/other.py:2.")
    assert "Изменений нет." in out
    assert "run:" not in out and "применить" not in out
    assert "· " not in out and "search_text(" not in out
    assert "search_text(query=call_tool_once) → 4 совпадений в 2 файлах" in err
    assert "read_file(path=src/app.py) → прочитано 1–5 из 5" in err
    assert "read_file(path=src/other.py) → прочитано 1–2 из 2" in err
    assert "· раундов 3  ·  вызовов 3  ·  прочитано файлов 2 / символов" in err
    assert "· токены 400/40/440  ·  стоимость $" in err
    assert "max_context_length" not in err
    assert "diff:" not in err
    assert (result.rounds, result.calls, result.diff) == (3, 3, "")
    assert {t["function"]["name"] for t in model.tools_seen[0]} == {
        "edit_file",
        "list_files",
        "read_file",
        "search_text",
        "write_file",
    }
    assert len(model.tools_seen[0]) == 5


def test_journal_one_row_per_round_with_snapshots_and_no_schemas(repo, capsys) -> None:
    run(repo, scenario1())
    rows = journal_rows()
    assert [r["round"] for r in rows] == [1, 2, 3]
    assert {r["day"] for r in rows} == {34}
    assert {r["week"] for r in rows} == {7}
    assert {r["command"] for r in rows} == {"assistant_files"}
    assert len({r["run_id"] for r in rows}) == 1
    assert rows[0]["tool_calls"] == ["search_text"]
    assert rows[1]["tool_calls"] == ["read_file", "read_file"]
    assert rows[2]["tool_calls"] == []

    def roles(row):
        return [m["role"] for m in row["messages"]]

    assert "tool" not in roles(rows[0])
    assert roles(rows[1]).count("tool") == 1
    assert roles(rows[2]).count("tool") == 3
    assert "совпадений: 4" not in json.dumps(rows[0], ensure_ascii=False)
    assert "совпадений: 4" in json.dumps(rows[1], ensure_ascii=False)
    assert "parameters" not in json.dumps(rows, ensure_ascii=False)
    # the files prompt, not the project persona, is the system message
    assert "только через инструменты" in rows[0]["messages"][0]["content"]


README = "# Неделя 7\n\n`files` — агент работает с файлами.\n`files-apply` — применить changeset.\n"


def scenario2() -> Model:
    return Model(
        tool_round(
            ("list_files", {"pattern": "week_07/*"}),
            ("read_file", {"path": "week_07/task_07.md"}),
            ("read_file", {"path": "adventassistant/cli.py"}),
        ),
        tool_round(("read_file", {"path": "docs/a.md"})),
        tool_round(("write_file", {"path": "week_07/README.md", "content": README})),
        final("README недели 7 подготовлен: week_07/README.md."),
        expect={
            1: ("week_07/task_07.md", "Day 34 files", '@app.command("files-apply")'),
            2: ("Beta line",),
            # three distinct files read plus the staged write, all visible to the last call
            3: ("Day 34 files", '@app.command("files")', "Beta line", "staged week_07/README.md"),
        },
    )


def test_scenario_readme_diff_shown_tree_untouched_then_apply_once(
    repo, capsys, monkeypatch
) -> None:
    before = git(repo, "status", "--porcelain")
    result = run(repo, scenario2())
    out, err = capsys.readouterr()

    assert git(repo, "status", "--porcelain") == before
    assert not (repo / "week_07" / "README.md").exists()
    assert "Изменения:\n--- /dev/null\n+++ b/week_07/README.md" in out
    assert "+`files-apply` — применить changeset." in out
    assert f"run: {result.run_id}" in out
    assert f"применить: adventassistant files-apply {result.run_id}" in out
    assert out.index("Изменения:") < out.index("run:")
    assert "write_file(path=week_07/README.md, content=<" in err
    assert "→ staged week_07/README.md (+4)" in " ".join(err.split())
    assert "обрезка идёт по символьному" not in err
    assert f"diff: logs/files/{result.run_id}/changes.diff" in err
    assert (repo / "logs" / "files" / result.run_id / "changes.diff").read_text(
        encoding="utf-8"
    ) == result.diff
    # README lists exactly the commands of the (fixture) cli.py
    cli_text = (repo / "adventassistant" / "cli.py").read_text(encoding="utf-8")
    commands = set(re.findall(r'app\.command\("([^"]+)"\)', cli_text))
    mentioned = set(re.findall(r"`([a-z-]+)`", README))
    assert commands == mentioned

    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    applied = CliRunner().invoke(cli.app, ["files-apply", result.run_id])
    assert applied.exit_code == 0, applied.output
    assert (repo / "week_07" / "README.md").read_text(encoding="utf-8") == README
    assert "записано: week_07/README.md" in applied.stderr
    assert "?? week_07/README.md" in applied.stderr
    assert applied.stdout == ""

    monkeypatch.setattr(sys, "argv", ["adventassistant", "files-apply", result.run_id])
    with pytest.raises(SystemExit) as again:
        cli.main()
    assert again.value.code == 12
    assert "уже применён" in capsys.readouterr().err


def test_configured_secret_in_diff_context_is_redacted_everywhere(
    repo, capsys, monkeypatch
) -> None:
    secret = "ctx-secret-value-not-a-key"
    write(repo, "docs/leak.md", f"head\ntoken = {secret}\ntail\nchange me\n")
    git(repo, "add", "-f", "docs/leak.md")
    git(repo, *GIT_ID, "commit", "-qm", "leak")
    monkeypatch.setenv("MISTRAL_API_KEY", secret)
    model = Model(
        tool_round(("edit_file", {"path": "docs/leak.md", "old": "change me", "new": "changed"})),
        final("готово"),
    )
    result = run(repo, model)
    out, err = capsys.readouterr()

    assert secret not in out and secret not in err
    assert "token = ***" in out
    assert "+changed" in out
    saved = (repo / "logs" / "files" / result.run_id / "changes.diff").read_text(encoding="utf-8")
    assert secret not in saved and secret not in result.diff
    staged = repo / "logs" / "files" / result.run_id / "files" / "docs" / "leak.md"
    assert secret in staged.read_text(encoding="utf-8")

    files_store.apply_run(result.run_id, repo)
    assert secret in (repo / "docs" / "leak.md").read_text(encoding="utf-8")


def test_calls_limit_with_staged_change_exits_11_after_partial_diff(repo, capsys) -> None:
    model = Model(
        tool_round(
            ("write_file", {"path": "docs/new.md", "content": "x\n"}),
            ("search_text", {"query": "alpha"}),
        ),
        final("не дойдём"),
    )
    with pytest.raises(files_pipeline.LimitReachedError) as caught:
        run(repo, model, max_calls=1)
    assert caught.value.exit_code == 11
    out, err = capsys.readouterr()
    assert "Изменения (частичный результат):" in out
    assert "+++ b/docs/new.md" in out
    assert "применить: adventassistant files-apply" in out
    assert "лимит вызовов" in err
    assert "ОШИБКА: лимит вызовов инструментов исчерпан" in err
    assert not (repo / "docs" / "new.md").exists()
    assert [r["round"] for r in journal_rows()] == [1]  # the shared last round once


def test_rounds_limit_exits_11_and_empty_diff_is_marked_partial(repo, capsys) -> None:
    model = Model(tool_round(("list_files", {})))
    with pytest.raises(files_pipeline.LimitReachedError) as caught:
        run(repo, model, max_rounds=2)
    assert caught.value.exit_code == 11
    out, _err = capsys.readouterr()
    assert "Изменений нет. (частичный результат)" in out
    assert "run:" not in out
    assert [r["round"] for r in journal_rows()] == [1, 2]


def cut_off(reason: str) -> CallResult:
    return replace(final("# README\n\nполовина текст"), finish_reason=reason)


@pytest.mark.parametrize("reason", ["error", "length"])
def test_cut_off_final_with_staged_change_exits_7_and_hides_text(repo, capsys, reason) -> None:
    model = Model(
        tool_round(("write_file", {"path": "docs/new.md", "content": "x\n"})),
        cut_off(reason),
    )
    with pytest.raises(files_pipeline.StreamTruncated) as caught:
        run(repo, model)
    assert caught.value.exit_code == 7
    assert f"finish_reason={reason}" in caught.value.message
    out, _err = capsys.readouterr()
    assert "Изменения (частичный результат):" in out
    assert "+++ b/docs/new.md" in out
    assert "run:" in out
    assert "половина текст" not in out
    assert [r["round"] for r in journal_rows()] == [1, 2]


@pytest.mark.parametrize("reason", ["error", "length"])
def test_cut_off_final_without_changes_is_partial_not_success(repo, capsys, reason) -> None:
    with pytest.raises(files_pipeline.StreamTruncated) as caught:
        run(repo, Model(cut_off(reason)))
    assert caught.value.exit_code == 7
    out, err = capsys.readouterr()
    assert "Изменений нет. (частичный результат)" in out
    assert "половина текст" not in out
    assert "половина текст" in err
    assert [r["round"] for r in journal_rows()] == [1]


def test_read_budget_returns_tool_error_and_stops_counting(repo, capsys, monkeypatch) -> None:
    monkeypatch.setattr(files_pipeline, "READ_CALLS_BUDGET", 2)
    model = Model(
        tool_round(
            ("read_file", {"path": "docs/a.md"}),
            ("read_file", {"path": "docs/b.txt"}),
            ("read_file", {"path": "src/app.py"}),
        ),
        final("ок"),
    )
    result = run(repo, model)
    _out, err = capsys.readouterr()
    assert "read_file(path=src/app.py) → ОШИБКА: бюджет чтения исчерпан" in err
    assert "прочитано файлов 2 /" in err
    assert result.calls == 3


def test_read_chars_budget(repo, capsys, monkeypatch) -> None:
    monkeypatch.setattr(files_pipeline, "READ_CHARS_BUDGET", 10)
    model = Model(
        tool_round(("read_file", {"path": "docs/a.md"})),
        tool_round(("read_file", {"path": "docs/b.txt"})),
        final("ок"),
    )
    run(repo, model)
    _out, err = capsys.readouterr()
    assert "read_file(path=docs/b.txt) → ОШИБКА: бюджет чтения исчерпан" in err


def test_dirty_tracked_files_warned_before_the_first_model_call(repo, capsys) -> None:
    write(repo, "docs/a.md", "dirty\n")
    model = Model(final("готово"))
    seen: list[str] = []
    model.on_call = lambda _n: seen.append(capsys.readouterr().err)
    run(repo, model)
    assert len(seen) == 1
    assert "незакоммиченные правки tracked-файлов" in seen[0]
    assert "docs/a.md" in seen[0]


def test_clean_tree_has_no_dirty_warning(repo, capsys) -> None:
    run(repo, Model(final("готово")))
    assert "незакоммиченные" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("goal", "kwargs"),
    [
        ("", {}),
        ("   ", {}),
        ("x" * 2001, {}),
        ("ok", {"max_rounds": 0}),
        ("ok", {"max_rounds": 16}),
        ("ok", {"max_calls": 0}),
        ("ok", {"max_calls": 61}),
    ],
)
def test_validation_precedes_any_model_call_or_run_dir(repo, goal, kwargs) -> None:
    model = Model(final("не должно вызываться"))
    with pytest.raises(AdventError):
        run(repo, model, goal, **kwargs)
    assert model.n == 0
    assert not (repo / "logs").exists()


def test_not_a_git_repo_is_refused_before_the_model(tmp_path) -> None:
    model = Model(final("x"))
    with pytest.raises(AdventError, match="не является git-репозиторием"):
        run(tmp_path, model)
    assert model.n == 0


def test_api_error_prints_partial_diff_keeps_changeset_and_journals_paid_rounds(
    repo, capsys
) -> None:
    model = Model(
        tool_round(("write_file", {"path": "docs/new.md", "content": "x\n"})),
        AdventError("fake api failure"),
    )
    with pytest.raises(AdventError, match="fake api failure"):
        run(repo, model)
    out, err = capsys.readouterr()
    assert "Изменения (частичный результат):" in out and "+++ b/docs/new.md" in out
    assert "раундов 1" in err
    runs = list((repo / "logs" / "files").iterdir())
    assert len(runs) == 1
    assert "docs/new.md" in files_store.load_manifest(runs[0].name, repo)["files"]
    assert [r["round"] for r in journal_rows()] == [1]


def test_mcp_error_on_a_call_prints_diff_and_reraises(repo, capsys) -> None:
    calls: list[str] = []

    def broken(name, arguments):
        calls.append(name)
        raise MCPError("сервер упал")

    model = Model(tool_round(("list_files", {})), final("x"))
    with pytest.raises(MCPError, match="сервер упал"):
        run(repo, model, call_tool=broken)
    out, err = capsys.readouterr()
    assert calls == ["list_files"]
    assert "ОШИБКА MCP: сервер упал" in err
    assert "Изменений нет. (частичный результат)" in out


def test_injected_call_tool_is_used_but_budget_still_applies(repo, capsys) -> None:
    seen: list[str] = []

    def fake(name, arguments):
        seen.append(name)
        return ToolCallOutcome(text="a.md: строки 1-1 из 1\n1: x", is_error=False)

    model = Model(tool_round(("read_file", {"path": "a.md"})), final("ок"))
    result = run(repo, model, call_tool=fake)
    assert seen == ["read_file"]
    assert result.calls == 1


def test_trace_renders_at_width_80_without_ellipsis(repo, monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(
        console, "err", Console(file=buffer, width=80, force_terminal=False, highlight=False)
    )
    long_path = "docs/" + "a" * 70 + ".md"
    model = Model(
        tool_round(("read_file", {"path": long_path}), ("search_text", {"query": "q" * 90})),
        final("ок"),
    )
    run(repo, model)
    text = buffer.getvalue()
    assert "…" not in text
    assert "search_text(" in text


def test_summaries() -> None:
    s = files_pipeline.summarize_result
    assert s("list_files", "a\nb\nвсего: 2", False) == "2 файлов"
    assert (
        s("search_text", "x:1: y\nсовпадений: 3 в файлах: 1; скан обрезан (…)", False)
        == "3 совпадений в 1 файлах (скан обрезан)"
    )
    assert s("read_file", "p: строки 1-120 из 230\n1: a", False) == "прочитано 1–120 из 230"
    assert s("write_file", "staged a.md (+48)", False) == "staged a.md (+48)"
    assert s("read_file", "boom\nmore", True) == "ОШИБКА: boom"
    assert files_pipeline.short_args("edit_file", {"path": "a", "old": "xx", "new": "y"}) == (
        "path=a, old=<2 симв.>, new=<1 симв.>"
    )


def test_render_output_shapes() -> None:
    plain = files_pipeline.render_output("ответ", "", "RID", partial=False)
    assert plain == "ответ\n\nИзменений нет."
    with_diff = files_pipeline.render_output("ответ", "--- a\n+++ b\n", "RID", partial=False)
    assert with_diff.endswith("run: RID\nприменить: adventassistant files-apply RID")
    assert "Изменения:\n--- a\n+++ b\n\nrun:" in with_diff


# --- CLI ----------------------------------------------------------------------


def test_cli_files_runs_end_to_end_with_explicit_complete(repo, monkeypatch) -> None:
    model = scenario2()
    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    monkeypatch.setattr(cli.chat_core, "complete", model.complete)
    result = CliRunner().invoke(cli.app, ["files", "Напиши README недели 7", "--max-rounds", "9"])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert result.stdout.startswith("README недели 7 подготовлен")
    assert "+++ b/week_07/README.md" in result.stdout
    assert "write_file(" in result.stderr
    assert "write_file(" not in result.stdout


def test_cli_files_passes_options(monkeypatch) -> None:
    seen: dict = {}

    def fake(goal, **kwargs):
        seen.update(goal=goal, **kwargs)

    monkeypatch.setattr(cli, "run_files", fake)
    CliRunner().invoke(cli.app, ["files", "цель"])
    assert (seen["max_rounds"], seen["max_calls"]) == (12, 24)
    assert seen["complete"] is cli.chat_core.complete
    CliRunner().invoke(cli.app, ["files", "цель", "--max-rounds", "15", "--max-calls", "60"])
    assert (seen["max_rounds"], seen["max_calls"]) == (15, 60)


@pytest.mark.parametrize(
    "args",
    [
        ["--max-rounds", "0"],
        ["--max-rounds", "16"],
        ["--max-calls", "0"],
        ["--max-calls", "61"],
    ],
)
def test_cli_option_range_errors(monkeypatch, args) -> None:
    monkeypatch.setattr(cli, "run_files", lambda *a, **k: pytest.fail("must not run"))
    result = CliRunner().invoke(cli.app, ["files", "цель", *args])
    assert result.exit_code == 2


def test_cli_limit_exit_code_goes_through_main(repo, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    monkeypatch.setattr(cli.chat_core, "complete", Model(tool_round(("list_files", {}))).complete)
    monkeypatch.setattr(sys, "argv", ["adventassistant", "files", "цель", "--max-rounds", "1"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 11
    assert "Лимит раундов исчерпан" in capsys.readouterr().err


def test_cli_files_apply_unknown_run_is_a_clean_error(repo, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["adventassistant", "files-apply", "bad"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1
    assert "Некорректный run_id" in capsys.readouterr().err


def test_existing_commands_still_listed_at_width_80(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "80")
    result = CliRunner().invoke(cli.app, ["--help"], env={"COLUMNS": "80"})
    assert result.exit_code == 0
    for name in (
        "index",
        "help",
        "index-review",
        "index-support",
        "support",
        "review",
        "watch-pr",
        "files",
        "files-apply",
    ):
        assert re.search(rf"^\W*{re.escape(name)}\s", result.stdout, re.MULTILINE), name
    assert "…" not in result.stdout
