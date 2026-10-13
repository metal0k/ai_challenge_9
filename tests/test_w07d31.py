"""Week 07 day 31: docs assistant (SPEC-w07d31.md §8). No network, no real journal."""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from rich.console import Console

from advent_core import console
from advent_core.config import Config
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult
from advent_core.errors import AdventError
from advent_core.mcp_client import ToolCallOutcome
from advent_core.telemetry import CallResult
from adventassistant import cli, corpus, pipeline, server
from adventassistant.corpus import WEEK
from week_05.chunking import MAX_CHUNK_CHARS
from week_05.index import ENDPOINT_CLOUD, load_chunks, load_meta, load_runs, write_index
from week_05.rag import RERANK_MAX_TOKENS

ROOT = Path(__file__).resolve().parents[1]
DAY = 31


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    return repo


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _commit(repo: Path) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "commit"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _seed_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Readme\n\nCOMMITTED readme text.\n")
    _write(repo, "docs/a.md", "# Документ\n\nАбзац корпуса.\n")
    _write(repo, "advent_core/schemas/demo.json", '{"type": "object"}\n')
    _write(repo, "week_02/cli.py", "print('agent')\n")
    _write(repo, "week_04/servers.json", "{}\n")
    _write(repo, "week_01/schemas/nested/no.json", "{}\n")
    sha = _commit(repo)
    _write(repo, "README.md", "# Readme\n\nDIRTY working tree.\n")
    _write(repo, "docs/secret.md", "UNTRACKED\n")
    return repo, sha


class _Ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc: object) -> bool:
        return False


def _cloud(model: str) -> Config:
    return Config(
        api_key="k" * 32,
        model=model,
        stream=False,
        system_prompt_path=Path("should-be-cleared.md"),
    )


def _patch_embed(monkeypatch, target, *, command: str) -> dict:
    captured: dict = {}

    def fake(client, model, texts, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        captured["n"] = len(texts)
        matrix = np.zeros((len(texts), 4), dtype=np.float32)
        matrix[:, 0] = 1.0
        return EmbedResult(
            vectors=matrix,
            model=model,
            prompt_tokens=len(texts) * 10,
            requests=1,
            latency_ms=1,
        )

    monkeypatch.setattr(target, "embed_texts_fn", fake)
    monkeypatch.setattr(target, "cloud_config", _cloud)
    monkeypatch.setattr(target, "mistral_client", lambda config: _Ctx())
    captured["command"] = command
    return captured


@pytest.fixture
def err_text(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(console, "err", Console(file=buf, width=200, no_color=True))
    return buf


def _chunk_row(i: int):
    from week_05.chunking import Chunk

    source = "docs/structure.md" if i == 0 else f"docs/f{i}.md"
    return Chunk(
        chunk_id=f"docs:{source}#0",
        strategy="docs",
        source=source,
        title="t",
        section=f"раздел {i}",
        ordinal=0,
        char_start=0,
        char_end=8,
        line_start=1,
        line_end=1,
        text=f"фрагмент {i}",
    )


def _write_docs_index(
    path: Path,
    *,
    n: int = 8,
    model: str = DEFAULT_EMBED_MODEL,
    purpose: str = "w07d31",
    strategy_key: str = "docs",
) -> None:
    chunks = [_chunk_row(i) for i in range(n)]
    vectors = np.zeros((n, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    embedded = EmbedResult(vectors=vectors, model=model, prompt_tokens=n, requests=1, latency_ms=1)
    write_index(
        path,
        {strategy_key: (chunks, embedded)},
        corpus_rev="a" * 40,
        corpus_files=n,
        corpus_chars=n,
        endpoint=ENDPOINT_CLOUD,
        meta={"purpose": purpose},
    )


class _Complete:
    def __init__(self, scores: list[int]) -> None:
        self.scores = scores
        self.calls: list[tuple[object, list]] = []

    def __call__(self, config, messages):
        self.calls.append((config, messages))
        if messages[0]["role"] == "system":
            return CallResult(text="три слоя", sent_messages=list(messages))
        scores = [{"id": i, "score": self.scores[i - 1]} for i in range(1, len(self.scores) + 1)]
        payload = {"scores": scores}
        return CallResult(text=json.dumps(payload), sent_messages=list(messages))


def _install_tools(monkeypatch, *, fail_branch: bool = False) -> list[str]:
    calls: list[str] = []

    def fake(cmd, name, arguments, timeout=15, quiet=True):
        calls.append(name)
        assert quiet is True
        assert timeout == 15.0
        assert cmd[1:3] == ["-m", "adventassistant.server"]
        if name == "git_branch":
            if fail_branch:
                return ToolCallOutcome(text="git упал", is_error=True)
            return ToolCallOutcome(text="ветка: main\n", is_error=False)
        if name == "git_diff_stat":
            return ToolCallOutcome(text="изменений нет\n", is_error=False)
        raise AssertionError(name)

    monkeypatch.setattr(pipeline, "call_tool_once", fake)
    return calls


def _ask(monkeypatch, db: Path, question: str, scores: list[int], *, hits: bool = True):
    monkeypatch.setattr(pipeline, "cloud_config", _cloud)
    monkeypatch.setattr(
        pipeline, "embed_question", lambda text: np.array([1, 0, 0, 0], dtype=np.float32)
    )
    if not hits:
        monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    tools = _install_tools(monkeypatch)
    complete = _Complete(scores)
    text = pipeline.run_help(question, db_path=db, complete=complete)
    return text, complete, tools


def _system_calls(complete: _Complete) -> list:
    return [call for call in complete.calls if call[1][0]["role"] == "system"]


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def test_collect_docs_keeps_only_the_allowlist(tmp_path: Path) -> None:
    repo, sha = _seed_repo(tmp_path)
    full, docs = corpus.collect_docs(repo)
    assert full == sha
    assert len(full) > 12
    sources = [doc.source for doc in docs]
    assert sources == ["README.md", "advent_core/schemas/demo.json", "docs/a.md"]
    readme = next(doc for doc in docs if doc.source == "README.md")
    assert "COMMITTED" in readme.text
    assert "DIRTY" not in readme.text
    assert all("UNTRACKED" not in doc.text for doc in docs)


def test_missing_docs_on_the_commit_is_an_error(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Readme\n")
    _write(repo, "advent_core/schemas/demo.json", "{}\n")
    _commit(repo)
    with pytest.raises(AdventError, match=r"docs/\*\*/\*\.md"):
        corpus.collect_docs(repo)


def test_missing_readme_and_schema_name_the_gap(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "docs/a.md", "# A\n")
    _commit(repo)
    with pytest.raises(AdventError, match="README.md"):
        corpus.collect_docs(repo)
    _write(repo, "README.md", "# Readme\n")
    _commit(repo)
    with pytest.raises(AdventError, match="schemas"):
        corpus.collect_docs(repo)


def test_markdown_chunks_are_relabelled_docs_and_json_is_one_chunk(tmp_path: Path) -> None:
    repo, _sha = _seed_repo(tmp_path)
    _sha, docs = corpus.collect_docs(repo)
    chunks = corpus.build_chunks(docs)
    assert chunks
    assert all(chunk.strategy == "docs" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("docs:") for chunk in chunks)
    assert all(not chunk.chunk_id.startswith("structure:") for chunk in chunks)
    json_chunks = [chunk for chunk in chunks if chunk.source.endswith(".json")]
    assert [chunk.chunk_id for chunk in json_chunks] == ["docs:advent_core/schemas/demo.json#0"]


def test_schema_longer_than_the_cap_is_not_embedded(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Readme\n")
    _write(repo, "docs/a.md", "# A\n")
    _write(repo, "advent_core/schemas/demo.json", "x" * (MAX_CHUNK_CHARS + 1))
    _commit(repo)
    captured = _patch_embed(monkeypatch, corpus, command="assistant_index")
    with pytest.raises(AdventError, match="4000"):
        corpus.build_index(repo, db_path=tmp_path / "index.sqlite3")
    assert "n" not in captured
    assert not (tmp_path / "index.sqlite3").exists()


def test_build_index_summary_and_journal_args(tmp_path: Path, monkeypatch) -> None:
    repo, sha = _seed_repo(tmp_path)
    captured = _patch_embed(monkeypatch, corpus, command="assistant_index")
    db = tmp_path / "index.sqlite3"
    summary = corpus.build_index(repo, db_path=db)
    lines = summary.splitlines()
    assert lines[0] == sha[:12]
    assert len(lines[0]) == 12
    assert lines[1] == "файлов: 3"
    n_chunks = int(lines[2].split(": ")[1])
    assert lines[3] == f"токенов: {n_chunks * 10}"
    assert lines[4] == f"стоимость: ${n_chunks * 10 / 1_000_000 * 0.1:.6f}"
    assert captured["kwargs"]["week"] == WEEK
    assert captured["kwargs"]["day"] == DAY
    assert captured["kwargs"]["journal_extra"] == {"command": "assistant_index"}
    assert "journal_path" not in captured["kwargs"]
    assert load_meta(db)["purpose"] == "w07d31"
    run = load_runs(db)["docs"]
    assert run.model == DEFAULT_EMBED_MODEL
    assert run.corpus_rev == sha
    stored = load_chunks(db, "docs")
    assert {chunk.source for chunk in stored} == {
        "README.md",
        "advent_core/schemas/demo.json",
        "docs/a.md",
    }


def test_require_index_refuses_a_foreign_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(AdventError, match="adventassistant index"):
        corpus.require_index(missing)

    wrong_purpose = tmp_path / "purpose.sqlite3"
    _write_docs_index(wrong_purpose, purpose="w05d21")
    with pytest.raises(AdventError, match="adventassistant index"):
        corpus.require_index(wrong_purpose)

    no_docs = tmp_path / "other.sqlite3"
    _write_docs_index(no_docs, strategy_key="fixed")
    with pytest.raises(AdventError, match="adventassistant index"):
        corpus.require_index(no_docs)

    wrong_model = tmp_path / "model.sqlite3"
    _write_docs_index(wrong_model, model="other-embed")
    with pytest.raises(AdventError, match="adventassistant index"):
        corpus.require_index(wrong_model)


# --------------------------------------------------------------------------
# Classification and MCP formatting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("на какой ветке", "branch"),
        ("неизменно ли это", "docs"),
        ("что изменилось", "diff"),
        ("покажи diff", "diff"),
        ("что изменилось на ветке", "diff"),
        ("что нового в дереве", "diff"),
        ("HEAD", "branch"),
        ("header файла", "docs"),
        ("покажи diff2", "diff"),
        ("which branch is it", "branch"),
    ],
)
def test_classify(question: str, kind: str) -> None:
    assert pipeline.classify(question) == kind


def test_run_help_complete_has_no_import_time_default() -> None:
    default = inspect.signature(pipeline.run_help).parameters["complete"].default
    assert default is inspect.Parameter.empty


def test_tool_names_and_diff_formatting() -> None:
    assert server.TOOL_NAMES == ("git_branch", "git_diff_stat")
    assert server.format_branch("HEAD", "abcdef123456") == "ветка: detached abcdef123456"
    assert server.format_diff("", "") == "изменений нет"
    assert server.format_diff("", "a.py | 1 +\n") == "незакоммичено:\nнет\nв индексе:\na.py | 1 +"
    kept = [f"f{i}.py | 1 +" for i in range(80)]
    assert server.clip_diff("\n".join(kept)) == "\n".join(kept)
    long = [f"f{i}.py | 1 +" for i in range(81)]
    clipped = server.clip_diff("\n".join(long)).splitlines()
    assert len(clipped) == 80
    assert clipped[-1] == "… ещё 2 строк"
    assert "f79.py" not in clipped[-1]
    assert long[78] == clipped[78]


def test_git_branch_uses_the_server_root_and_a_ten_second_timeout(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["timeout"] = kwargs["timeout"]
        seen["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="fatal: nope")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    with pytest.raises(ToolError, match="fatal: nope"):
        server.git_branch()
    assert seen["timeout"] == 10.0
    assert seen["cwd"] == server.REPO_ROOT
    assert seen["args"][0] == "git"


def test_git_branch_detached_reads_a_twelve_char_sha(monkeypatch) -> None:
    answers = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD\n",
        ("rev-parse", "--short=12", "HEAD"): "abcdef123456\n",
    }
    monkeypatch.setattr(server, "_git", lambda args: answers[tuple(args)])
    assert server.git_branch() == "ветка: detached abcdef123456"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_embed_question_journal_args(monkeypatch) -> None:
    captured = _patch_embed(monkeypatch, pipeline, command="assistant_embed")
    pipeline.embed_question("вопрос")
    assert captured["kwargs"]["week"] == 7
    assert captured["kwargs"]["day"] == 31
    assert captured["kwargs"]["journal_extra"] == {"command": "assistant_embed"}
    assert "journal_path" not in captured["kwargs"]


def test_doc_question_answers_from_chunks(tmp_path: Path, monkeypatch, err_text) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    logged: list[dict] = []
    monkeypatch.setattr(pipeline, "log_call", lambda result, messages, **kw: logged.append(kw))
    scores = [9, 9, 9, 9, 9, 9, 0, 0]
    text, complete, tools = _ask(
        monkeypatch, db, "Из каких трёх слоёв состоит репозиторий?", scores
    )
    assert text == "три слоя"
    assert tools == ["git_branch"]
    assert len(_system_calls(complete)) == 1
    config, messages = _system_calls(complete)[0]
    help_path = ROOT / "adventassistant" / "prompts" / "help.md"
    help_text = help_path.read_text(encoding="utf-8").strip()
    assert messages[0]["content"] == help_text
    assert config.system_prompt_path is None
    assert config.params.temperature is None
    assert config.params.max_tokens is None
    assert config.params.format is None
    assert config.model == "ministral-14b-latest"
    user = messages[1]["content"]
    assert "ветка: main" in user
    assert "Вопрос: Из каких трёх слоёв состоит репозиторий?" in user
    assert "незакоммичено" not in user
    assert "изменений нет" not in user
    rerank = [call for call in complete.calls if call[1][0]["role"] == "user"]
    assert len(rerank) == 1
    assert rerank[0][0].model == "ministral-14b-latest"
    assert rerank[0][0].params.temperature == 0
    assert rerank[0][0].params.max_tokens == RERANK_MAX_TOKENS
    assert rerank[0][0].params.format == "json"
    err = err_text.getvalue()
    assert "mcp: git_branch" in err
    assert "mcp: git_diff_stat" not in err
    assert err.count("источник:") == 4
    assert "порог: 6/8" in err
    assert [row["extra"]["command"] for row in logged] == ["assistant_rerank", "assistant_help"]
    assert all(row["week"] == 7 and row["day"] == 31 for row in logged)


def test_branch_question_prints_the_tool_and_still_reranks(
    tmp_path: Path, monkeypatch, err_text
) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    text, complete, tools = _ask(monkeypatch, db, "На какой ветке сейчас репозиторий?", [9] * 8)
    assert text == "ветка: main"
    assert tools == ["git_branch"]
    assert _system_calls(complete) == []
    assert len(complete.calls) == 1
    err = err_text.getvalue()
    assert "mcp: git_branch" in err
    assert "mcp: git_diff_stat" not in err
    assert "источник:" in err
    assert "порог: 8/8" in err


def test_diff_question_prints_the_tool_even_when_chunks_pass(
    tmp_path: Path, monkeypatch, err_text
) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    text, complete, tools = _ask(monkeypatch, db, "Что изменилось в рабочей копии?", [9] * 8)
    assert text == "изменений нет"
    assert tools == ["git_branch", "git_diff_stat"]
    assert _system_calls(complete) == []
    err = err_text.getvalue()
    assert "mcp: git_branch" in err
    assert "mcp: git_diff_stat" in err
    assert "порог: 8/8" in err


def test_borscht_below_threshold_is_exactly_ne_znayu(tmp_path: Path, monkeypatch, err_text) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    text, complete, tools = _ask(monkeypatch, db, "Как сварить борщ?", [0] * 8)
    assert text == "не знаю"
    assert tools == ["git_branch"]
    assert _system_calls(complete) == []
    assert len(complete.calls) == 1
    err = err_text.getvalue()
    assert "порог: 0/8" in err
    assert "источник:" not in err


def test_zero_hits_skip_rerank(tmp_path: Path, monkeypatch, err_text) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    text, complete, tools = _ask(monkeypatch, db, "Как сварить борщ?", [], hits=False)
    assert text == "не знаю"
    assert tools == ["git_branch"]
    assert complete.calls == []
    assert "порог: 0/0" in err_text.getvalue()


def test_rerank_garbage_is_an_error_not_ne_znayu(tmp_path: Path, monkeypatch, err_text) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)

    class _Bad(_Complete):
        def __call__(self, config, messages):
            self.calls.append((config, messages))
            return CallResult(text="это не json", sent_messages=list(messages))

    monkeypatch.setattr(pipeline, "cloud_config", _cloud)
    monkeypatch.setattr(
        pipeline, "embed_question", lambda text: np.array([1, 0, 0, 0], dtype=np.float32)
    )
    _install_tools(monkeypatch)
    complete = _Bad([])
    with pytest.raises(AdventError, match="JSON"):
        pipeline.run_help("Как сварить борщ?", db_path=db, complete=complete)
    assert _system_calls(complete) == []
    assert "не знаю" not in err_text.getvalue()


def test_tool_error_skips_embed_and_complete(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    _install_tools(monkeypatch, fail_branch=True)
    monkeypatch.setattr(
        pipeline, "embed_question", lambda text: (_ for _ in ()).throw(AssertionError("embed"))
    )

    def complete(config, messages):
        raise AssertionError("complete")

    with pytest.raises(AdventError, match="git упал"):
        pipeline.run_help("Как сварить борщ?", db_path=db, complete=complete)


def test_tool_exception_skips_embed(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)

    def boom(*args, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(pipeline, "call_tool_once", boom)
    monkeypatch.setattr(
        pipeline, "embed_question", lambda text: (_ for _ in ()).throw(AssertionError("embed"))
    )
    with pytest.raises(AdventError, match="spawn failed"):
        pipeline.run_help("вопрос", db_path=db, complete=lambda *a, **k: None)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_repl_help_without_text_does_not_run_the_pipeline(monkeypatch, err_text, capsys) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("pipeline")

    monkeypatch.setattr(cli, "run_help", boom)
    assert cli.handle_line("/help") == "continue"
    assert cli.handle_line("/help   ") == "continue"
    assert cli.handle_line("/helpful") == "continue"
    assert cli.handle_line("просто текст") == "continue"
    assert cli.handle_line("/exit") == "exit"
    assert cli.handle_line("/quit") == "exit"
    err = err_text.getvalue()
    assert "Как звать" in err
    assert "Неизвестная команда" in err
    assert capsys.readouterr().out == ""


def test_repl_help_passes_complete(monkeypatch, capsys) -> None:
    seen: dict = {}

    def fake(question, *, db_path, complete):
        seen["question"] = question
        seen["complete"] = complete
        return "ответ"

    monkeypatch.setattr(cli, "run_help", fake)
    assert cli.handle_line("/help сколько слоёв") == "continue"
    assert seen["question"] == "сколько слоёв"
    assert seen["complete"] is cli.chat_core.complete
    assert capsys.readouterr().out == "ответ\n"


def test_cli_tool_error_exits_nonzero(tmp_path: Path, monkeypatch, err_text) -> None:
    db = tmp_path / "index.sqlite3"
    _write_docs_index(db)
    _install_tools(monkeypatch, fail_branch=True)
    monkeypatch.setattr(
        pipeline, "embed_question", lambda text: (_ for _ in ()).throw(AssertionError("embed"))
    )
    monkeypatch.setattr(sys, "argv", ["adventassistant", "help", "борщ", "--db", str(db)])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code not in (0, None)
    err = err_text.getvalue()
    assert "git упал" in err
    assert "Traceback" not in err


def test_empty_question_exits_nonzero(monkeypatch, err_text) -> None:
    monkeypatch.setattr(sys, "argv", ["adventassistant", "help", "   "])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code not in (0, None)
    assert "Пустой вопрос" in err_text.getvalue()
    assert "Traceback" not in err_text.getvalue()


def test_help_flag_is_not_translated_into_an_advent_error(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["adventassistant", "--help"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0


def _run_module(args: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "adventassistant.cli", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=False,
    )


def test_missing_index_exits_nonzero_without_a_traceback(tmp_path: Path) -> None:
    proc = _run_module(["help", "вопрос", "--db", str(tmp_path / "missing.sqlite3")])
    assert proc.returncode != 0
    # Rich wraps a narrow stderr; the command name is still in the message.
    assert "adventassistant index" in " ".join(proc.stderr.split())
    assert "Traceback" not in proc.stderr


def test_repl_subprocess_help_does_not_run_the_pipeline() -> None:
    proc = _run_module([], stdin="/help\n/exit\n")
    assert proc.returncode == 0
    assert "Как звать" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Индекс не найден" not in proc.stderr


def test_server_module_stays_up() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "adventassistant.server"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.5)
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise AssertionError(err)
    finally:
        proc.kill()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------
# Docs and the recording scenario
# --------------------------------------------------------------------------


def test_docs_name_the_layers_and_the_commands() -> None:
    structure = (ROOT / "docs" / "structure.md").read_text(encoding="utf-8")
    for needle in (
        "advent_core/",
        "week_NN/",
        "advent_cli/",
        "adventassistant/",
        "advent_core/schemas/",
        "week_01/schemas/",
    ):
        assert needle in structure
    commands = (ROOT / "docs" / "commands.md").read_text(encoding="utf-8")
    tokens = [part for part in commands.split("`")]
    for name in (
        "advent",
        "adventagent",
        "adventmcp",
        "adventrag",
        "adventlocal",
        "adventassistant",
    ):
        assert name in tokens
    assert "Команда `/help` ассистента читает этот корпус и текущую ветку." in commands
    blob = structure + "\n" + commands
    # The drive prefix is split so this test file itself does not contain it:
    # check_staged.py scans the index for that exact string.
    drive = "D:" + "\\"
    assert drive not in blob
    assert "yandex" not in blob.lower()
    assert "yadi.sk" not in blob
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Структура")
    end = readme.index("\n## ", start + 3)
    section = readme[start:end]
    for needle in ("advent_core/", "week_NN/", "advent_cli/", "docs/structure.md"):
        assert needle in section


def test_demo_steps_day_31_and_prepare(tmp_path: Path, monkeypatch) -> None:
    from advent_cli import record

    steps = record.demo_steps(7, 31)
    questions = [
        "На какой ветке сейчас репозиторий?",
        "Что изменилось в рабочей копии?",
        "Как сварить борщ?",
        "Из каких трёх слоёв состоит репозиторий?",
    ]
    assert len(steps) == 4
    for step, question in zip(steps, questions, strict=True):
        assert step.module == "adventassistant.cli"
        assert step.args == ["help", question]
    assert [step.line_pause for step in steps] == [None, None, None, 8.0]
    with pytest.raises(AdventError):
        record.demo_steps(7, 32)
    assert any("models" in step.args for step in record.demo_steps(1, 1))

    rehearsal = record.rehearsal_steps(7, 31)
    assert len(rehearsal) == 1
    assert rehearsal[0].module == "adventassistant.cli"
    assert rehearsal[0].expect_failure is True
    assert "Как сварить борщ?" in rehearsal[0].args
    earlier = record.rehearsal_steps(1, 1)
    assert earlier[0].args[:2] == ["w01", "chat"]

    record._prepare_w07d31(7, 30)
    record._prepare_w07d31(1, 1)
    monkeypatch.setattr(corpus, "INDEX_PATH", tmp_path / "missing.sqlite3")
    with pytest.raises(AdventError, match="adventassistant index"):
        record._prepare_w07d31(7, 31)
    db = tmp_path / "ok.sqlite3"
    _write_docs_index(db)
    monkeypatch.setattr(corpus, "INDEX_PATH", db)
    record._prepare_w07d31(7, 31)
