"""Week 07 day 32: PR review (SPEC-w07d32.md §8). No network, no real journal."""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from advent_core import console, journal
from advent_core.config import Config
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult
from advent_core.errors import AdventError
from advent_core.telemetry import CallResult
from adventassistant import cli, corpus, review, review_index
from adventassistant.review import DemoFacts, check_demo_range, run_review
from adventassistant.review_index import (
    WEEK,
    build_chunks,
    build_review_index,
    collect_review,
)
from adventassistant.server import TOOL_NAMES
from week_05.chunking import Document
from week_05.index import ENDPOINT_CLOUD, ENDPOINT_LOCAL, load_meta, write_index

ROOT = Path(__file__).resolve().parents[1]
DAY = 32


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
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
    _write(repo, "adventassistant/a.py", "def ping():\n    return 1\n")
    _write(repo, "advent_cli/b.py", "def cli():\n    return 1\n")
    _write(repo, "week_01/c.py", "def week():\n    return 1\n")
    _write(repo, "tests/test_a.py", "def test_a():\n    pass\n")
    sha = _commit(repo)
    _write(repo, "README.md", "# Readme\n\nDIRTY working tree.\n")
    _write(repo, "notes.md", "UNTRACKED\n")
    return repo, sha


def _pair(tmp_path: Path, *, second: str = "v2\n") -> tuple[Path, str, str]:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "v1\n")
    base = _commit(repo)
    _write(repo, "README.md", second)
    head = _commit(repo)
    return repo, base, head


def _cloud(model: str) -> Config:
    return Config(
        api_key="k" * 32,
        model=model,
        stream=False,
        system_prompt_path=Path("should-be-cleared.md"),
    )


def _patch_embed(monkeypatch, module) -> list[dict]:
    captured: list[dict] = []

    def fake(client, model, texts, **kwargs):
        captured.append({"kwargs": kwargs, "texts": list(texts), "n": len(texts)})
        matrix = np.zeros((len(texts), 4), dtype=np.float32)
        matrix[:, 0] = 1.0
        return EmbedResult(
            vectors=matrix,
            model=model,
            prompt_tokens=len(texts) * 10,
            requests=1,
            latency_ms=1,
        )

    monkeypatch.setattr(module, "embed_texts_fn", fake)
    monkeypatch.setattr(module, "cloud_config", _cloud)
    monkeypatch.setattr(module, "mistral_client", lambda config: _Ctx())
    return captured


class _Ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc: object) -> bool:
        return False


def _boom(*_args, **_kwargs):
    raise AssertionError("paid call")


@pytest.fixture
def err_text(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(console, "err", Console(file=buf, width=200, no_color=True))
    return buf


def _chunk(i: int, *, source: str | None = None):
    from week_05.chunking import Chunk

    source = source or f"docs/f{i}.md"
    return Chunk(
        chunk_id=f"review:{source}#0",
        strategy="review",
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


def _write_review_index(
    path: Path,
    *,
    corpus_rev: str,
    n: int = 8,
    purpose: str = "w07d32",
    endpoint: str = ENDPOINT_CLOUD,
    doc_prefix: str = "",
    query_prefix: str = "",
    model: str = DEFAULT_EMBED_MODEL,
) -> None:
    doc_count = (n + 1) // 2
    chunks = [
        _chunk(
            i,
            source=(f"docs/f{i}.md" if i < doc_count else f"adventassistant/f{i}.py"),
        )
        for i in range(n)
    ]
    vectors = np.zeros((n, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    embedded = EmbedResult(vectors=vectors, model=model, prompt_tokens=n, requests=1, latency_ms=1)
    write_index(
        path,
        {"review": (chunks, embedded)},
        corpus_rev=corpus_rev,
        corpus_files=n,
        corpus_chars=n,
        endpoint=endpoint,
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
        meta={"purpose": purpose},
    )


class _Complete:
    def __init__(self, payload: dict | None = None, *, scores: list[int] | None = None) -> None:
        self.payload = payload or {"bugs": [], "architecture": [], "recommendations": []}
        self.scores = scores if scores is not None else [0] * 8
        self.calls: list[tuple[object, list]] = []
        self.format_ok = True

    def __call__(self, config, messages):
        self.calls.append((config, messages))
        if messages[0]["role"] == "system":
            return CallResult(
                text=json.dumps(self.payload, ensure_ascii=False),
                format_ok=self.format_ok,
                sent_messages=list(messages),
            )
        scores = [{"id": i, "score": self.scores[i - 1]} for i in range(1, len(self.scores) + 1)]
        return CallResult(
            text=json.dumps({"scores": scores}),
            format_ok=True,
            sent_messages=list(messages),
        )


def _system_calls(complete: _Complete) -> list:
    return [call for call in complete.calls if call[1][0]["role"] == "system"]


def _install_gh(monkeypatch, *, code: int = 0) -> list:
    real = subprocess.run
    calls: list = []

    def fake(args, **kwargs):
        if args and args[0] == "gh":
            body = ""
            if "--body-file" in args:
                body = Path(args[args.index("--body-file") + 1]).read_text(encoding="utf-8")
            calls.append((list(args), body))
            return subprocess.CompletedProcess(args, code, "", "gh failed")
        return real(args, **kwargs)

    monkeypatch.setattr(review.subprocess, "run", fake)
    return calls


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


def test_collect_review_keeps_the_allowlist(tmp_path: Path) -> None:
    repo, sha = _seed_repo(tmp_path)
    full, docs = collect_review(repo)
    assert full == sha
    sources = [doc.source for doc in docs]
    for included in (
        "README.md",
        "docs/a.md",
        "advent_core/schemas/demo.json",
        "adventassistant/a.py",
        "advent_cli/b.py",
        "week_01/c.py",
    ):
        assert included in sources
    assert "tests/test_a.py" not in sources
    assert "notes.md" not in sources
    readme = next(doc for doc in docs if doc.source == "README.md")
    assert "COMMITTED" in readme.text
    assert "DIRTY" not in readme.text
    assert not review_index._is_included("adventassistant/__pycache__/a.py")
    assert not review_index._is_included("tests/test_a.py")


def test_missing_python_on_the_commit_is_an_error(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "# Readme\n")
    _write(repo, "docs/a.md", "# A\n")
    _write(repo, "advent_core/schemas/demo.json", "{}\n")
    _commit(repo)
    with pytest.raises(AdventError, match=r"\.py"):
        collect_review(repo)


def test_python_chunks_are_relabelled_review(tmp_path: Path) -> None:
    repo, _sha = _seed_repo(tmp_path)
    _sha, docs = collect_review(repo)
    chunks = build_chunks(docs)
    python = [chunk for chunk in chunks if chunk.source.endswith(".py")]
    assert python
    assert all(chunk.chunk_id.startswith("review:") for chunk in python)
    assert all(chunk.strategy == "review" for chunk in chunks)
    assert all(not chunk.chunk_id.startswith("structure:") for chunk in chunks)


def test_unparsed_python_longer_than_the_cap_is_split() -> None:
    docs = [
        Document(source="README.md", text="# Readme\n"),
        Document(source="docs/a.md", text="# A\n\nТекст.\n"),
        Document(source="advent_core/schemas/demo.json", text="{}\n"),
        Document(source="adventassistant/bad.py", text="def =\n" + ("x\n" * 5000)),
    ]
    bad = [chunk for chunk in build_chunks(docs) if chunk.source.endswith("bad.py")]
    assert len(bad) > 1
    assert all("(unparsed)" in chunk.section for chunk in bad)
    assert all(chunk.chunk_id.startswith("review:") for chunk in bad)


def test_index_review_refuses_foreign_targets_and_rewrites_its_own(
    tmp_path: Path, monkeypatch
) -> None:
    captured = _patch_embed(monkeypatch, review_index)
    before = corpus.INDEX_PATH.read_bytes() if corpus.INDEX_PATH.exists() else None
    with pytest.raises(AdventError, match="дня 31"):
        build_review_index(tmp_path, db_path=corpus.INDEX_PATH)
    after = corpus.INDEX_PATH.read_bytes() if corpus.INDEX_PATH.exists() else None
    assert before == after

    foreign = tmp_path / "foreign.sqlite3"
    _write_review_index(foreign, corpus_rev="b" * 40, purpose="w07d31", n=1)
    with pytest.raises(AdventError, match="w07d31"):
        build_review_index(tmp_path, db_path=foreign)
    assert load_meta(foreign)["purpose"] == "w07d31"
    assert captured == []

    repo, sha = _seed_repo(tmp_path / "own")
    db = tmp_path / "own.sqlite3"
    first = build_review_index(repo, db_path=db)
    second = build_review_index(repo, db_path=db)
    assert first.splitlines()[0] == sha[:12]
    assert second.splitlines()[0] == sha[:12]
    assert "файлов:" in first
    assert "чанков:" in first
    assert "токенов:" in first
    assert "стоимость:" in first
    assert load_meta(db)["purpose"] == "w07d32"
    assert len(captured) == 2
    for call in captured:
        assert call["kwargs"]["week"] == WEEK
        assert call["kwargs"]["day"] == DAY
        assert call["kwargs"]["journal_extra"] == {"command": "review_index"}
        assert "journal_path" not in call["kwargs"]


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------


def test_empty_range_prints_no_changes_and_calls_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "v1\n")
    _commit(repo)
    gh = _install_gh(monkeypatch)
    monkeypatch.setattr(review, "embed_query", _boom)
    text = run_review(
        repo,
        base="HEAD",
        head="HEAD",
        db_path=tmp_path / "missing.sqlite3",
        complete=_boom,
    )
    assert text == "изменений нет"
    assert capsys.readouterr().out == "изменений нет\n"
    assert gh == []
    assert not (tmp_path / "missing.sqlite3").exists()


def test_diff_only_prints_the_diff_and_does_not_open_the_index(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    repo, base, head = _pair(tmp_path, second="CHANGED_LINE\n")
    monkeypatch.setattr(review_index, "load_meta", _boom)
    monkeypatch.setattr(review, "embed_query", _boom)
    text = run_review(
        repo,
        base=base,
        head=head,
        db_path=tmp_path / "missing.sqlite3",
        diff_only=True,
        complete=_boom,
    )
    assert "CHANGED_LINE" in text
    assert capsys.readouterr().out == text if text.endswith("\n") else text + "\n"
    assert not (tmp_path / "missing.sqlite3").exists()


def test_long_diff_is_capped_but_the_later_name_stays_in_the_prompt(
    tmp_path: Path, monkeypatch, err_text
) -> None:
    assert review.DIFF_CHAR_CAP == 24000
    assert review.QUERY_CHAR_CAP == 6000
    repo = _init_repo(tmp_path)
    _write(repo, "README.md", "v1\n")
    base = _commit(repo)
    _write(repo, "aaa_pad.py", "x = " + ("a" * 40000) + "\n")
    _write(repo, "zzz_late.py", "y = 1\n")
    head = _commit(repo)
    real_embed = review.embed_query
    monkeypatch.setattr(review, "embed_query", _boom)
    diff_only = run_review(
        repo,
        base=base,
        head=head,
        db_path=tmp_path / "missing.sqlite3",
        diff_only=True,
        complete=_boom,
    )
    assert "показано 24000 из " in diff_only
    assert "zzz_late.py" not in diff_only
    assert "… diff обрезан: показано 24000 из " in err_text.getvalue()

    monkeypatch.setattr(review, "embed_query", real_embed)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    captured = _patch_embed(monkeypatch, review)
    complete = _Complete(scores=[0])
    run_review(repo, base=base, head=head, db_path=db, complete=complete)
    user = _system_calls(complete)[0][1][1]["content"]
    files_part, _, diff_part = user.partition("\n\nDiff:\n\n")
    assert "zzz_late.py" in files_part
    assert "zzz_late.py" not in diff_part
    assert "показано 24000 из " in diff_part
    assert len(captured[0]["texts"]) == 1
    assert captured[0]["texts"][0].startswith("ревью изменений")
    assert len(captured[0]["texts"][0]) == review.QUERY_CHAR_CAP


def test_threshold_zero_still_answers_once(tmp_path: Path, monkeypatch, err_text) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=8)
    captured = _patch_embed(monkeypatch, review)
    complete = _Complete(scores=[0] * 8)
    text = run_review(repo, base=base, head=head, db_path=db, complete=complete)
    system = _system_calls(complete)
    assert len(system) == 1
    assert len(complete.calls) == 2
    err = err_text.getvalue()
    assert "порог: 0/8" in err
    assert "источник:" not in err
    assert "Потенциальные баги" in text
    assert "- нет" in text
    answer = system[0][0]
    assert answer.model == "ministral-14b-latest"
    assert answer.system_prompt_path is None
    assert answer.params.temperature == 0
    assert answer.params.max_tokens == 1200
    assert answer.params.format == "schema"
    assert str(answer.params.schema_file).endswith("review.json")
    assert captured[0]["kwargs"]["week"] == WEEK
    assert captured[0]["kwargs"]["day"] == DAY
    assert captured[0]["kwargs"]["journal_extra"] == {"command": "review_embed"}
    assert "journal_path" not in captured[0]["kwargs"]
    rows = [
        json.loads(line)
        for line in (journal.LOG_DIR / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["command"] for row in rows] == ["review_rerank", "review_answer"]
    assert all(row["week"] == 7 and row["day"] == 32 for row in rows)


def test_sixteen_candidates_all_reach_rerank_and_the_answer_keeps_four(
    tmp_path: Path, monkeypatch, err_text
) -> None:
    assert review.SEARCH_K == 16
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=16)
    _patch_embed(monkeypatch, review)
    complete = _Complete(scores=[10] * 6 + [0] * 10)
    run_review(repo, base=base, head=head, db_path=db, complete=complete)
    rerank_prompt = complete.calls[0][1][0]["content"]
    assert complete.calls[0][1][0]["role"] == "user"
    assert "для всех 16 фрагментов" in rerank_prompt
    assert rerank_prompt.count("docs/f") == review.DOC_SEARCH_K
    assert rerank_prompt.count("adventassistant/f") == review.SEARCH_K - review.DOC_SEARCH_K
    for number in range(1, 17):
        assert f"[{number}] " in rerank_prompt
    answer_user = _system_calls(complete)[0][1][1]["content"]
    assert answer_user.count("— раздел") == review.ANSWER_K
    assert "[5] " not in answer_user
    err = err_text.getvalue()
    assert "порог: 6/16" in err
    assert err.count("источник:") == 4


def test_render_prints_three_headers_empty_and_outside(
    tmp_path: Path, monkeypatch, err_text
) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    _patch_embed(monkeypatch, review)
    complete = _Complete(
        {
            "bugs": [],
            "architecture": [{"file": "nope.py", "text": "секрет"}],
            "recommendations": [{"text": "поправь границу"}],
        },
        scores=[10],
    )
    text = run_review(repo, base=base, head=head, db_path=db, complete=complete)
    assert text.splitlines()[:6] == [
        "Потенциальные баги",
        "- нет",
        "Архитектурные проблемы",
        "- вне diff: nope.py: секрет",
        "Рекомендации",
        "- поправь границу",
    ]
    assert "вне diff: 1" in err_text.getvalue()


def test_format_ok_false_exits_nonzero_without_a_comment(
    tmp_path: Path, monkeypatch, capsys, err_text
) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    _patch_embed(monkeypatch, review)
    gh = _install_gh(monkeypatch)
    complete = _Complete(scores=[10])
    complete.format_ok = False
    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    monkeypatch.setattr(cli.chat_core, "complete", complete)
    monkeypatch.setenv("GH_TOKEN", "review-test-token")
    monkeypatch.setattr(
        sys,
        "argv",
        ["adventassistant", "review", "--base", base, "--head", head, "--db", str(db), "--pr", "4"],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code not in (0, None)
    assert "Потенциальные баги" not in capsys.readouterr().out
    assert "Traceback" not in err_text.getvalue()
    assert gh == []


def test_day31_index_is_refused_before_embed(tmp_path: Path, monkeypatch) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "day31.sqlite3"
    _write_review_index(db, corpus_rev=base, purpose="w07d31", n=1)
    monkeypatch.setattr(review, "embed_query", _boom)
    with pytest.raises(AdventError, match="adventassistant index-review"):
        run_review(repo, base=base, head=head, db_path=db, complete=_boom)


def test_corpus_rev_mismatch_is_refused_before_embed(tmp_path: Path, monkeypatch) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev="c" * 40, n=1)
    monkeypatch.setattr(review, "embed_query", _boom)
    with pytest.raises(AdventError, match="adventassistant index-review"):
        run_review(repo, base=base, head=head, db_path=db, complete=_boom)


def test_review_refuses_non_cloud_endpoint_and_prefixes(tmp_path: Path, monkeypatch) -> None:
    repo, base, head = _pair(tmp_path)
    monkeypatch.setattr(review, "embed_query", _boom)
    local = tmp_path / "local.sqlite3"
    _write_review_index(local, corpus_rev=base, endpoint=ENDPOINT_LOCAL, n=1)
    with pytest.raises(AdventError, match="endpoint"):
        run_review(repo, base=base, head=head, db_path=local, complete=_boom)
    prefixed = tmp_path / "prefix.sqlite3"
    _write_review_index(prefixed, corpus_rev=base, doc_prefix="search_document: ", n=1)
    with pytest.raises(AdventError, match="doc_prefix"):
        run_review(repo, base=base, head=head, db_path=prefixed, complete=_boom)
    queried = tmp_path / "query.sqlite3"
    _write_review_index(queried, corpus_rev=base, query_prefix="search_query: ", n=1)
    with pytest.raises(AdventError, match="query_prefix"):
        run_review(repo, base=base, head=head, db_path=queried, complete=_boom)


def test_pr_without_a_token_fails_before_embed(tmp_path: Path, monkeypatch) -> None:
    repo, base, head = _pair(tmp_path)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(review, "load_env", lambda: None)
    monkeypatch.setattr(review, "embed_query", _boom)
    monkeypatch.setattr(review_index, "load_meta", _boom)
    with pytest.raises(AdventError, match="GH_TOKEN"):
        run_review(
            repo,
            base=base,
            head=head,
            db_path=tmp_path / "missing.sqlite3",
            pr=3,
            complete=_boom,
        )


def _clear_tokens(monkeypatch) -> None:
    # setenv first so monkeypatch restores "absent" after load_dotenv writes os.environ.
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)


def test_require_token_reads_project_dotenv(tmp_path: Path, monkeypatch) -> None:
    from advent_core import config as core_config

    (tmp_path / ".env").write_text("GITHUB_TOKEN=dotenv-test-token\n", encoding="utf-8")
    _clear_tokens(monkeypatch)
    monkeypatch.setattr(core_config, "PROJECT_ROOT", tmp_path)
    review._require_token()
    assert review._has_token()


def test_require_token_keeps_existing_env_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    from advent_core import config as core_config

    (tmp_path / ".env").write_text("GH_TOKEN=from-dotenv\n", encoding="utf-8")
    _clear_tokens(monkeypatch)
    monkeypatch.setenv("GH_TOKEN", "from-action")
    monkeypatch.setattr(core_config, "PROJECT_ROOT", tmp_path)
    review._require_token()
    assert review.os.environ["GH_TOKEN"] == "from-action"


def test_require_token_without_env_or_dotenv_still_refuses(tmp_path: Path, monkeypatch) -> None:
    from advent_core import config as core_config

    _clear_tokens(monkeypatch)
    monkeypatch.setattr(core_config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(AdventError, match="GH_TOKEN"):
        review._require_token()


def test_empty_range_and_diff_only_do_not_load_env(tmp_path: Path, monkeypatch) -> None:
    repo, base, head = _pair(tmp_path)
    monkeypatch.setattr(review, "load_env", _boom)
    monkeypatch.setattr(review, "embed_query", _boom)
    db = tmp_path / "missing.sqlite3"
    run_review(repo, base=base, head=base, db_path=db, pr=3, complete=_boom)
    run_review(repo, base=base, head=head, db_path=db, pr=3, diff_only=True, complete=_boom)


def test_cyrillic_paths_are_collected_with_quotepath_on(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    subprocess.run(["git", "config", "core.quotePath", "true"], cwd=repo, check=True)
    _write(repo, "README.md", "# Readme\n")
    _write(repo, "docs/документ.md", "# Документ\n\nАбзац.\n")
    _write(repo, "advent_core/schemas/demo.json", '{"type": "object"}\n')
    _write(repo, "adventassistant/модуль.py", "def ping():\n    return 1\n")
    sha = _commit(repo)
    resolved, docs = review_index.collect_review(repo, sha)
    sources = {doc.source for doc in docs}
    assert resolved == sha
    assert "docs/документ.md" in sources
    assert "adventassistant/модуль.py" in sources
    assert not any(source.startswith('"') or "\\3" in source for source in sources)


def test_pr_comment_uses_body_file_and_no_token_url(tmp_path: Path, monkeypatch, capsys) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    _patch_embed(monkeypatch, review)
    gh = _install_gh(monkeypatch)
    monkeypatch.setenv("GH_TOKEN", "review-test-token")
    complete = _Complete(scores=[10])
    text = run_review(repo, base=base, head=head, db_path=db, pr=12, complete=complete)
    assert len(gh) == 1
    args, body = gh[0]
    assert args[:4] == ["gh", "pr", "comment", "12"]
    assert args[4] == "--body-file"
    assert len(args) == 6
    blob = " ".join(args)
    assert "https://" not in blob
    assert "review-test-token" not in blob
    assert "<!-- adventassistant-review -->" in body
    assert "Потенциальные баги" in body
    assert "Архитектурные проблемы" in body
    assert "Рекомендации" in body
    assert f"ревью: {base[:12]}..{head[:12]}" in body
    assert "Потенциальные баги" in capsys.readouterr().out
    assert text.startswith("Потенциальные баги")


def test_gh_failure_keeps_stdout_and_exits_nonzero(
    tmp_path: Path, monkeypatch, capsys, err_text
) -> None:
    repo, base, head = _pair(tmp_path)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    _patch_embed(monkeypatch, review)
    _install_gh(monkeypatch, code=1)
    complete = _Complete(scores=[10])
    monkeypatch.setattr(cli, "PROJECT_ROOT", repo)
    monkeypatch.setattr(cli.chat_core, "complete", complete)
    monkeypatch.setenv("GH_TOKEN", "review-test-token")
    monkeypatch.setattr(
        sys,
        "argv",
        ["adventassistant", "review", "--base", base, "--head", head, "--db", str(db), "--pr", "5"],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code not in (0, None)
    assert "Потенциальные баги" in capsys.readouterr().out
    assert "Traceback" not in err_text.getvalue()


def test_rename_and_a_spaced_path_come_through_nul(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "old name.py", "def a():\n    return 1\n")
    _write(repo, "spaced file.py", "x = 1\n")
    base = _commit(repo)
    subprocess.run(["git", "mv", "old name.py", "new name.py"], cwd=repo, check=True)
    _write(repo, "spaced file.py", "x = 2\n")
    head = _commit(repo)
    db = tmp_path / "index.sqlite3"
    _write_review_index(db, corpus_rev=base, n=1)
    _patch_embed(monkeypatch, review)
    complete = _Complete(
        {
            "bugs": [
                {"file": "old name.py", "text": "старый путь"},
                {"file": "new name.py", "text": "новый путь"},
                {"file": "nope.py", "text": "чужой"},
            ],
            "architecture": [],
            "recommendations": [],
        },
        scores=[10],
    )
    text = run_review(repo, base=base, head=head, db_path=db, complete=complete)
    user = _system_calls(complete)[0][1][1]["content"]
    assert "old name.py -> new name.py" in user
    assert "spaced file.py" in user
    for line in text.splitlines():
        if "old name.py" in line or "new name.py" in line:
            assert "вне diff" not in line
        if "nope.py" in line:
            assert line.startswith("- вне diff:")


def test_run_review_complete_has_no_default_and_cli_passes_it(monkeypatch, capsys) -> None:
    assert inspect.signature(run_review).parameters["complete"].default is inspect.Parameter.empty
    seen: dict = {}

    def fake(root, **kwargs):
        seen["complete"] = kwargs["complete"]
        return "изменений нет"

    monkeypatch.setattr(cli, "run_review", fake)
    monkeypatch.setattr(sys, "argv", ["adventassistant", "review", "--base", "a", "--head", "b"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code in (0, None)
    assert seen["complete"] is cli.chat_core.complete
    assert capsys.readouterr().out == ""


def test_day31_surface_is_unchanged() -> None:
    assert TOOL_NAMES == ("git_branch", "git_diff_stat")
    assert corpus.PURPOSE == "w07d31"
    assert corpus.STRATEGY == "docs"
    assert corpus.DAY == 31
    assert corpus.INDEX_PATH.name == "index.w07d31.sqlite3"


# --------------------------------------------------------------------------
# Docs, recording
# --------------------------------------------------------------------------


def test_docs_name_review_and_the_token_rule() -> None:
    structure = (ROOT / "docs" / "structure.md").read_text(encoding="utf-8")
    assert "index-review" in structure
    assert "advent_core/" in structure
    assert "advent_cli/" in structure
    assert "adventassistant/" in structure
    assert "week_*/" in structure
    assert "tests/" in structure
    commands = (ROOT / "docs" / "commands.md").read_text(encoding="utf-8")
    assert "index-review" in commands
    assert "`review`" in commands or "review" in commands
    assert "Токен не помещается в URL, который передаётся git." in commands
    blob = structure + "\n" + commands
    assert "D:" + "\\" not in blob
    assert "yandex" not in blob.lower()
    assert "yadi.sk" not in blob
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Команда `review` разбирает диапазон `base...head`." in readme


def test_demo_steps_day_32_and_the_pure_range_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    steps = record.demo_steps(7, 32)
    assert len(steps) == 3
    assert steps[0].module == "adventassistant.cli"
    assert steps[0].args == ["review", "--base", "main", "--head", "HEAD", "--diff-only"]
    assert all(step.module == "adventassistant.cli" for step in steps)
    assert steps[2].line_pause == 8.0
    assert (steps[0].line_pause or record.STEP_PAUSE) < steps[2].line_pause
    with pytest.raises(AdventError):
        record.demo_steps(7, 33)

    rehearsal = record.rehearsal_steps(7, 32)
    assert len(rehearsal) == 1
    step = rehearsal[0]
    assert step.module == "adventassistant.cli"
    assert step.expect_failure is True
    assert step.args == ["review", "--diff-only", "--base", "нет-такой-базы", "--head", "HEAD"]
    assert step.args[:2] != ["w07", "chat"]

    sha = "a" * 40
    ok = DemoFacts(
        branch="w07d32-demo",
        name_only_z="adventassistant/demo_bug.py\0",
        target_diff="",
        purpose="w07d32",
        corpus_rev=sha,
        base_sha=sha,
    )
    check_demo_range(ok)
    with pytest.raises(AdventError):
        check_demo_range(replace(ok, branch="main"))
    with pytest.raises(AdventError):
        check_demo_range(replace(ok, name_only_z="adventassistant/demo_bug.py\0README.md\0"))
    with pytest.raises(AdventError):
        check_demo_range(replace(ok, corpus_rev="b" * 40))
    with pytest.raises(AdventError):
        check_demo_range(replace(ok, target_diff="diff --git a/adventassistant/demo_bug.py\n"))


_BASE = "a" * 40
_HEAD = "b" * 40


def _boom(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("must not be reached")


class _FakeGithub:
    def __init__(self, pull: object = None, error: Exception | None = None) -> None:
        self.pull = pull
        self.error = error
        self.asked: list[int] = []
        self.closed = 0

    def get_pull(self, number: int) -> object:
        self.asked.append(number)
        if self.error is not None:
            raise self.error
        return self.pull

    def close(self) -> None:
        self.closed += 1


def _pull(**changes: object) -> object:
    from adventassistant.github import PullRequest

    pull = PullRequest(
        number=41,
        state="open",
        base_sha=_BASE,
        head_sha=_HEAD,
        base_repo="owner/repo",
        head_repo="owner/repo",
    )
    return replace(pull, **changes)


def _stub_prepare(monkeypatch: pytest.MonkeyPatch, fake: _FakeGithub) -> None:
    """Git facts and the GitHub client are doubles: no repo, no index, no network."""
    from adventassistant import github

    facts = DemoFacts(
        branch="w07d32-demo",
        name_only_z="adventassistant/demo_bug.py\0",
        target_diff="",
        purpose="w07d32",
        corpus_rev=_BASE,
        base_sha=_BASE,
    )
    monkeypatch.setattr(review, "collect_demo_facts", lambda root: facts)
    monkeypatch.setattr(review, "resolve_rev", lambda root, rev: _HEAD)
    monkeypatch.setattr(github.GithubClient, "from_project", lambda root: fake)


@pytest.mark.parametrize("raw", [None, "", "  ", "0", "-3", "abc", "1.5", "4 1", "٣"])
def test_day_32_demo_needs_a_valid_pr_number(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    from advent_cli import record
    from adventassistant import github

    if raw is None:
        monkeypatch.delenv("ADVENT_DEMO_PR", raising=False)
    else:
        monkeypatch.setenv("ADVENT_DEMO_PR", raw)
    monkeypatch.setattr(review, "collect_demo_facts", _boom)
    monkeypatch.setattr(review, "resolve_rev", _boom)
    monkeypatch.setattr(github.GithubClient, "from_project", _boom)
    with pytest.raises(AdventError, match="ADVENT_DEMO_PR"):
        record.demo_steps(7, 32)
    with pytest.raises(AdventError, match="ADVENT_DEMO_PR"):
        record._prepare_w07d32(7, 32)


def test_day_32_prepare_accepts_the_matching_open_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    fake = _FakeGithub(_pull())
    _stub_prepare(monkeypatch, fake)
    record._prepare_w07d32(7, 32)
    assert fake.asked == [41]
    assert fake.closed == 1
    # Other days never read the environment or the network.
    monkeypatch.delenv("ADVENT_DEMO_PR")
    monkeypatch.setattr(review, "collect_demo_facts", _boom)
    record._prepare_w07d32(7, 31)


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "closed"},
        {"head_repo": "someone/fork"},
        {"head_repo": ""},
        {"base_sha": "c" * 40},
        {"head_sha": "c" * 40},
    ],
    ids=["closed", "fork", "fork-gone", "base-mismatch", "head-mismatch"],
)
def test_day_32_prepare_refuses_a_pr_the_take_cannot_show(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    fake = _FakeGithub(_pull(**changes))
    _stub_prepare(monkeypatch, fake)
    with pytest.raises(AdventError):
        record._prepare_w07d32(7, 32)
    assert fake.closed == 1


def test_day_32_prepare_closes_the_client_when_the_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    fake = _FakeGithub(error=AdventError("GitHub недоступен"))
    _stub_prepare(monkeypatch, fake)
    with pytest.raises(AdventError, match="недоступен"):
        record._prepare_w07d32(7, 32)
    assert fake.closed == 1


def test_day_32_prepare_still_runs_the_pure_gate_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    fake = _FakeGithub(_pull())
    _stub_prepare(monkeypatch, fake)
    bad = DemoFacts(
        branch="main",
        name_only_z="adventassistant/demo_bug.py\0",
        target_diff="",
        purpose="w07d32",
        corpus_rev=_BASE,
        base_sha=_BASE,
    )
    monkeypatch.setattr(review, "collect_demo_facts", lambda root: bad)
    with pytest.raises(AdventError, match="w07d32-demo"):
        record._prepare_w07d32(7, 32)
    assert fake.asked == []


def test_day_32_watcher_steps_share_one_scenario_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from advent_cli import record

    monkeypatch.setenv("ADVENT_DEMO_PR", "41")
    steps = record.demo_steps(7, 32)
    first, second = steps[1], steps[2]
    assert first.args == second.args
    args = first.args
    assert args[:4] == ["watch-pr", "--once", "--pr", "41"]
    assert args[4] == "--state"
    state = args[5]
    assert len(args) == 6
    assert state.startswith("logs/demo/w07d32/")
    assert state.endswith("/watcher.json")
    assert not Path(state).is_absolute()
    assert state != "data/review/watcher.json"
    # --post is the default and --cache-dir is left alone: the shared index cache is reused.
    assert "--no-post" not in args
    assert "--cache-dir" not in args
    for step in (first, second):
        assert "day32" in step.title
        assert "local watcher" in step.title
        assert "live Mistral" in step.title
    assert not (ROOT / state).exists()


def test_day_32_rehearsal_stays_offline_without_a_pr_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from advent_cli import record

    monkeypatch.delenv("ADVENT_DEMO_PR", raising=False)
    steps = record.rehearsal_steps(7, 32)
    assert len(steps) == 1
    assert steps[0].expect_failure is True
    assert steps[0].args == ["review", "--diff-only", "--base", "нет-такой-базы", "--head", "HEAD"]
