"""week_05/cli.py (`adventrag`) — index/compare/search/show, no network.

The network seam is `cli.embed_texts`/`cli.mistral_client`, replaced per test
(the same explicit-seam approach as tests/test_cli_bench.py's `cli.chat_core`
patch): week_05/cli.py calls these as plain module globals, never through a
default-bound parameter, so a plain monkeypatch reaches every call site
(CLAUDE.md's "default argument binds at import time" trap does not apply
here). A couple of scenarios that need the real error path end to end
(missing index, stdout/stderr split) run as a subprocess instead, mirroring
tests/test_week04_cli.py.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

import week_05.cli as cli
from advent_core import config as config_module
from advent_core import console
from advent_core.embeddings import EmbedResult
from advent_core.errors import AdventError
from week_05.chunking import Chunk
from week_05.index import EvalReport, StrategyMetrics, write_index

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "COLUMNS": "200"}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


def run_cli(*args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "week_05.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env=ENV,
        timeout=timeout,
    )


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)


@pytest.fixture
def stdout_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200, no_color=True))
    return buffer


@pytest.fixture
def stderr_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "err", Console(file=buffer, width=200, no_color=True))
    return buffer


@pytest.fixture
def no_client(monkeypatch):
    """`mistral_client` that fails the test the moment it is entered."""

    def _forbidden(config):
        raise AssertionError("mistral_client() must not be called")

    monkeypatch.setattr(cli, "mistral_client", _forbidden)


class _FakeClientCtx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


class _FakeEmbedTexts:
    """Replacement for advent_core.embeddings.embed_texts: deterministic shape, no network."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[dict] = []

    def __call__(
        self,
        client,
        model,
        texts,
        *,
        ids=None,
        on_batch=None,
        week=5,
        day=21,
        journal_path=None,
        journal_extra=None,
    ) -> EmbedResult:
        self.calls.append(
            {"model": model, "texts": list(texts), "ids": ids, "journal_extra": journal_extra}
        )
        rng = np.random.default_rng(len(self.calls))
        matrix = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / norms
        if on_batch is not None:
            on_batch(len(texts), len(texts))
        return EmbedResult(
            vectors=matrix, model=model, prompt_tokens=len(texts) * 10, requests=1, latency_ms=5
        )


@pytest.fixture
def fake_embed(monkeypatch):
    fake = _FakeEmbedTexts()
    monkeypatch.setattr(cli, "mistral_client", lambda config: _FakeClientCtx())
    monkeypatch.setattr(cli, "embed_texts", fake)
    return fake


def _chunk(chunk_id: str, *, strategy: str, source: str = "foo.md", **kw) -> Chunk:
    fields = {
        "title": "foo",
        "section": "(preamble)",
        "ordinal": 0,
        "char_start": 0,
        "char_end": 10,
        "line_start": 1,
        "line_end": 1,
        "text": "hello world",
        **kw,
    }
    return Chunk(chunk_id=chunk_id, strategy=strategy, source=source, **fields)


def _seed_index(db_path: Path, *, strategies=("fixed", "structure"), dim: int = 4, **run_kw):
    """Write a minimal, mutually comparable index for both strategies."""
    rng = np.random.default_rng(0)
    results = {}
    for strat in strategies:
        chunk = _chunk(f"{strat}:foo.md#0", strategy=strat)
        vec = rng.normal(size=(1, dim)).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        results[strat] = (
            [chunk],
            EmbedResult(
                vectors=vec, model="mistral-embed", prompt_tokens=10, requests=1, latency_ms=5
            ),
        )
    write_index(
        db_path,
        results,
        corpus_rev=run_kw.pop("corpus_rev", "HEAD"),
        corpus_files=run_kw.pop("corpus_files", 1),
        corpus_chars=run_kw.pop("corpus_chars", 100),
    )


# --------------------------------------------------------------------------
# index --dry-run
# --------------------------------------------------------------------------


def test_dry_run_never_constructs_the_client(no_client, tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    cli.index_command(
        strategy="fixed", model="mistral-embed", rev="HEAD", db=str(db_path), dry_run=True
    )
    assert not db_path.exists()
    assert "Статистика чанков" in stdout_capture.getvalue()


def test_dry_run_prints_chunk_stats_table_for_every_requested_strategy(
    no_client, tmp_path, stdout_capture
):
    cli.index_command(
        strategy="all",
        model="mistral-embed",
        rev="HEAD",
        db=str(tmp_path / "index.sqlite3"),
        dry_run=True,
    )
    out = _flat(stdout_capture.getvalue())
    assert "fixed" in out
    assert "structure" in out


# --------------------------------------------------------------------------
# index: chunk-size preflight (finding 11)
# --------------------------------------------------------------------------


def test_index_rejects_an_oversized_chunk_before_any_embed_call(no_client, tmp_path, monkeypatch):
    from week_05.chunking import MAX_CHUNK_CHARS

    def oversized(docs, strat):
        return [
            _chunk(
                f"{strat}:x.md#0", strategy=strat, source="x.md", text="x" * (MAX_CHUNK_CHARS + 1)
            )
        ]

    monkeypatch.setattr(cli, "chunk_corpus", oversized)
    db_path = tmp_path / "index.sqlite3"

    with pytest.raises(AdventError, match="fixed:x.md#0"):
        cli.index_command(
            strategy="fixed", model="mistral-embed", rev="HEAD", db=str(db_path), dry_run=False
        )
    assert not db_path.exists()


# --------------------------------------------------------------------------
# index (live): summary line, one write_index call, journal_extra
# --------------------------------------------------------------------------


def test_index_live_embeds_each_strategy_and_writes_one_index(fake_embed, tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    cli.index_command(
        strategy="all", model="mistral-embed", rev="HEAD", db=str(db_path), dry_run=False
    )
    assert db_path.exists()
    assert {c["journal_extra"]["command"] for c in fake_embed.calls} == {"index"}
    assert {c["journal_extra"]["strategy"] for c in fake_embed.calls} == {"fixed", "structure"}
    out = _flat(stdout_capture.getvalue())
    assert "fixed" in out and "structure" in out
    assert "запросов" in out


# --------------------------------------------------------------------------
# missing index: message + non-zero exit (real subprocess, no mocking needed)
# --------------------------------------------------------------------------


def test_missing_index_gives_a_readable_message_and_exits_nonzero(tmp_path):
    missing = tmp_path / "does-not-exist.sqlite3"
    proc = run_cli("show", "foo.md:1", "--db", str(missing))
    assert proc.returncode == 1
    assert "Ошибка: Индекс не найден" in proc.stderr
    assert "adventrag index" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_missing_index_message_is_the_same_for_compare_and_search(tmp_path):
    missing = tmp_path / "does-not-exist.sqlite3"
    for args in (["compare", "--db", str(missing)], ["search", "вопрос", "--db", str(missing)]):
        proc = run_cli(*args)
        assert proc.returncode == 1
        assert "Ошибка: Индекс не найден" in proc.stderr


def test_corrupt_db_gives_a_readable_message_not_a_traceback(tmp_path):
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"this is not a sqlite database")
    proc = run_cli("show", "foo.md:1", "--db", str(garbage))
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr
    assert "Ошибка" in proc.stderr


# --------------------------------------------------------------------------
# stdout/stderr contract
# --------------------------------------------------------------------------


def test_dry_run_progress_goes_to_stderr_table_goes_to_stdout():
    proc = run_cli("index", "--strategy", "fixed", "--dry-run")
    assert proc.returncode == 0
    assert "собираю корпус" in proc.stderr
    assert "chunking" in proc.stderr
    assert "Статистика чанков" in proc.stdout
    assert "собираю корпус" not in proc.stdout
    assert "Статистика чанков" not in proc.stderr


# --------------------------------------------------------------------------
# compare: mismatched runs are refused
# --------------------------------------------------------------------------


def test_compare_refuses_strategies_indexed_with_different_models(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    rng = np.random.default_rng(0)
    for strat, model in (("fixed", "mistral-embed"), ("structure", "other-embed")):
        vec = rng.normal(size=(1, 4)).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        write_index(
            db_path,
            {
                strat: (
                    [_chunk(f"{strat}:foo.md#0", strategy=strat)],
                    EmbedResult(
                        vectors=vec, model=model, prompt_tokens=10, requests=1, latency_ms=5
                    ),
                )
            },
            corpus_rev="HEAD",
            corpus_files=1,
            corpus_chars=100,
        )

    with pytest.raises(AdventError) as excinfo:
        cli.compare_command(db=str(db_path))
    assert "разными моделями" in excinfo.value.message


def test_compare_refuses_when_only_one_strategy_is_indexed(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path, strategies=("fixed",))
    with pytest.raises(AdventError) as excinfo:
        cli.compare_command(db=str(db_path))
    assert "обе стратегии" in excinfo.value.message


# --------------------------------------------------------------------------
# compare: tie wording (CLAUDE.md — never name a winner on a tie)
# --------------------------------------------------------------------------


def test_compare_prints_tie_wording_when_mrr_is_equal(
    fake_embed, tmp_path, stdout_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    tied = StrategyMetrics(n_questions=6, hit_at_1=0.5, hit_at_3=0.5, hit_at_5=0.5, mrr_at_5=0.5)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": tied, "structure": tied}, broken=[]
        ),
    )

    cli.compare_command(db=str(db_path))

    out = _flat(stdout_capture.getvalue())
    assert "Ничья по MRR@5" in out
    assert "Лидер" not in out


def test_compare_names_the_leader_when_mrr_differs(
    fake_embed, tmp_path, stdout_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    winner = StrategyMetrics(n_questions=6, hit_at_1=1.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=0.9)
    loser = StrategyMetrics(n_questions=6, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.1)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": winner, "structure": loser}, broken=[]
        ),
    )

    cli.compare_command(db=str(db_path))

    out = _flat(stdout_capture.getvalue())
    assert "Выше MRR@5 в этой выборке: fixed" in out
    assert "0.900" in out
    assert "0.100" in out
    assert "Ничья" not in out


def test_compare_lists_broken_questions_on_stderr(
    fake_embed, tmp_path, stdout_capture, stderr_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    empty = StrategyMetrics(n_questions=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.0)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": empty, "structure": empty},
            broken=[cli.index_module.BrokenQuestion(id=1, question="q?", reason="якорь не найден")],
        ),
    )

    cli.compare_command(db=str(db_path))

    assert "q?" in stderr_capture.getvalue()
    assert "q?" not in stdout_capture.getvalue()


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_search_embeds_once_and_prints_a_table_per_strategy(fake_embed, tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)

    cli.search_command(query="hello", strategy="all", k=5, db=str(db_path))

    assert len(fake_embed.calls) == 1
    assert fake_embed.calls[0]["texts"] == ["hello"]
    assert fake_embed.calls[0]["journal_extra"]["command"] == "search"
    out = _flat(stdout_capture.getvalue())
    assert "Поиск — fixed" in out
    assert "Поиск — structure" in out
    assert "foo.md" in out


def test_search_single_strategy_only_prints_one_table(fake_embed, tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)

    cli.search_command(query="hello", strategy="fixed", k=3, db=str(db_path))

    out = _flat(stdout_capture.getvalue())
    assert "Поиск — fixed" in out
    assert "Поиск — structure" not in out


# --------------------------------------------------------------------------
# show: chunk_id and SOURCE:LINE
# --------------------------------------------------------------------------


def test_show_by_chunk_id(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)

    cli.show_command(target="fixed:foo.md#0", strategy="fixed", db=str(db_path), full=False)

    out = _flat(stdout_capture.getvalue())
    assert "fixed:foo.md#0" in out
    assert "hello world" in out


def test_show_by_source_and_line(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(1, 4)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    chunk = _chunk(
        "structure:bar.md#0",
        strategy="structure",
        source="bar.md",
        line_start=10,
        line_end=20,
        text="the line-addressed chunk",
    )
    write_index(
        db_path,
        {
            "structure": (
                [chunk],
                EmbedResult(
                    vectors=vec, model="mistral-embed", prompt_tokens=10, requests=1, latency_ms=5
                ),
            )
        },
        corpus_rev="HEAD",
        corpus_files=1,
        corpus_chars=100,
    )

    cli.show_command(target="bar.md:15", strategy="structure", db=str(db_path), full=False)

    out = _flat(stdout_capture.getvalue())
    assert "the line-addressed chunk" in out
    assert "10/20" in out


def test_show_unresolvable_source_line_raises_advent_error(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    with pytest.raises(AdventError, match="Чанк не найден"):
        cli.show_command(target="foo.md:999", strategy="fixed", db=str(db_path), full=False)


def test_show_target_without_hash_or_colon_is_a_local_error_no_db_touched(no_client, tmp_path):
    missing = tmp_path / "never-created.sqlite3"
    with pytest.raises(AdventError, match="ожидается chunk_id"):
        cli.show_command(
            target="no-separator-at-all", strategy="fixed", db=str(missing), full=False
        )
    assert not missing.exists()


# --------------------------------------------------------------------------
# _db_display: repo-relative POSIX, never absolute (finding 14)
# --------------------------------------------------------------------------


def test_db_display_default_db_is_repo_relative_posix(monkeypatch):
    monkeypatch.setattr(
        cli.index_module, "DEFAULT_DB", cli.PROJECT_ROOT / "data" / "rag" / "index.sqlite3"
    )
    assert cli._db_display(None) == "data/rag/index.sqlite3"


def test_db_display_custom_path_inside_repo_is_repo_relative():
    custom = cli.PROJECT_ROOT / "week_05" / "custom.sqlite3"
    assert cli._db_display(custom) == "week_05/custom.sqlite3"


def test_db_display_custom_path_outside_repo_is_just_the_file_name(tmp_path):
    outside = tmp_path / "index.sqlite3"
    assert cli._db_display(outside) == "index.sqlite3"


def test_short_rev_truncates_to_twelve_chars():
    assert cli._short_rev("a" * 40) == "a" * 12


# --------------------------------------------------------------------------
# compare: two compact tables at width 80, no truncation (finding 10, 13)
# --------------------------------------------------------------------------


def test_compare_tables_fit_80_columns_with_no_truncation(fake_embed, tmp_path, monkeypatch):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=80, no_color=True))
    metrics = StrategyMetrics(
        n_questions=6, hit_at_1=0.5, hit_at_3=0.5, hit_at_5=0.5, mrr_at_5=0.333
    )
    other = StrategyMetrics(n_questions=6, hit_at_1=0.6, hit_at_3=0.6, hit_at_5=0.6, mrr_at_5=0.354)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": metrics, "structure": other}, broken=[]
        ),
    )

    cli.compare_command(db=str(db_path))

    out = buffer.getvalue()
    assert "…" not in out
    flat = _flat(out)
    # concrete numeric cells from the seeded 1-chunk-per-strategy index and
    # the injected metrics — proof the tables actually rendered, not just
    # that no truncation marker appeared.
    assert "11" in flat  # n_chars of the seeded chunk's "hello world" text
    assert "10" in flat  # prompt_tokens from _seed_index
    assert "0.333" in flat
    assert "0.354" in flat


# --------------------------------------------------------------------------
# search: two-line layout; show: preview + --full; compare: order
# --------------------------------------------------------------------------


def _seed_long_index(db_path: Path, *, text: str, source: str, section: str) -> None:
    rng = np.random.default_rng(0)
    results = {}
    for strat in ("fixed", "structure"):
        chunk = _chunk(
            f"{strat}:{source}#0",
            strategy=strat,
            source=source,
            section=section,
            line_start=1888,
            line_end=1904,
            text=text,
        )
        vec = rng.normal(size=(1, 4)).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        results[strat] = (
            [chunk],
            EmbedResult(
                vectors=vec, model="mistral-embed", prompt_tokens=10, requests=1, latency_ms=5
            ),
        )
    write_index(db_path, results, corpus_rev="HEAD", corpus_files=1, corpus_chars=100)


def test_search_prints_two_lines_per_hit_at_width_80(fake_embed, tmp_path, monkeypatch):
    db_path = tmp_path / "index.sqlite3"
    source = "advent_core/some/deeply/nested/module_name.py"
    _seed_long_index(
        db_path,
        text="word   " * 100 + "\n\nmore\ttext",
        source=source,
        section="class Agent " + "x" * 80,
    )
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=80, no_color=True))

    cli.search_command(query="hello", strategy="fixed", k=5, db=str(db_path))

    lines = _ANSI.sub("", buffer.getvalue()).splitlines()
    assert lines[0] == "Поиск — fixed"
    assert len(lines) == 3
    head, snippet = lines[1], lines[2]
    assert head.startswith("  1. ")
    assert f"{source}:1888-1904" in head
    assert "chunk_id" not in _ANSI.sub("", buffer.getvalue())
    assert "#0" not in _ANSI.sub("", buffer.getvalue())
    assert head.endswith("…") and len(head) <= 80
    assert snippet.startswith("     word word word")
    assert snippet.endswith("…") and len(snippet) <= 80
    assert "  " not in snippet.strip()


def test_search_wide_console_keeps_short_section_and_snippet_whole(
    fake_embed, tmp_path, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(db_path, text="short text", source="a.md", section="class Agent")
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=160, no_color=True))

    cli.search_command(query="hello", strategy="structure", k=5, db=str(db_path))

    lines = _ANSI.sub("", buffer.getvalue()).splitlines()
    assert lines[1].endswith("a.md:1888-1904  · class Agent")
    assert lines[2] == "     short text"


def test_show_truncates_to_twelve_lines_and_says_how_many_remain(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="\n".join(f"row-{i}" for i in range(20)),
        source="a.md",
        section="s",
    )

    cli.show_command(target="fixed:a.md#0", strategy="fixed", db=str(db_path), full=False)

    out = _ANSI.sub("", stdout_capture.getvalue())
    assert "row-11" in out
    assert "row-12" not in out
    assert "… ещё 8 строк (полный текст: --full)" in out
    assert "chunk_id" in out  # metadata still complete


def test_show_full_prints_everything_without_the_hint(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="\n".join(f"row-{i}" for i in range(20)),
        source="a.md",
        section="s",
    )

    cli.show_command(target="fixed:a.md#0", strategy="fixed", db=str(db_path), full=True)

    out = _ANSI.sub("", stdout_capture.getvalue())
    assert "row-19" in out
    assert "--full" not in out


def test_compare_prints_table_then_verdict_and_mrr_has_three_decimals(
    fake_embed, tmp_path, stdout_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    a = StrategyMetrics(n_questions=6, hit_at_1=0.5, hit_at_3=0.5, hit_at_5=0.5, mrr_at_5=0.347)
    b = StrategyMetrics(n_questions=6, hit_at_1=0.5, hit_at_3=0.5, hit_at_5=0.5, mrr_at_5=0.25)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": a, "structure": b}, broken=[]
        ),
    )

    cli.compare_command(db=str(db_path))

    out = _ANSI.sub("", stdout_capture.getvalue())
    title = out.index("Качество поиска")
    verdict = out.index("Выше MRR@5 в этой выборке")
    assert verdict > title
    assert out.count("0.347") == 2  # table cell and verdict line
    assert out.count("0.250") == 2


# --------------------------------------------------------------------------
# review round 2: cell width, DB path in errors, compare caveat, show, search -k
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", [80, 160])
def test_search_hit_with_wide_emoji_renders_exactly_two_lines(
    fake_embed, tmp_path, monkeypatch, width
):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="день первый 🔥 запрос 🚀 " * 40,
        source="week_01/tasks_01.md",
        section="## День 🔥 " + "x" * 120,
    )
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=width, no_color=True))

    cli.search_command(query="hello", strategy="fixed", k=5, db=str(db_path))

    lines = _ANSI.sub("", buffer.getvalue()).splitlines()
    assert len(lines) == 3
    assert all(cli.cell_len(line) <= width for line in lines)


def test_cut_counts_terminal_cells_not_characters():
    cut = cli._cut("🔥" * 10, 7)
    assert cli.cell_len(cut) <= 7
    assert cut.endswith("…")


def test_missing_index_inside_repo_shows_relative_path_not_absolute():
    missing = cli.PROJECT_ROOT / "week_05" / "definitely-missing.sqlite3"
    with pytest.raises(AdventError) as info:
        cli.index_module.load_runs(missing)
    message = str(info.value)
    assert "week_05/definitely-missing.sqlite3" in message
    assert str(cli.PROJECT_ROOT) not in message


def test_corrupt_index_inside_repo_shows_relative_path_not_absolute():
    garbage = cli.PROJECT_ROOT / "week_05" / "definitely-garbage.sqlite3"
    garbage.write_bytes(b"this is not a sqlite database")
    try:
        with pytest.raises(AdventError) as info:
            cli.index_module.load_runs(garbage)
    finally:
        garbage.unlink()
    message = str(info.value)
    assert "week_05/definitely-garbage.sqlite3" in message
    assert str(cli.PROJECT_ROOT) not in message


def test_missing_index_outside_repo_shows_only_the_file_name(tmp_path):
    with pytest.raises(AdventError) as info:
        cli.index_module.load_runs(tmp_path / "gone.sqlite3")
    message = str(info.value)
    assert "gone.sqlite3" in message
    assert str(tmp_path) not in message


def test_compare_with_zero_scored_questions_prints_no_verdict(
    fake_embed, tmp_path, stdout_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    empty = StrategyMetrics(n_questions=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.0)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": empty, "structure": empty},
            broken=[cli.index_module.BrokenQuestion(id=1, question="q?", reason="r")],
        ),
    )

    cli.compare_command(db=str(db_path))

    out = _flat(stdout_capture.getvalue())
    assert "Ничья" not in out
    assert "Выше MRR@5" not in out
    assert "Ни один вопрос нельзя оценить" in out


def test_compare_caveat_states_actual_count_on_stdout(
    fake_embed, tmp_path, stdout_capture, monkeypatch
):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)
    five = StrategyMetrics(n_questions=5, hit_at_1=0.5, hit_at_3=0.5, hit_at_5=0.5, mrr_at_5=0.5)
    monkeypatch.setattr(
        cli.index_module,
        "evaluate",
        lambda hits, questions, docs: EvalReport(
            per_strategy={"fixed": five, "structure": five}, broken=[]
        ),
    )

    cli.compare_command(db=str(db_path))

    out = _flat(stdout_capture.getvalue())
    assert "5 вопросов — малая выборка" in out
    assert "6 вопросов" not in out


def test_show_trailing_newline_is_not_an_extra_line(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="\n".join(f"row-{i}" for i in range(12)) + "\n",
        source="a.md",
        section="s",
    )

    cli.show_chunk("fixed:a.md#0", "fixed", db_path)

    assert "ещё" not in _ANSI.sub("", stdout_capture.getvalue())


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (13, "… ещё 1 строка (полный текст: --full)"),
        (15, "… ещё 3 строки (полный текст: --full)"),
        (20, "… ещё 8 строк (полный текст: --full)"),
        (63, "… ещё 51 строка (полный текст: --full)"),
        (23, "… ещё 11 строк (полный текст: --full)"),
    ],
)
def test_show_hint_uses_russian_plural_forms(tmp_path, stdout_capture, total, expected):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="\n".join(f"row-{i}" for i in range(total)),
        source="a.md",
        section="s",
    )

    cli.show_chunk("fixed:a.md#0", "fixed", db_path)

    assert expected in _ANSI.sub("", stdout_capture.getvalue())


def test_show_chunk_without_full_truncates_by_default(tmp_path, stdout_capture):
    db_path = tmp_path / "index.sqlite3"
    _seed_long_index(
        db_path,
        text="\n".join(f"row-{i}" for i in range(20)),
        source="a.md",
        section="s",
    )

    cli.show_chunk("fixed:a.md#0", "fixed", str(db_path))

    out = _ANSI.sub("", stdout_capture.getvalue())
    assert "row-12" not in out
    assert "ещё 8 строк" in out


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k_before_any_embedding_call(fake_embed, tmp_path, k):
    db_path = tmp_path / "index.sqlite3"
    _seed_index(db_path)

    with pytest.raises(AdventError, match="-k"):
        cli.search_hits("hello", "all", k, str(db_path))

    assert fake_embed.calls == []


def test_search_cli_rejects_k_zero_at_parse_time(tmp_path):
    proc = run_cli("search", "вопрос", "-k", "0", "--db", str(tmp_path / "x.sqlite3"))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
