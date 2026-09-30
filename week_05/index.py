"""SQLite index: storage, search, chunk stats, eval (SPEC-w05d21.md §5, §6).

Plain module functions, not typer commands — week_05/server.py (a later day,
per SPEC §0) imports these directly for its search tool, the way
week_04/server.py already calls week_04 functions in-process rather than
shelling out to a CLI.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from advent_core.config import PROJECT_ROOT
from advent_core.embeddings import EmbedResult, embed_cost_usd
from advent_core.errors import AdventError
from week_05.chunking import Chunk, Document

DEFAULT_DB = PROJECT_ROOT / "data" / "rag" / "index.sqlite3"
SCHEMA_VERSION = 1
SMALL_CHUNK_CHARS = 200

_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    n_chars INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);

CREATE TABLE IF NOT EXISTS runs (
    strategy TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    n_chunks INTEGER NOT NULL,
    prompt_tokens INTEGER,
    requests INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd REAL,
    corpus_rev TEXT NOT NULL,
    corpus_files INTEGER NOT NULL,
    corpus_chars INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunInfo:
    """One row of `runs` — one strategy's last indexing run."""

    strategy: str
    model: str
    dim: int
    n_chunks: int
    prompt_tokens: int | None
    requests: int
    latency_ms: int
    cost_usd: float | None
    corpus_rev: str
    corpus_files: int
    corpus_chars: int
    created_at: str


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: full chunk metadata + its cosine score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Stats:
    """Size distribution + heuristic cut-share for one strategy's chunks (SPEC §6)."""

    n_chunks: int
    min_chars: int
    median_chars: float
    p90_chars: float
    max_chars: int
    small_share: float
    cut_share: float


@dataclass(frozen=True, slots=True)
class ExpectedAnswer:
    source: str
    anchor: str


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: int
    question: str
    expected: list[ExpectedAnswer]
    placeholder: bool = False


@dataclass(frozen=True, slots=True)
class BrokenQuestion:
    """A question dropped from the metric: its anchor is a dataset error, not a miss."""

    id: int
    question: str
    reason: str


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    n_questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    per_strategy: dict[str, StrategyMetrics]
    broken: list[BrokenQuestion]


# ---------------------------------------------------------------------------
# chunk_stats — pure, no DB, no API (powers `index --dry-run`).
# ---------------------------------------------------------------------------


def _is_cut_md(chunk: Chunk) -> bool:
    """Ends mid-sentence, or an odd number of opening fence lines (SPEC §6)."""
    stripped = chunk.text.rstrip()
    if not stripped:
        return False
    if stripped[-1] not in ".!?:)":
        return True
    fence_lines = sum(1 for line in chunk.text.split("\n") if _FENCE_RE.match(line.strip()))
    return fence_lines % 2 == 1


def _py_def_spans(doc: Document) -> list[tuple[int, int]]:
    """(lineno, end_lineno) of every top-level def/class, from the WHOLE file's ast."""
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return []
    return [
        (node.lineno, node.end_lineno)
        for node in tree.body
        if isinstance(node, _DEF_TYPES) and node.end_lineno is not None
    ]


def _is_cut_py(chunk: Chunk, spans: list[tuple[int, int]]) -> bool:
    """Starts or ends strictly inside a top-level def/class body (SPEC §6).

    A chunk starting/ending exactly AT a span's boundary is a legitimate
    structure-strategy cut, not a cut-through — only strictly inside counts.
    """
    for lineno, end_lineno in spans:
        starts_inside = lineno < chunk.line_start <= end_lineno
        ends_inside = lineno <= chunk.line_end < end_lineno
        if starts_inside or ends_inside:
            return True
    return False


def chunk_stats(chunks: list[Chunk], docs: list[Document]) -> Stats:
    """Size distribution + cut-share heuristics (SPEC §6). No network — used by --dry-run too."""
    if not chunks:
        return Stats(0, 0, 0.0, 0.0, 0, 0.0, 0.0)
    sizes = sorted(c.n_chars for c in chunks)
    n = len(chunks)
    small = sum(1 for s in sizes if s < SMALL_CHUNK_CHARS)

    docs_by_source = {d.source: d for d in docs}
    py_span_cache: dict[str, list[tuple[int, int]]] = {}
    cuts = 0
    for c in chunks:
        doc = docs_by_source.get(c.source)
        if doc is None:
            continue
        if c.source.endswith(".py"):
            if c.source not in py_span_cache:
                py_span_cache[c.source] = _py_def_spans(doc)
            if _is_cut_py(c, py_span_cache[c.source]):
                cuts += 1
        elif _is_cut_md(c):
            cuts += 1

    return Stats(
        n_chunks=n,
        min_chars=sizes[0],
        median_chars=float(np.median(sizes)),
        p90_chars=float(np.percentile(sizes, 90)),
        max_chars=sizes[-1],
        small_share=small / n,
        cut_share=cuts / n,
    )


# ---------------------------------------------------------------------------
# SQLite: schema, write, read.
# ---------------------------------------------------------------------------


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def display_path(db_path: Path | None) -> str:
    """Repo-relative POSIX path when inside the repo, else just the file name.

    Never the absolute path: error text and notes end up on a public video.
    """
    path = (db_path or DEFAULT_DB).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _require_db(db_path: Path) -> None:
    if not db_path.exists():
        raise AdventError(
            f"Индекс не найден: {display_path(db_path)}. Сначала выполни `adventrag index`."
        )


@contextmanager
def _sqlite_errors(db_path: Path) -> Iterator[None]:
    """Translate a corrupt/unreadable DB into an AdventError, not a traceback.

    `db_path.exists()` (checked by `_require_db`) says nothing about the
    file's CONTENTS — a garbage file passes it and only fails once sqlite
    actually reads it (open succeeds lazily; the error surfaces on the first
    query).
    """
    try:
        yield
    except sqlite3.Error as exc:
        raise AdventError(
            f"Индекс повреждён или недоступен: {display_path(db_path)} ({exc}).",
            hint="Удали файл индекса и запусти `adventrag index` заново.",
        ) from exc


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    _require_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        strategy=row["strategy"],
        source=row["source"],
        title=row["title"],
        section=row["section"],
        ordinal=row["ordinal"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        text=row["text"],
    )


def _run_info_from_row(row: sqlite3.Row) -> RunInfo:
    return RunInfo(
        strategy=row["strategy"],
        model=row["model"],
        dim=row["dim"],
        n_chunks=row["n_chunks"],
        prompt_tokens=row["prompt_tokens"],
        requests=row["requests"],
        latency_ms=row["latency_ms"],
        cost_usd=row["cost_usd"],
        corpus_rev=row["corpus_rev"],
        corpus_files=row["corpus_files"],
        corpus_chars=row["corpus_chars"],
        created_at=row["created_at"],
    )


def write_index(
    db_path: Path | None,
    results: dict[str, tuple[list[Chunk], EmbedResult]],
    *,
    corpus_rev: str,
    corpus_files: int,
    corpus_chars: int,
) -> None:
    """Replace ALL given strategies' rows in one transaction (SPEC §5).

    A failure partway through (a bad batch, a disk error) rolls back the
    whole write: strategies untouched by this call, and the previous rows of
    the ones IN `results`, are left exactly as they were — never a mix of an
    old and a new corpus snapshot for the same strategy.
    """
    path = db_path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with _sqlite_errors(path):
        conn = sqlite3.connect(path)
        try:
            _ensure_schema(conn)
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            for strategy, (chunks, embed_result) in results.items():
                conn.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
                conn.execute("DELETE FROM runs WHERE strategy = ?", (strategy,))
                rows = [
                    (
                        c.chunk_id,
                        c.strategy,
                        c.source,
                        c.title,
                        c.section,
                        c.ordinal,
                        c.char_start,
                        c.char_end,
                        c.line_start,
                        c.line_end,
                        c.n_chars,
                        c.text,
                        np.asarray(embed_result.vectors[i], dtype=np.float32).tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ]
                conn.executemany(
                    "INSERT INTO chunks (chunk_id, strategy, source, title, section, ordinal, "
                    "char_start, char_end, line_start, line_end, n_chars, text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.execute(
                    "INSERT INTO runs (strategy, model, dim, n_chunks, prompt_tokens, requests, "
                    "latency_ms, cost_usd, corpus_rev, corpus_files, corpus_chars, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        strategy,
                        embed_result.model,
                        embed_result.dim,
                        len(chunks),
                        embed_result.prompt_tokens,
                        embed_result.requests,
                        embed_result.latency_ms,
                        embed_cost_usd(embed_result.model, embed_result.prompt_tokens),
                        corpus_rev,
                        corpus_files,
                        corpus_chars,
                        created_at,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def load_runs(db_path: Path | None = None) -> dict[str, RunInfo]:
    path = db_path or DEFAULT_DB
    with _sqlite_errors(path):
        conn = _connect_ro(path)
        try:
            rows = conn.execute("SELECT * FROM runs").fetchall()
        finally:
            conn.close()
    return {row["strategy"]: _run_info_from_row(row) for row in rows}


def check_comparable(runs: dict[str, RunInfo], strategies: list[str]) -> None:
    """Refuse to compare strategies indexed with a different model or corpus_rev (SPEC §5)."""
    infos: list[RunInfo] = []
    for strategy in strategies:
        if strategy not in runs:
            raise AdventError(
                f"Стратегия {strategy!r} не проиндексирована. "
                f"Сначала `adventrag index --strategy {strategy}`."
            )
        infos.append(runs[strategy])

    models = {info.model for info in infos}
    if len(models) > 1:
        detail = ", ".join(f"{s}={runs[s].model}" for s in strategies)
        raise AdventError(
            f"Стратегии проиндексированы разными моделями ({detail}) — сравнение "
            "было бы нечестным (разные векторные пространства). Переиндексируй "
            "одной моделью: `adventrag index --strategy all --model <модель>`."
        )

    revs = {info.corpus_rev for info in infos}
    if len(revs) > 1:
        detail = ", ".join(f"{s}={runs[s].corpus_rev}" for s in strategies)
        raise AdventError(
            f"Стратегии проиндексированы разными снимками корпуса ({detail}). "
            "Переиндексируй обе на одной ревизии: `adventrag index --strategy all`."
        )


def search(db_path: Path | None, strategy: str, query_vec: np.ndarray, k: int = 5) -> list[Hit]:
    """Top-k by cosine (= dot on L2-normalized vectors), full scan (SPEC §5)."""
    path = db_path or DEFAULT_DB
    with _sqlite_errors(path):
        conn = _connect_ro(path)
        try:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE strategy = ? ORDER BY source, ordinal", (strategy,)
            ).fetchall()
        finally:
            conn.close()
    if not rows:
        return []

    vectors = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if vectors.shape[1] != query.shape[0]:
        raise AdventError(
            f"Размерность запроса ({query.shape[0]}) не совпадает с размерностью индекса "
            f"стратегии {strategy!r} ({vectors.shape[1]}). Переиндексируй той же моделью "
            "или задай ту же модель для запроса."
        )

    scores = vectors @ query
    # np.argsort's tie order is not guaranteed stable (default quicksort) —
    # break ties by chunk_id so the result is deterministic run to run.
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["chunk_id"]))[:k]
    return [Hit(chunk=_chunk_from_row(rows[idx]), score=float(scores[idx])) for idx in order]


def get_chunk(db_path: Path | None, chunk_id: str) -> Chunk | None:
    path = db_path or DEFAULT_DB
    with _sqlite_errors(path):
        conn = _connect_ro(path)
        try:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        finally:
            conn.close()
    return _chunk_from_row(row) if row is not None else None


def find_chunk_by_line(db_path: Path | None, strategy: str, source: str, line: int) -> Chunk | None:
    """The chunk of `strategy` covering `source:line` (lowest ordinal, if several overlap)."""
    path = db_path or DEFAULT_DB
    with _sqlite_errors(path):
        conn = _connect_ro(path)
        try:
            row = conn.execute(
                "SELECT * FROM chunks WHERE strategy = ? AND source = ? "
                "AND line_start <= ? AND line_end >= ? ORDER BY ordinal LIMIT 1",
                (strategy, source, line, line),
            ).fetchone()
        finally:
            conn.close()
    return _chunk_from_row(row) if row is not None else None


def load_chunks(db_path: Path | None, strategy: str) -> list[Chunk]:
    """All chunks of `strategy` as stored — `compare`'s stats come from these, not a re-chunk."""
    path = db_path or DEFAULT_DB
    with _sqlite_errors(path):
        conn = _connect_ro(path)
        try:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE strategy = ? ORDER BY source, ordinal", (strategy,)
            ).fetchall()
        finally:
            conn.close()
    return [_chunk_from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Eval (SPEC §6).
# ---------------------------------------------------------------------------


def load_eval(path: Path) -> list[EvalQuestion]:
    """Load the eval_questions.json format (SPEC §6): id, question, expected[], placeholder?."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdventError(f"Файл вопросов для оценки не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AdventError(f"Файл вопросов повреждён ({path}): {exc}") from exc

    questions: list[EvalQuestion] = []
    for item in raw:
        expected = [
            ExpectedAnswer(source=e["source"], anchor=e["anchor"]) for e in item["expected"]
        ]
        questions.append(
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                expected=expected,
                placeholder=bool(item.get("placeholder", False)),
            )
        )
    return questions


def _valid_expected(
    question: EvalQuestion, docs_by_source: dict[str, Document]
) -> tuple[list[ExpectedAnswer], list[str]]:
    """Split a question's expected answers into (valid, reasons the rest were dropped).

    An anchor absent from the current corpus snapshot is a dataset error, not
    a strategy miss (SPEC §6) — it is dropped here rather than counted against
    every strategy's score.
    """
    valid: list[ExpectedAnswer] = []
    reasons: list[str] = []
    for e in question.expected:
        doc = docs_by_source.get(e.source)
        if doc is None:
            reasons.append(f"{e.source!r} нет в снимке корпуса")
        elif e.anchor not in doc.text:
            reasons.append(f"якорь {e.anchor!r} не найден в {e.source!r}")
        else:
            valid.append(e)
    return valid, reasons


def _rank_of(hits: list[Hit], expected: list[ExpectedAnswer]) -> int | None:
    """1-based rank of the first hit matching any expected (source, anchor-in-text); None = miss."""
    for rank, hit in enumerate(hits, start=1):
        if any(e.source == hit.chunk.source and e.anchor in hit.chunk.text for e in expected):
            return rank
    return None


def _metrics_from_ranks(ranks: list[int | None]) -> StrategyMetrics:
    """hit@1/3/5 + MRR@5 from each question's rank (1-based, None = not in top-5)."""
    n = len(ranks)
    if n == 0:
        return StrategyMetrics(
            n_questions=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.0
        )
    hit_at_1 = sum(1 for r in ranks if r == 1) / n
    hit_at_3 = sum(1 for r in ranks if r is not None and r <= 3) / n
    hit_at_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr_at_5 = sum((1 / r) if r is not None else 0.0 for r in ranks) / n
    return StrategyMetrics(
        n_questions=n, hit_at_1=hit_at_1, hit_at_3=hit_at_3, hit_at_5=hit_at_5, mrr_at_5=mrr_at_5
    )


def evaluate(
    hits_by_question: dict[str, dict[int, list[Hit]]],
    questions: list[EvalQuestion],
    docs_by_source: dict[str, Document],
) -> EvalReport:
    """Per-strategy hit@1/3/5 + MRR@5 (SPEC §6).

    `hits_by_question[strategy][question.id]` is that question's top-k hits
    for that strategy (SPEC fixes k=5 for the metric regardless of what the
    caller's search used). A question with no valid expected answer left
    after `_valid_expected` is excluded from every strategy and reported in
    `broken` instead of scored as a miss.
    """
    broken: list[BrokenQuestion] = []
    evaluated: list[tuple[EvalQuestion, list[ExpectedAnswer]]] = []
    for q in questions:
        valid, reasons = _valid_expected(q, docs_by_source)
        # ANY invalid location is a dataset error, not just an entirely-invalid
        # question: a question half-valid stays scoreable in a way that hides
        # the broken half (SPEC §6 — the whole question drops to `broken`).
        if reasons:
            broken.append(BrokenQuestion(id=q.id, question=q.question, reason="; ".join(reasons)))
            continue
        evaluated.append((q, valid))

    per_strategy: dict[str, StrategyMetrics] = {}
    for strategy, by_question in hits_by_question.items():
        ranks = [_rank_of(by_question.get(q.id, []), valid) for q, valid in evaluated]
        per_strategy[strategy] = _metrics_from_ranks(ranks)
    return EvalReport(per_strategy=per_strategy, broken=broken)


def leader(values: dict[str, float]) -> str | None:
    """Key with the strict max value, or None on a tie — never print a winner on a tie."""
    if not values:
        return None
    best = max(values.values())
    winners = [k for k, v in values.items() if v == best]
    return winners[0] if len(winners) == 1 else None
