"""Docs corpus for the day-31 assistant, read from a git commit.

The working tree is never read: an uncommitted `docs/` must fail the index,
not become an empty one. Markdown is cut by `chunk_structure`, which stamps
`strategy="structure"`; search filters on that column, so the stamp is
rewritten to `docs` before the index is written.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from advent_core import console
from advent_core.client import mistral_client
from advent_core.config import PROJECT_ROOT, Config, ConfigError
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult, embed_cost_usd
from advent_core.embeddings import embed_texts as embed_texts_fn
from advent_core.errors import AdventError
from week_05.chunking import MAX_CHUNK_CHARS, Chunk, Document, chunk_structure
from week_05.index import (
    ENDPOINT_CLOUD,
    check_endpoint_free,
    display_path,
    load_meta,
    load_runs,
    write_index,
)

INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "index.w07d31.sqlite3"
PURPOSE = "w07d31"
STRATEGY = "docs"
WEEK = 7
DAY = 31
SHORT_SHA = 12

_SCHEMA_DIRS = ("advent_core/schemas/", "week_01/schemas/")
_INDEX_HINT = "Сначала выполни `adventassistant index`."


def cloud_config(model: str) -> Config:
    """Cloud Mistral config.

    `ADVENT_BASE_URL` on this machine often points at LM Studio. Day 31 is the
    cloud path; a loopback URL would embed with the wrong model and a stub key.
    """
    base = Config.resolve(model=model, stream=False)
    if not base.is_local:
        return base
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise ConfigError("Не найден MISTRAL_API_KEY.")
    return replace(base, api_key=key, base_url=None, offline=False)


def _run_git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as exc:
        raise AdventError(f"Не удалось запустить git: {exc}") from exc


def _resolve_rev(root: Path, rev: str) -> str:
    result = _run_git(["rev-parse", "--verify", f"{rev}^{{commit}}"], root)
    if result.returncode != 0:
        raise AdventError(f"Git не знает ревизию {rev!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def _is_schema(path: str) -> bool:
    for prefix in _SCHEMA_DIRS:
        if path.startswith(prefix) and path.endswith(".json"):
            rest = path[len(prefix) :]
            return bool(rest) and "/" not in rest
    return False


def _is_included(path: str) -> bool:
    if path == "README.md":
        return True
    if path.startswith("docs/") and path.endswith(".md"):
        return True
    return _is_schema(path)


def _paths_at_rev(root: Path, sha: str) -> list[str]:
    result = _run_git(["ls-tree", "-r", "--name-only", sha], root)
    if result.returncode != 0:
        raise AdventError(f"git ls-tree завершился с ошибкой: {result.stderr.strip()}")
    return [path for path in result.stdout.split("\n") if path and _is_included(path)]


def _read_at_rev(root: Path, sha: str, path: str) -> Document:
    result = _run_git(["show", f"{sha}:{path}"], root)
    if result.returncode != 0:
        raise AdventError(f"git show не нашёл {path} в ревизии {sha}: {result.stderr.strip()}")
    return Document(source=path, text=result.stdout.replace("\r\n", "\n"))


def _require_kinds(paths: list[str]) -> None:
    missing: list[str] = []
    if "README.md" not in paths:
        missing.append("README.md")
    if not any(path.startswith("docs/") and path.endswith(".md") for path in paths):
        missing.append("docs/**/*.md")
    if not any(_is_schema(path) for path in paths):
        missing.append("advent_core/schemas/*.json или week_01/schemas/*.json")
    if missing:
        raise AdventError(
            "В корпусе на этом commit не хватает: " + ", ".join(missing) + ". Индекс не построен."
        )


def collect_docs(root: Path, rev: str = "HEAD") -> tuple[str, list[Document]]:
    """(full sha, included files sorted by path). Fails if a required kind is absent."""
    sha = _resolve_rev(root, rev)
    paths = _paths_at_rev(root, sha)
    _require_kinds(paths)
    docs = [_read_at_rev(root, sha, path) for path in paths]
    docs.sort(key=lambda doc: doc.source)
    return sha, docs


def _as_docs(chunk: Chunk) -> Chunk:
    prefix = "structure:"
    if chunk.strategy != "structure" or not chunk.chunk_id.startswith(prefix):
        raise AdventError(f"Чанк {chunk.chunk_id!r} не помечен стратегией structure.")
    chunk_id = STRATEGY + ":" + chunk.chunk_id[len(prefix) :]
    return replace(chunk, strategy=STRATEGY, chunk_id=chunk_id)


def _json_chunk(doc: Document) -> Chunk:
    if len(doc.text) > MAX_CHUNK_CHARS:
        raise AdventError(
            f"Схема {doc.source} длиннее {MAX_CHUNK_CHARS} символов "
            f"({len(doc.text)}) — разрезать файл схемы нельзя."
        )
    lines = doc.text.split("\n")
    return Chunk(
        chunk_id=f"{STRATEGY}:{doc.source}#0",
        strategy=STRATEGY,
        source=doc.source,
        title=Path(doc.source).name,
        section=doc.source,
        ordinal=0,
        char_start=0,
        char_end=len(doc.text),
        line_start=1,
        line_end=max(1, len(lines)),
        text=doc.text,
    )


def build_chunks(docs: list[Document]) -> list[Chunk]:
    """Markdown via structure-chunking relabelled to `docs`; one `docs` chunk per schema."""
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.source.endswith(".json"):
            chunks.append(_json_chunk(doc))
            continue
        chunks.extend(_as_docs(chunk) for chunk in chunk_structure(doc))
    if not chunks:
        raise AdventError("Корпус не дал ни одного чанка — индекс не построен.")
    mislabelled = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.strategy != STRATEGY or not chunk.chunk_id.startswith(f"{STRATEGY}:")
    ]
    if mislabelled:
        raise AdventError(f"Чанк корпуса не помечен стратегией docs: {mislabelled[0]}.")
    return chunks


def _check_embeddings(chunks: list[Chunk], result: EmbedResult) -> None:
    try:
        matrix = np.asarray(result.vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise AdventError(f"Эмбеддинг не является матрицей float32: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
        raise AdventError(f"Эмбеддинг вернул форму {matrix.shape} на {len(chunks)} чанков.")
    # Finite only after the float32 cast: 1e40 is a finite Python float and Inf as float32.
    if not np.isfinite(matrix).all():
        raise AdventError("Эмбеддинг содержит нечисловое значение (NaN/Inf).")


def _embed_documents(chunks: list[Chunk]) -> EmbedResult:
    config = cloud_config(DEFAULT_EMBED_MODEL)
    with mistral_client(config) as client:
        # journal_path stays omitted: None lands in journal.LOG_DIR, which tests patch.
        return embed_texts_fn(
            client,
            DEFAULT_EMBED_MODEL,
            [chunk.text for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
            week=WEEK,
            day=DAY,
            journal_extra={"command": "assistant_index"},
        )


def format_index_summary(
    sha: str,
    n_files: int,
    n_chunks: int,
    tokens: int | None,
    cost: float | None,
) -> str:
    token_label = "?" if tokens is None else str(tokens)
    cost_label = "?" if cost is None else f"${cost:.6f}"
    return (
        f"{sha[:SHORT_SHA]}\n"
        f"файлов: {n_files}\n"
        f"чанков: {n_chunks}\n"
        f"токенов: {token_label}\n"
        f"стоимость: {cost_label}\n"
    )


def build_index(root: Path, *, rev: str = "HEAD", db_path: Path | None = None) -> str:
    """Chunk, embed and write the day-31 index. The returned text is the stdout product."""
    path = db_path or INDEX_PATH
    sha, docs = collect_docs(root, rev)
    chunks = build_chunks(docs)
    check_endpoint_free(path, ENDPOINT_CLOUD)
    console.note(f"эмбеддинг docs: {len(chunks)} чанков, модель {DEFAULT_EMBED_MODEL}")
    embedded = _embed_documents(chunks)
    _check_embeddings(chunks, embedded)
    cost = embed_cost_usd(embedded.model, embedded.prompt_tokens)
    write_index(
        path,
        {STRATEGY: (chunks, embedded)},
        corpus_rev=sha,
        corpus_files=len(docs),
        corpus_chars=sum(len(doc.text) for doc in docs),
        endpoint=ENDPOINT_CLOUD,
        meta={"purpose": PURPOSE},
    )
    console.note(f"индекс записан: {display_path(path)}")
    return format_index_summary(sha, len(docs), len(chunks), embedded.prompt_tokens, cost)


def require_index(db_path: Path | None = None) -> Path:
    """Refuse anything that is not this day's `docs` run of `mistral-embed`."""
    path = db_path or INDEX_PATH
    if not path.exists():
        raise AdventError(f"Индекс не найден: {display_path(path)}. {_INDEX_HINT}")
    try:
        meta = load_meta(path)
        runs = load_runs(path)
    except AdventError as exc:
        message = exc.message
        if "adventassistant index" not in message:
            message = f"{message} {_INDEX_HINT}"
        raise AdventError(message, hint=exc.hint) from exc
    if meta.get("purpose") != PURPOSE:
        raise AdventError(
            f"Индекс {display_path(path)} не для этого дня "
            f"(purpose={meta.get('purpose')!r}). {_INDEX_HINT}"
        )
    run = runs.get(STRATEGY)
    if run is None:
        raise AdventError(f"В индексе {display_path(path)} нет run docs. {_INDEX_HINT}")
    if run.model != DEFAULT_EMBED_MODEL:
        raise AdventError(
            f"Run docs проиндексирован моделью {run.model}, нужна {DEFAULT_EMBED_MODEL}. "
            f"{_INDEX_HINT}"
        )
    return path
