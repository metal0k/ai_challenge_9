"""Review corpus for day 32, read from a git commit into its own sqlite.

`write_index` stores `purpose` in the shared `meta` table and deletes only the
strategy being written. Putting `review` into the day-31 file would relabel
that file and `help` would stop opening it, so this index is a different path.
Markdown and Python are cut by `chunk_structure`, which stamps
`strategy="structure"`; search filters on that column, so the stamp is
rewritten to `review` before the index is written.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from advent_core import console
from advent_core.client import mistral_client
from advent_core.config import PROJECT_ROOT
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult, embed_cost_usd
from advent_core.embeddings import embed_texts as embed_texts_fn
from advent_core.errors import AdventError
from adventassistant.corpus import INDEX_PATH, cloud_config, format_index_summary
from week_05.chunking import MAX_CHUNK_CHARS, Chunk, Document, chunk_structure
from week_05.index import (
    ENDPOINT_CLOUD,
    check_endpoint_free,
    display_path,
    load_meta,
    load_runs,
    write_index,
)

REVIEW_INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "index.w07d32.sqlite3"
PURPOSE = "w07d32"
STRATEGY = "review"
WEEK = 7
DAY = 32
SHORT_SHA = 12
# Corpus/index compatibility: bump when chunking changes, so a new cut of the
# same base SHA is never mistaken for an index built with the old one.
REVIEW_INDEX_VERSION = "1"

_SCHEMA_DIRS = ("advent_core/schemas/", "week_01/schemas/")
_PY_ROOTS = ("advent_core/", "advent_cli/", "adventassistant/")
_INDEX_HINT = "Сначала выполни `adventassistant index-review`."


def run_git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
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


def resolve_rev(root: Path, rev: str) -> str:
    result = run_git(["rev-parse", "--verify", f"{rev}^{{commit}}"], root)
    if result.returncode != 0:
        raise AdventError(f"Git не знает ревизию {rev!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_stdout(root: Path, args: list[str]) -> str:
    result = run_git(args, root)
    if result.returncode != 0:
        raise AdventError(f"git {' '.join(args)} завершился с ошибкой: {result.stderr.strip()}")
    return result.stdout


def _has_pycache(path: str) -> bool:
    return path.startswith("__pycache__/") or "/__pycache__/" in path


def _is_schema(path: str) -> bool:
    if _has_pycache(path) or path.startswith("tests/"):
        return False
    for prefix in _SCHEMA_DIRS:
        if path.startswith(prefix) and path.endswith(".json"):
            rest = path[len(prefix) :]
            return bool(rest) and "/" not in rest
    return False


def _is_python(path: str) -> bool:
    if not path.endswith(".py") or _has_pycache(path) or path.startswith("tests/"):
        return False
    if any(path.startswith(root) for root in _PY_ROOTS):
        return True
    head, sep, _rest = path.partition("/")
    return bool(sep) and head.startswith("week_") and head.removeprefix("week_").isdigit()


def _is_included(path: str) -> bool:
    if _has_pycache(path) or path.startswith("tests/"):
        return False
    if path == "README.md":
        return True
    if path.startswith("docs/") and path.endswith(".md"):
        return True
    return _is_schema(path) or _is_python(path)


def _paths_at_rev(root: Path, sha: str) -> list[str]:
    # -z: without it git C-quotes non-ASCII names ("\320\264...") and they never match.
    result = run_git(["ls-tree", "-r", "-z", "--name-only", sha], root)
    if result.returncode != 0:
        raise AdventError(f"git ls-tree завершился с ошибкой: {result.stderr.strip()}")
    return [path for path in result.stdout.split("\0") if path and _is_included(path)]


def _read_at_rev(root: Path, sha: str, path: str) -> Document:
    result = run_git(["show", f"{sha}:{path}"], root)
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
    if not any(_is_python(path) for path in paths):
        missing.append("production .py (advent_core, advent_cli, adventassistant, week_*)")
    if missing:
        raise AdventError(
            "В корпусе на этом commit не хватает: " + ", ".join(missing) + ". Индекс не построен."
        )


def collect_review(root: Path, rev: str = "HEAD") -> tuple[str, list[Document]]:
    """(full sha, included files sorted by path). Fails if a required kind is absent."""
    sha = resolve_rev(root, rev)
    paths = _paths_at_rev(root, sha)
    _require_kinds(paths)
    docs = [_read_at_rev(root, sha, path) for path in paths]
    docs.sort(key=lambda doc: doc.source)
    return sha, docs


def _as_review(chunk: Chunk) -> Chunk:
    prefix = "structure:"
    if chunk.strategy != "structure" or not chunk.chunk_id.startswith(prefix):
        raise AdventError(f"Чанк {chunk.chunk_id!r} не помечен стратегией structure.")
    return replace(
        chunk,
        strategy=STRATEGY,
        chunk_id=STRATEGY + ":" + chunk.chunk_id[len(prefix) :],
    )


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
    """Markdown and Python via structure-chunking relabelled to `review`; one chunk per schema.

    A Python file that does not parse stays one `(unparsed)` section, split
    further when it is over 4000 characters. An empty file yields nothing and
    is skipped. The index is not failed for either.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.source.endswith(".json"):
            chunks.append(_json_chunk(doc))
            continue
        chunks.extend(_as_review(chunk) for chunk in chunk_structure(doc))
    if not chunks:
        raise AdventError("Корпус не дал ни одного чанка — индекс не построен.")
    mislabelled = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.strategy != STRATEGY or not chunk.chunk_id.startswith(f"{STRATEGY}:")
    ]
    if mislabelled:
        raise AdventError(f"Чанк корпуса не помечен стратегией review: {mislabelled[0]}.")
    return chunks


def _check_embeddings(chunks: list[Chunk], result: EmbedResult) -> None:
    try:
        matrix = np.asarray(result.vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise AdventError(f"Эмбеддинг не является матрицей float32: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
        raise AdventError(f"Эмбеддинг вернул форму {matrix.shape} на {len(chunks)} чанков.")
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
            journal_extra={"command": "review_index"},
        )


def _refuse_db(path: Path) -> None:
    """Do not let `write_index` relabel the day-31 file or any foreign purpose."""
    if path.expanduser().resolve() == INDEX_PATH.expanduser().resolve():
        raise AdventError(
            "index-review не пишет в индекс дня 31 "
            f"({INDEX_PATH.name}): запись сменила бы purpose, и help перестал бы "
            "открывать свой файл."
        )
    if not path.exists():
        return
    meta = load_meta(path)
    purpose = meta.get("purpose")
    if purpose is not None and purpose != PURPOSE:
        raise AdventError(
            f"База {display_path(path)} уже помечена purpose={purpose!r}, не {PURPOSE!r}. "
            "index-review не перезаписывает чужой индекс."
        )


def build_review_index(root: Path, *, rev: str = "HEAD", db_path: Path | None = None) -> str:
    """Chunk, embed and write the day-32 index. The returned text is the stdout product."""
    path = db_path or REVIEW_INDEX_PATH
    _refuse_db(path)
    sha, docs = collect_review(root, rev)
    chunks = build_chunks(docs)
    check_endpoint_free(path, ENDPOINT_CLOUD)
    console.note(f"эмбеддинг review: {len(chunks)} чанков, модель {DEFAULT_EMBED_MODEL}")
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


def require_review_index(db_path: Path, base_sha: str) -> Path:
    """Refuse anything that is not this day's cloud `review` run of the base commit.

    Called before the query embedding. A mismatch names `adventassistant index-review`.
    """
    if not db_path.exists():
        raise AdventError(f"Индекс не найден: {display_path(db_path)}. {_INDEX_HINT}")
    try:
        meta = load_meta(db_path)
        runs = load_runs(db_path)
    except AdventError as exc:
        message = exc.message
        if "adventassistant index-review" not in message:
            message = f"{message} {_INDEX_HINT}"
        raise AdventError(message, hint=exc.hint) from exc
    if meta.get("purpose") != PURPOSE:
        raise AdventError(
            f"Индекс {display_path(db_path)} не для ревью "
            f"(purpose={meta.get('purpose')!r}). {_INDEX_HINT}"
        )
    run = runs.get(STRATEGY)
    if run is None:
        raise AdventError(f"В индексе {display_path(db_path)} нет run review. {_INDEX_HINT}")
    if run.model != DEFAULT_EMBED_MODEL:
        raise AdventError(
            f"Run review проиндексирован моделью {run.model}, нужна {DEFAULT_EMBED_MODEL}. "
            f"{_INDEX_HINT}"
        )
    if run.corpus_rev != base_sha:
        raise AdventError(
            f"corpus_rev индекса ({run.corpus_rev}) не равен базе {base_sha}. {_INDEX_HINT}"
        )
    if run.endpoint != ENDPOINT_CLOUD:
        raise AdventError(
            f"Run review записан с endpoint={run.endpoint!r}, нужен {ENDPOINT_CLOUD}. {_INDEX_HINT}"
        )
    if run.doc_prefix or run.query_prefix:
        which: list[str] = []
        if run.doc_prefix:
            which.append("doc_prefix")
        if run.query_prefix:
            which.append("query_prefix")
        raise AdventError(
            f"Run review хранит непустой {', '.join(which)}. "
            f"Облачный индекс дня 32 их не использует. {_INDEX_HINT}"
        )
    return db_path
