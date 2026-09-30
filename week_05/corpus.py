"""Collect the tracked corpus at a pinned git revision (SPEC-w05d21.md §1).

Content is read from the commit (`git show <rev>:<path>`), never from the
working tree, so an uncommitted edit never leaks into the index or the demo
frame — the same reasoning `check_staged.py` already applies to the public
push guard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from advent_core.errors import AdventError
from week_05.chunking import Document

# advent_core/*.py: top level only, not the whole subtree (prompts/, schemas/, …).
_PY_INCLUDE_DIR = "advent_core"
_MD_EXCLUDE_PREFIX = ".agents/"


def _is_included(path: str) -> bool:
    if path.endswith(".md"):
        return not path.startswith(_MD_EXCLUDE_PREFIX)
    if path.endswith(".py") and path.startswith(f"{_PY_INCLUDE_DIR}/"):
        rest = path[len(_PY_INCLUDE_DIR) + 1 :]
        return "/" not in rest
    return False


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


def _paths_at_rev(root: Path, sha: str) -> list[str]:
    """Paths tracked AT `sha`, not in the working index — a file added or
    deleted after `sha` must not affect a pinned-rev collection."""
    result = _run_git(["ls-tree", "-r", "--name-only", sha], root)
    if result.returncode != 0:
        raise AdventError(f"git ls-tree завершился с ошибкой: {result.stderr.strip()}")
    return [p for p in result.stdout.split("\n") if p and _is_included(p)]


def _read_at_rev(root: Path, sha: str, path: str) -> Document:
    result = _run_git(["show", f"{sha}:{path}"], root)
    if result.returncode != 0:
        raise AdventError(f"git show не нашёл {path} в ревизии {sha}: {result.stderr.strip()}")
    text = result.stdout.replace("\r\n", "\n")
    return Document(source=path, text=text)


def collect_corpus(root: Path, rev: str = "HEAD") -> tuple[str, list[Document]]:
    """(full commit sha, docs sorted by source) — see module docstring for the "why"."""
    sha = _resolve_rev(root, rev)
    paths = _paths_at_rev(root, sha)
    docs = [_read_at_rev(root, sha, path) for path in paths]
    docs.sort(key=lambda d: d.source)
    return sha, docs
