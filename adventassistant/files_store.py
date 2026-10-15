"""Changeset store for the day-34 file assistant: runs, overlay reads, diff, apply.

`call_tool_once` spawns a fresh server per call, so all state lives on disk in
`logs/files/<run_id>/`. The model never sees or names that path; the server gets only
a run id and rebuilds the path itself.
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from advent_core.config import PROJECT_ROOT, redact
from advent_core.errors import AdventError
from adventassistant import files_scope as scope
from adventassistant.files_scope import ScopeError, Tracked

MANIFEST_VERSION = 1
RUN_ID_RE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}\Z", re.ASCII)
LIST_LIMIT = 200
SEARCH_HITS = 100
SEARCH_LINE_CHARS = 200
SEARCH_FILES = 400
SEARCH_BYTES = 8 * 1024 * 1024
REGEX_MAX = 200
READ_LINES = 400
READ_CHARS = 40_000


class FilesError(AdventError):
    """A tool-level refusal (bad argument, missing file, limit)."""


class ApplyConflictError(AdventError):
    """`files-apply` found a sha256/validation mismatch; nothing was written."""

    exit_code = 12

    def __init__(self, message: str, *, already_applied: bool = False) -> None:
        super().__init__(message)
        self.already_applied = already_applied


class ApplyRolledBackError(AdventError):
    """A write failed mid-apply; earlier files were restored from backup."""

    exit_code = 13

    def __init__(
        self,
        message: str,
        *,
        rolled_back: list[str],
        rollback_failed: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.rollback_failed = rollback_failed or []


@dataclass(slots=True)
class ApplyResult:
    written: list[str] = field(default_factory=list)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- runs ---------------------------------------------------------------------


def new_run_id(now: datetime | None = None) -> str:
    return f"{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise FilesError("Некорректный run_id: ожидается YYYYMMDD-HHMMSS-<8 hex>.")
    return run_id


def runs_dir(root: Path) -> Path:
    return root / "logs" / "files"


def run_dir_path(run_id: str, root: Path | None = None) -> Path:
    """Validated run directory: a plain directory directly under logs/files."""
    root = root or PROJECT_ROOT
    validate_run_id(run_id)
    for part in (root / "logs", runs_dir(root)):
        if scope.is_link(part):
            raise FilesError("logs/files не должен быть ссылкой или reparse point.")
    path = runs_dir(root) / run_id
    if not os.path.lexists(path):
        raise FilesError(f"Прогон {run_id} не найден.")
    if scope.is_link(path) or not path.is_dir():
        raise FilesError(f"Каталог прогона {run_id} не является обычным каталогом.")
    try:
        path.resolve().relative_to(runs_dir(root).resolve())
    except ValueError:
        raise FilesError(f"Каталог прогона {run_id} вне logs/files.") from None
    return path


def create_run(
    goal: str, model: str, root: Path | None = None, *, now: datetime | None = None
) -> tuple[str, Path]:
    """Create `logs/files/<run_id>` exclusively and write an empty manifest."""
    root = root or PROJECT_ROOT
    for part in (root / "logs", runs_dir(root)):
        if scope.is_link(part):
            raise FilesError("logs/files не должен быть ссылкой или reparse point.")
    runs_dir(root).mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(now)
    path = runs_dir(root) / run_id
    try:
        path.mkdir()
    except FileExistsError:
        raise FilesError(f"Прогон {run_id} уже существует.") from None
    manifest = {
        "version": MANIFEST_VERSION,
        "run_id": run_id,
        "goal": goal,
        "model": model,
        "files": {},
    }
    _atomic_write(path / "manifest.json", _dump(manifest).encode("utf-8"))
    _atomic_write(path / "changes.diff", b"")
    return run_id, path


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _atomic_write(target: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def load_manifest(run_id: str, root: Path | None = None) -> dict[str, Any]:
    """Read and validate manifest.json of a saved run."""
    root = root or PROJECT_ROOT
    run_dir = run_dir_path(run_id, root)
    try:
        data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FilesError("Манифест прогона не читается или повреждён.") from None
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        raise FilesError("Неподдерживаемая версия манифеста.")
    if data.get("run_id") != run_id:
        raise FilesError("run_id в манифесте не совпадает с каталогом.")
    files = data.get("files")
    if not isinstance(files, dict):
        raise FilesError("Манифест повреждён: files.")
    for key, entry in files.items():
        try:
            same = scope.canonical(key) == key
        except ScopeError:
            same = False
        base = entry.get("base") if isinstance(entry, dict) else 0
        sha = entry.get("sha256") if isinstance(entry, dict) else None
        if not same or not (base is None or isinstance(base, str)) or not isinstance(sha, str):
            raise FilesError(f"Манифест повреждён: запись {key!r}.")
    return data


def read_diff(run_id: str, root: Path | None = None) -> str:
    root = root or PROJECT_ROOT
    try:
        return _read_exact(run_dir_path(run_id, root) / "changes.diff")
    except (OSError, UnicodeDecodeError):
        raise FilesError("changes.diff не читается.") from None


# --- diff ---------------------------------------------------------------------


def write_saved_diff(run_id: str, text: str, root: Path | None = None) -> None:
    """Replace changes.diff with its (re)redacted form; apply validation redacts both sides."""
    _atomic_write(run_dir_path(run_id, root or PROJECT_ROOT) / "changes.diff", text.encode("utf-8"))


def unified_diff(path: str, old: str | None, new: str) -> str:
    """git-style headers; `/dev/null` for a new file; empty string if nothing changed."""
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    if old is not None and old == new:
        return ""
    if old is None and not new_lines:
        return f"--- /dev/null\n+++ b/{path}\n"
    out: list[str] = []
    for line in difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="/dev/null" if old is None else f"a/{path}",
        tofile=f"b/{path}",
        lineterm="\n",
    ):
        out.append(line)
        if line[:1] in " +-" and not line.startswith(("---", "+++")) and not line.endswith("\n"):
            out.append("\n\\ No newline at end of file\n")
    return "".join(out)


def _read_exact(path: Path) -> str:
    """UTF-8 text with line endings untouched (Path.read_text has no newline= before 3.13)."""
    return path.read_bytes().decode("utf-8")


def _build_diff(run_dir: Path, files: dict[str, Any]) -> str:
    """Displayed diff, redacted; staged bytes on disk stay exact."""
    return redact(_build_diff_raw(run_dir, files))


def _build_diff_raw(run_dir: Path, files: dict[str, Any]) -> str:
    parts: list[str] = []
    for path in sorted(files):
        entry = files[path]
        new = _read_exact(run_dir / "files" / path)
        old = None if entry["base"] is None else _read_exact(run_dir / "base" / path)
        parts.append(unified_diff(path, old, new))
    return "".join(parts)


def _count_lines(old: str | None, new: str) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.ndiff((old or "").splitlines(), new.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


# --- workspace ----------------------------------------------------------------


def _safe_subdir_file(base: Path, rel: str) -> Path:
    """Create parents of base/rel one by one, refusing links; return the file path."""
    if scope.is_link(base):
        raise FilesError("Служебный каталог прогона заменён ссылкой.")
    current = base
    current.mkdir(exist_ok=True)
    parts = rel.split("/")
    for part in parts[:-1]:
        current = current / part
        current.mkdir(exist_ok=True)
        if scope.is_link(current):
            raise FilesError("Служебный каталог прогона заменён ссылкой.")
    return current / parts[-1]


class Workspace:
    """Tool operations over repo files plus the run's changeset overlay."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else PROJECT_ROOT
        self.run_id = validate_run_id(run_id)
        self.run_dir = run_dir_path(run_id, self.root)
        self._tracked: Tracked | None = None

    # internals

    @property
    def tracked(self) -> Tracked:
        if self._tracked is None:
            self._tracked = scope.load_tracked(self.root)
        return self._tracked

    def invalidate(self) -> None:
        self._tracked = None

    def manifest(self) -> dict[str, Any]:
        return load_manifest(self.run_id, self.root)

    @staticmethod
    def _staged_key(files: dict[str, Any], canon: str) -> str | None:
        fold = canon.casefold()
        return next((key for key in files if key.casefold() == fold), None)

    def _read_staged(self, key: str) -> str:
        path = self.run_dir / "files" / key
        if scope.is_link(self.run_dir / "files") or scope.is_link(path):
            raise FilesError("Файл changeset заменён ссылкой.")
        try:
            data = path.read_bytes()
        except OSError:
            raise FilesError(f"Файл changeset недоступен: {key}") from None
        return scope.check_text_bytes(data, key)

    def _disk_bytes(self, real: str) -> bytes:
        """Tracked file content with every read-policy gate applied."""
        scope.check_readable_name(real)
        if self.tracked.is_symlink(real):
            raise ScopeError(f"Символические ссылки не читаются: {real}")
        target = scope.assert_inside_root(self.root, real)
        try:
            if target.stat().st_size > scope.MAX_READ_BYTES:
                raise ScopeError(f"Файл больше 1 MiB: {real}")
            return target.read_bytes()
        except OSError:
            raise FilesError(f"Файл недоступен: {real}") from None

    def _current(self, path: str) -> tuple[str, str, int]:
        """(real path, text, byte size) of the changeset version, else the disk version."""
        canon = scope.canonical(path)
        files = self.manifest()["files"]
        key = self._staged_key(files, canon)
        if key is not None:
            text = self._read_staged(key)
            return key, text, len(text.encode("utf-8"))
        real = self.tracked.lookup(canon)
        if real is None:
            raise FilesError(f"Файл не найден или вне области (не tracked): {canon}")
        data = self._disk_bytes(real)
        return real, scope.check_text_bytes(data, real), len(data)

    def _candidates(self, files: dict[str, Any]) -> list[str]:
        seen: dict[str, str] = {}
        for path in self.tracked.paths():
            if not scope.is_denied(path):
                seen[path.casefold()] = path
        for key in files:
            seen[key.casefold()] = key
        return sorted(seen.values(), key=str.casefold)

    @staticmethod
    def _match(path: str, pattern: str | None) -> bool:
        return pattern is None or fnmatch.fnmatchcase(path.casefold(), pattern.casefold())

    # tools

    def list_files(self, pattern: str = "*") -> str:
        if not isinstance(pattern, str) or not pattern or len(pattern) > scope.MAX_PATH_CHARS:
            raise FilesError("Некорректный pattern.")
        pattern = pattern.replace("\\", "/")
        matched = [p for p in self._candidates(self.manifest()["files"]) if self._match(p, pattern)]
        shown = matched[:LIST_LIMIT]
        tail = f"всего: {len(matched)}"
        if len(matched) > LIST_LIMIT:
            tail += f", показано {LIST_LIMIT}"
        return redact("\n".join([*shown, tail]))

    def search_text(self, query: str, glob: str | None = None, regex: bool = False) -> str:
        if not isinstance(query, str) or not query:
            raise FilesError("Пустой запрос.")
        if regex:
            if len(query) > REGEX_MAX:
                raise FilesError(f"Шаблон regex длиннее {REGEX_MAX} символов.")
            try:
                rx = re.compile(query, re.IGNORECASE)
            except re.error as exc:
                raise FilesError(f"Невалидный regex: {exc}") from None
            test: Callable[[str], bool] = lambda line: rx.search(line) is not None  # noqa: E731
        else:
            needle = query.casefold()
            test = lambda line: needle in line.casefold()  # noqa: E731
        glob = glob.replace("\\", "/") if isinstance(glob, str) and glob else None
        files = self.manifest()["files"]
        hits: list[str] = []
        total = matched_files = scanned = scanned_bytes = 0
        truncated = False
        for path in self._candidates(files):
            if not self._match(path, glob):
                continue
            if scanned >= SEARCH_FILES:
                truncated = True
                break
            try:
                real, text, size = self._current(path)
            except AdventError:
                continue  # binary, non-UTF-8, oversize or unreadable: skipped silently
            if scanned_bytes + size > SEARCH_BYTES:
                truncated = True
                break
            scanned += 1
            scanned_bytes += size
            found = False
            for number, line in enumerate(text.splitlines(), 1):
                if test(line):
                    found = True
                    total += 1
                    if len(hits) < SEARCH_HITS:
                        hits.append(f"{real}:{number}: {line.strip()[:SEARCH_LINE_CHARS]}")
            matched_files += found
        tail = f"совпадений: {total} в файлах: {matched_files}"
        if total > SEARCH_HITS:
            tail += f", показано {SEARCH_HITS}"
        if truncated:
            tail += f"; скан обрезан (просканировано файлов: {scanned}, байт: {scanned_bytes})"
        return redact("\n".join([*hits, tail]))

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        real, text, _ = self._current(path)
        lines = text.splitlines()
        total = len(lines)
        start = 1 if start_line is None else start_line
        if type(start) is not int or start < 1:
            raise FilesError("start_line должен быть целым числом >= 1.")
        if end_line is not None and (type(end_line) is not int or end_line < start):
            raise FilesError("end_line должен быть целым числом >= start_line.")
        if total and start > total:
            raise FilesError(f"start_line за пределами файла (строк: {total}).")
        last = min(total, start + READ_LINES - 1, end_line if end_line is not None else total)
        out: list[str] = []
        used = 0
        shown_last = start - 1
        for number in range(start, last + 1):
            row = f"{number}: {lines[number - 1]}"
            if used + len(row) + 1 > READ_CHARS and out:
                break
            out.append(row[:READ_CHARS])
            used += len(row) + 1
            shown_last = number
        header = f"{real}: строки {start}-{shown_last} из {total}"
        if shown_last < min(total, end_line if end_line is not None else total):
            header += f" (ответ обрезан лимитом, продолжай со строки {shown_last + 1})"
        return redact("\n".join([header, *out]))

    def write_file(self, path: str, content: str) -> str:
        return self._stage(scope.canonical(path), content)

    def edit_file(self, path: str, old: str, new: str) -> str:
        if not isinstance(old, str) or not old:
            raise FilesError("old не может быть пустым.")
        if not isinstance(new, str):
            raise FilesError("new должен быть строкой.")
        real, text, _ = self._current(path)
        pattern = old
        count = text.count(pattern)
        if count == 0 and "\r\n" in text and "\n" in old:
            pattern = old.replace("\r\n", "\n").replace("\n", "\r\n")
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
            count = text.count(pattern)
        if count != 1:
            raise FilesError(f"old должен встречаться ровно 1 раз, найдено вхождений: {count}.")
        return self._stage(real, text.replace(pattern, new, 1))

    # changeset

    def _stage(self, canon: str, text: str) -> str:
        scope.check_write_name(canon)
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            raise FilesError("Содержимое не кодируется в UTF-8.") from None
        scope.check_write_size(data)
        manifest = self.manifest()
        files: dict[str, Any] = manifest["files"]
        key = self._staged_key(files, canon)
        old_text: str | None
        base_bytes: bytes | None = None
        if key is not None:
            entry = files[key]
            base_sha = entry["base"]
            old_text = None if base_sha is None else _read_exact(self.run_dir / "base" / key)
        else:
            if len(files) >= scope.MAX_WRITE_FILES:
                raise FilesError(
                    f"За прогон можно изменить не более {scope.MAX_WRITE_FILES} файлов."
                )
            real = self.tracked.lookup(canon)
            if real is not None:
                scope.check_write_name(real)
                base_bytes = self._disk_bytes(real)
                old_text = scope.check_text_bytes(base_bytes, real)
                key, base_sha = real, sha256_hex(base_bytes)
            else:
                target = self.root / canon
                if os.path.lexists(target):
                    raise FilesError(f"Существующий файл не отслеживается git: {canon}")
                parent = canon.casefold().rsplit("/", 1)[0] if "/" in canon else ""
                if parent not in self.tracked.parent_dirs:
                    raise FilesError(
                        f"Новый файл возможен только в каталоге с tracked-файлами: {canon}"
                    )
                scope.assert_inside_root(self.root, canon)
                key, base_sha, old_text = canon, None, None
        scope.assert_inside_root(self.root, key)
        dest = _safe_subdir_file(self.run_dir / "files", key)
        if base_bytes is not None:
            _atomic_write(_safe_subdir_file(self.run_dir / "base", key), base_bytes)
        _atomic_write(dest, data)
        files[key] = {"base": base_sha, "sha256": sha256_hex(data)}
        _atomic_write(self.run_dir / "manifest.json", _dump(manifest).encode("utf-8"))
        _atomic_write(self.run_dir / "changes.diff", self._rebuild_diff(files).encode("utf-8"))
        added, removed = _count_lines(old_text, text)
        summary = f"+{added}" + (f" -{removed}" if old_text is not None else "")
        return redact(f"staged {key} ({summary})")

    def _rebuild_diff(self, files: dict[str, Any]) -> str:
        return _build_diff(self.run_dir, files)

    def diff(self) -> str:
        return self._rebuild_diff(self.manifest()["files"])


# --- apply --------------------------------------------------------------------


def _sha_of_file(path: Path) -> str | None:
    try:
        return sha256_hex(path.read_bytes())
    except FileNotFoundError:
        return None


def _validate_run_for_apply(ws: Workspace, manifest: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, Any] = manifest["files"]
    if not files:
        raise ApplyConflictError("В прогоне нет изменений: применять нечего.")
    if len(files) > scope.MAX_WRITE_FILES:
        raise ApplyConflictError(f"В манифесте больше {scope.MAX_WRITE_FILES} файлов.")
    try:
        for key, entry in files.items():
            scope.check_write_name(key)
            scope.assert_inside_root(ws.root, key)
            if ws.tracked.is_symlink(key):
                raise ScopeError(f"Целевой файл — символическая ссылка: {key}")
            if ws.tracked.lookup(key) is None:
                parent = key.casefold().rsplit("/", 1)[0] if "/" in key else ""
                if parent not in ws.tracked.parent_dirs:
                    raise ScopeError(f"Новый файл вне каталога с tracked-файлами: {key}")
            staged = ws.run_dir / "files" / key
            if scope.is_link(staged) or not staged.is_file():
                raise ScopeError(f"Файл changeset отсутствует или подменён: {key}")
            data = staged.read_bytes()
            scope.check_write_size(data)
            if sha256_hex(data) != entry["sha256"]:
                raise ScopeError(f"Содержимое changeset не совпадает с манифестом: {key}")
            if (
                entry["base"] is not None
                and _sha_of_file(ws.run_dir / "base" / key) != entry["base"]
            ):
                raise ScopeError(f"Копия исходника не совпадает с манифестом: {key}")
        if _build_diff(ws.run_dir, files) != redact(read_diff(ws.run_id, ws.root)):
            raise ScopeError("changes.diff не совпадает с содержимым changeset.")
    except ApplyConflictError:
        raise
    except (ScopeError, FilesError, OSError, UnicodeDecodeError) as exc:
        message = getattr(exc, "message", str(exc))
        raise ApplyConflictError(f"Валидация прогона не пройдена: {message}") from None
    return files


def _check_current(root: Path, key: str, entry: dict[str, Any]) -> str:
    """'ok' to write, 'applied' if disk already holds the new content, else raise."""
    current = _sha_of_file(root / key)
    if current == entry["sha256"] and current != entry["base"]:
        return "applied"
    if entry["base"] is None:
        if current is not None:
            raise ApplyConflictError(f"Файл уже существует и отличается: {key}")
    elif current != entry["base"]:
        raise ApplyConflictError(
            f"Файл изменился после прогона (sha256 не совпадает): {key}"
            if current is not None
            else f"Файл удалён после прогона: {key}"
        )
    return "ok"


def apply_run(
    run_id: str,
    root: Path | None = None,
    *,
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
) -> ApplyResult:
    """Write exactly the saved changeset; verify sha256 before each replace, roll back on failure.

    Not a file-system transaction: a window remains between the check and os.replace.
    """
    root = Path(root) if root is not None else PROJECT_ROOT
    ws = Workspace(run_id, root)
    files = _validate_run_for_apply(ws, ws.manifest())
    states = {key: _check_current(root, key, entry) for key, entry in files.items()}
    if all(state == "applied" for state in states.values()):
        raise ApplyConflictError(
            f"Прогон {run_id} уже применён: файлы совпадают с changeset.", already_applied=True
        )
    if any(state == "applied" for state in states.values()):
        done = ", ".join(k for k, s in states.items() if s == "applied")
        raise ApplyConflictError(f"Прогон применён частично, конфликт состояния: {done}")

    backup_dir = ws.run_dir / "backup"
    written: list[str] = []
    created: list[str] = []
    for key in sorted(files):
        entry = files[key]
        target = root / key
        tmp: Path | None = None
        try:
            scope.assert_inside_root(root, key)
            if _check_current(root, key, entry) != "ok":
                raise ApplyConflictError(f"Файл изменился во время применения: {key}")
            if entry["base"] is not None:
                backup = _safe_subdir_file(backup_dir, key)
                _atomic_write(backup, target.read_bytes())
            # Re-check right before each write: narrows the swap window, closes nothing.
            scope.assert_inside_root(root, key)
            fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
            tmp = Path(tmp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write((ws.run_dir / "files" / key).read_bytes())
            scope.assert_inside_root(root, key)
            replace(tmp, target)
            tmp = None
            written.append(key)
            if entry["base"] is None:
                created.append(key)
        except Exception as exc:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()
            if not written and isinstance(exc, ApplyConflictError):
                raise
            restored, failed = _rollback(root, backup_dir, written, created)
            reason = getattr(exc, "message", str(exc))
            message = f"Сбой записи {key}: {reason}. Откатано файлов: {len(restored)}."
            if failed:
                message += f" Откат не удался для: {', '.join(failed)}."
            raise ApplyRolledBackError(
                message, rolled_back=restored, rollback_failed=failed
            ) from None
    return ApplyResult(written=written)


def _rollback(
    root: Path, backup_dir: Path, written: list[str], created: list[str]
) -> tuple[list[str], list[str]]:
    restored: list[str] = []
    failed: list[str] = []
    for key in reversed(written):
        target = root / key
        try:
            # A link swapped in since the write must not redirect the restore.
            scope.assert_inside_root(root, key)
            if key in created:
                target.unlink()
            else:
                data = (backup_dir / key).read_bytes()
                _atomic_write(target, data)
            restored.append(key)
        except (OSError, ScopeError):
            failed.append(key)
    return restored, failed


def git_status_short(paths: list[str], root: Path | None = None) -> str:
    """`git status --short` limited to the given paths; empty string if git fails."""
    root = root or PROJECT_ROOT
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--", *paths],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""
