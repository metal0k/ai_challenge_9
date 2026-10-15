"""Path canonicalisation and read/write policy for the day-34 file assistant.

Pure checks live here; nothing touches a changeset. Every guard compares casefolded
names because Windows and macOS file systems do.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from advent_core.errors import AdventError

MAX_PATH_CHARS = 260
MAX_READ_BYTES = 1024 * 1024
MAX_WRITE_BYTES = 64 * 1024
MAX_WRITE_FILES = 5
BINARY_SNIFF_BYTES = 8192
WRITE_EXTENSIONS = (".md", ".txt")
DENY_PATTERNS = (".env*", "*.pem", "*.key", "*.sqlite3", "uv.lock")
FORBIDDEN_PREFIXES = (".github/", "specs/", "logs/", "data/")
FORBIDDEN_NAMES = frozenset({"claude.md", "agents.md"})
ROOT_FORBIDDEN = frozenset({"readme.md"})

_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_BAD_CHARS = frozenset('<>"|?*')
# 8.3 aliases such as GIT~1 would bypass the casefolded `.git` guard.
_SHORT_NAME = re.compile(r"~\d")
_REGULAR_MODES = frozenset({"100644", "100755"})


class ScopeError(AdventError):
    """A path outside the assistant's allowed scope."""


def canonical(path: object) -> str:
    """Return a safe repo-relative POSIX path or raise ScopeError."""
    if not isinstance(path, str) or not path:
        raise ScopeError("Путь должен быть непустой строкой.")
    if len(path) > MAX_PATH_CHARS:
        raise ScopeError(f"Путь длиннее {MAX_PATH_CHARS} символов.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        raise ScopeError("Путь содержит управляющие символы.")
    if any(ch in _BAD_CHARS for ch in path):
        raise ScopeError("Путь содержит недопустимые символы.")
    text = path.replace("\\", "/")
    if text.startswith("/"):
        raise ScopeError("Абсолютные пути, UNC и device-префиксы не разрешены.")
    if ":" in text:
        raise ScopeError("Двоеточие в пути не разрешено (drive или alternate data stream).")
    parts = text.split("/")
    for part in parts:
        if part == "":
            raise ScopeError("Пустой компонент пути.")
        if part in (".", ".."):
            raise ScopeError("Компоненты '.' и '..' не разрешены.")
        if part.endswith((".", " ")):
            raise ScopeError("Компонент пути не может заканчиваться точкой или пробелом.")
        if part.split(".")[0].rstrip(" ").upper() in _RESERVED:
            raise ScopeError(f"Зарезервированное имя Windows: {part!r}.")
        if _SHORT_NAME.search(part):
            raise ScopeError(f"Подозрительное короткое имя 8.3: {part!r}.")
    return "/".join(parts)


def is_denied(canon: str) -> bool:
    """`.git` anywhere in the path, or a secret-looking file name."""
    parts = canon.casefold().split("/")
    if ".git" in parts:
        return True
    if any(fnmatch.fnmatchcase(part, ".env*") for part in parts):
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pat) for pat in DENY_PATTERNS[1:])


def check_readable_name(canon: str) -> None:
    if is_denied(canon):
        raise ScopeError(f"Чтение запрещено политикой: {canon}")


def check_write_name(canon: str) -> None:
    """Extension, deny-list and forbidden locations; no file system access."""
    fold = canon.casefold()
    if is_denied(canon):
        raise ScopeError(f"Запись запрещена политикой: {canon}")
    if not fold.endswith(WRITE_EXTENSIONS):
        raise ScopeError("Записывать можно только файлы .md и .txt.")
    if fold.startswith(FORBIDDEN_PREFIXES):
        raise ScopeError(f"Запись в этот каталог запрещена: {canon}")
    if fold.rsplit("/", 1)[-1] in FORBIDDEN_NAMES or fold in ROOT_FORBIDDEN:
        raise ScopeError(f"Запись в этот файл запрещена: {canon}")


def check_write_size(data: bytes) -> None:
    if len(data) > MAX_WRITE_BYTES:
        raise ScopeError(f"Файл больше {MAX_WRITE_BYTES // 1024} KiB.")
    if b"\0" in data:
        raise ScopeError("Содержимое содержит NUL.")


def is_link(path: Path | str) -> bool:
    """Symlink, junction or any other reparse point (lstat, never follows)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(
        getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def assert_inside_root(root: Path, canon: str) -> Path:
    """Every existing component below root is a plain entry, and the real path stays inside."""
    current = root
    for part in canon.split("/"):
        current = current / part
        if os.path.lexists(current) and is_link(current):
            raise ScopeError(f"Символическая ссылка или reparse point в пути: {canon}")
    real_root = Path(os.path.realpath(root))
    try:
        Path(os.path.realpath(current)).relative_to(real_root)
    except ValueError:
        raise ScopeError(f"Путь выходит за пределы репозитория: {canon}") from None
    return current


@dataclass(slots=True)
class Tracked:
    """`git ls-files -s -z` snapshot, indexed by casefolded path."""

    by_fold: dict[str, str] = field(default_factory=dict)
    symlinks: set[str] = field(default_factory=set)
    parent_dirs: set[str] = field(default_factory=set)

    def lookup(self, canon: str) -> str | None:
        return self.by_fold.get(canon.casefold())

    def is_symlink(self, canon: str) -> bool:
        return canon.casefold() in self.symlinks

    def paths(self) -> list[str]:
        return [self.by_fold[key] for key in self.by_fold if key not in self.symlinks]


def load_tracked(root: Path) -> Tracked:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-s", "-z"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScopeError(f"Не удалось запустить git: {exc}") from None
    if proc.returncode != 0:
        raise ScopeError("git ls-files завершился с ошибкой: это не git-репозиторий?")
    tracked = Tracked()
    for record in proc.stdout.split(b"\0"):
        if not record or b"\t" not in record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode = meta.split(b" ", 1)[0].decode("ascii", "replace")
        path = raw_path.decode("utf-8", "replace")
        fold = path.casefold()
        if mode == "120000":
            tracked.symlinks.add(fold)
        elif mode not in _REGULAR_MODES:
            continue  # submodule gitlinks are not files
        tracked.by_fold.setdefault(fold, path)
        parent = fold.rsplit("/", 1)[0] if "/" in fold else ""
        tracked.parent_dirs.add(parent)
    return tracked


def check_text_bytes(data: bytes, canon: str) -> str:
    """Binary / non-UTF-8 / oversize gate; returns the decoded text."""
    if len(data) > MAX_READ_BYTES:
        raise ScopeError(f"Файл больше 1 MiB: {canon}")
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        raise ScopeError(f"Бинарный файл: {canon}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ScopeError(f"Файл не в UTF-8: {canon}") from None
