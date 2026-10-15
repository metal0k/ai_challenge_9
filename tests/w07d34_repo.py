"""Shared fixture helpers for the day-34 file-assistant tests (not collected by pytest)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def git(root: Path, *args: str, stdin: bytes | None = None) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "-c", "core.autocrlf=false", *args],
        input=stdin,
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode("utf-8", "replace").strip()


def write(root: Path, rel: str, data: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


def make_repo(root: Path) -> Path:
    """A tiny tracked tree with one file for every policy branch."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    write(root, "README.md", "# Root\n")
    write(root, "CLAUDE.md", "rules\n")
    write(root, "docs/a.md", "alpha\nBeta line\ngamma\n")
    write(root, "docs/b.txt", "text file\n")
    write(root, "src/app.py", "def call_tool_once():\n    return 1\n\n\ncall_tool_once()\n")
    write(root, "src/other.py", "from app import call_tool_once\nvalue = call_tool_once()\n")
    write(root, ".env", "MISTRAL_API_KEY=supersecretvalue\n")
    write(root, "certs/server.pem", "-----BEGIN-----\n")
    write(root, "uv.lock", "lock\n")
    write(root, "blob.bin", b"abc\0def")
    write(root, "latin.md", b"\xff\xfe broken")
    write(root, "big.txt", "x" * (1024 * 1024 + 10))
    write(root, "specs/spec.md", "spec\n")
    write(root, ".github/wf.md", "wf\n")
    write(root, "data/d.md", "data\n")
    write(root, "crlf.md", b"one\r\ntwo\r\n")
    git(root, "add", "-A", "-f")
    blob = git(root, "hash-object", "-w", "--stdin", stdin=b"docs/a.md")
    git(root, "update-index", "--add", "--cacheinfo", f"120000,{blob},linked.md")
    return root


def make_dir_link(link: Path, target: Path) -> bool:
    """Symlink or Windows junction to a directory; False if the OS refuses."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    return False
