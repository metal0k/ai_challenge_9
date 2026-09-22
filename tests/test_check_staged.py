"""tools/check_staged.py — guard against personal data reaching the public repo.

Runs the real script against a real staged tree (an isolated repo, not this
project's own), so the assertions cover the actual `git show :file` path
instead of a double of it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "check_staged.py"


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    # cp1252 (this machine's locale) cannot decode the script's own Cyrillic
    # stdout/stderr when subprocess.run does the decoding itself; the script
    # reconfigures its own streams to UTF-8, but only once it starts running.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    return repo


def _stage(repo: Path, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True)


def test_folder_link_is_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "spec.md", "папка со всеми записями: https://yadi.sk/d/AbCdEfGhIjKlMn\n")
    result = _run(repo)
    assert result.returncode == 1
    assert "персональная ссылка на папку Яндекс.Диска" in result.stderr
    assert "spec.md:1" in result.stderr


def test_disk_yandex_ru_folder_link_is_blocked_too(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "notes.md", "https://disk.yandex.ru/d/AbCdEfGhIjKlMn\n")
    result = _run(repo)
    assert result.returncode == 1
    assert "персональная ссылка на папку Яндекс.Диска" in result.stderr


def test_single_file_submission_link_is_not_blocked(tmp_path: Path) -> None:
    # The submission link `advent submit` publishes and README prints per
    # day is deliberately public — it must never trip the guard (2026-09-22:
    # this exact shape blocked `docs: publish w04d16 links` before the fix).
    repo = _repo(tmp_path)
    _stage(
        repo,
        "README.md",
        "| 04 | 16 | Подключение MCP | [`w04d16`](https://x) | "
        "[Yandex Disk](https://yadi.sk/i/rmFbB4QmtGj1Fw) |\n",
    )
    result = _run(repo)
    assert result.returncode == 0
    assert "личных данных нет" in result.stdout


def test_disk_yandex_ru_single_file_link_is_not_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "README.md", "[Yandex Disk](https://disk.yandex.ru/i/Udf4MNU_ocQaAw)\n")
    result = _run(repo)
    assert result.returncode == 0


def test_windows_path_is_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "config.example", "VIDEO_DIR=D:\\Media\\Study\n")
    result = _run(repo)
    assert result.returncode == 1
    assert "локальный путь Windows" in result.stderr


def test_self_referential_files_are_exempt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "CLAUDE.md", "guard blocks yadi.sk/d/ and D:\\ paths\n")
    result = _run(repo)
    assert result.returncode == 0


def test_chat_dump_pattern_is_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "week_04/tasks_04.md", "[21.09.2026 14:03] уточнение из чата\n")
    result = _run(repo)
    assert result.returncode == 1
    assert "дамп чата с датами реплик" in result.stderr


def test_clean_tree_passes_with_zero_exit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "week_04/README.md", "uv run adventmcp tools\n")
    result = _run(repo)
    assert result.returncode == 0
    assert result.stdout.strip() == "проверено файлов: 1 — личных данных нет"
