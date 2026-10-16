"""Stage `checks` runs its children with a sanitized environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from advent_core.config import SECRET_KEYS
from adventship import checks


def _ctx(tmp_path: Path, calls: list[dict]):
    def run(argv, *, timeout, env=None, inherit_env=True, input=None):
        calls.append({"argv": argv, "env": env, "inherit_env": inherit_env})
        return subprocess.CompletedProcess(argv, 0, "", "")

    return SimpleNamespace(root=tmp_path, run=run)


def test_children_get_sanitized_env_without_secrets(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\nCOURSE_SHEET_URL=https://example.invalid/x\nCUSTOM_FROM_DOTENV=1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COURSE_SHEET_URL", "https://example.invalid/x")
    monkeypatch.setenv("CUSTOM_FROM_DOTENV", "1")
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 20)
    monkeypatch.setenv("GH_TOKEN", "t" * 20)
    monkeypatch.setenv("ADVENT_OFFLINE", "1")
    monkeypatch.setenv("KEEP_ME_VAR", "keep")
    before = dict(os.environ)
    calls: list[dict] = []

    result = checks.run_checks(_ctx(tmp_path, calls))

    assert len(calls) == 4
    assert [c["argv"][-1] for c in calls[:1]] == ["."]
    for call in calls:
        assert call["inherit_env"] is False
        env = call["env"]
        for key in ("COURSE_SHEET_URL", "CUSTOM_FROM_DOTENV", "MISTRAL_API_KEY", "GH_TOKEN"):
            assert key not in env
        for key in SECRET_KEYS:
            assert key not in env
        assert "ADVENT_OFFLINE" not in env
        assert env["KEEP_ME_VAR"] == "keep"
        assert env["PATH"] == os.environ["PATH"]
    assert "pytest -q" in result.summary
    assert dict(os.environ) == before


def test_sanitized_env_without_dotenv_file_still_drops_secret_keys(tmp_path):
    base = {"PATH": "/bin", "GITHUB_TOKEN": "x" * 20, "OTHER": "1"}
    assert checks.sanitized_env(tmp_path, base) == {"PATH": "/bin", "OTHER": "1"}
