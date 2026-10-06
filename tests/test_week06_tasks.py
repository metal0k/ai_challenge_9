"""week_06/tasks.py: loader and the per-task fact check (no network)."""

from __future__ import annotations

import pytest

from week_06 import tasks


@pytest.mark.parametrize(
    "text",
    [
        "Канберра",
        "Канберра.",
        "**Канберра**",
        " `Канберра` ",
        "«Канберра»",
        "canberra!",
        "КАНБЕРРА",
    ],
)
def test_fact_accepts(text):
    result = tasks.check_task("fact", text)
    assert result.verdict == "ok" and result.ok


@pytest.mark.parametrize(
    "text",
    ["Сидней", "Не Канберра, а Сидней", "Канберра или Сидней", "", "Столица — Канберра"],
)
def test_fact_rejects(text):
    result = tasks.check_task("fact", text)
    assert result.verdict == "wrong" and not result.ok


def test_load_tasks_order_and_alice_prompt():
    loaded = tasks.load_tasks()
    assert list(loaded) == ["fact", "alice", "palindrome"]
    alice = loaded["alice"].prompt
    assert alice.startswith("У Алисы три брата и две сестры.")
    assert "ОТВЕТ" in alice
    # no persona, no hint at the trap
    assert "палиндром" not in loaded["fact"].prompt


def test_alice_check_uses_marker():
    assert tasks.check_task("alice", "рассуждение\nОТВЕТ: 3").verdict == "ok"
    assert tasks.check_task("alice", "ОТВЕТ: 2").verdict == "wrong"
    assert tasks.check_task("alice", "просто три").verdict == "no_marker"


def test_palindrome_dispatch_runs_hidden_tests():
    body = "def is_palindrome(t):\n    s = [c.lower() for c in t if c.isalnum()]\n"
    body += "    return s == s[::-1]"
    good = f"```python\n{body}\n```"
    result = tasks.check_task("palindrome", good)
    assert result.ok and result.label == "✓ 7/7"
    assert tasks.check_task("palindrome", "нет кода").verdict == "no_code"


def test_parse_task_ids():
    loaded = tasks.load_tasks()
    assert tasks.parse_task_ids("fact, alice", loaded) == ["fact", "alice"]
    with pytest.raises(tasks.ConfigError):
        tasks.parse_task_ids("fact,nope", loaded)


def test_unknown_task_has_no_check():
    with pytest.raises(tasks.ConfigError):
        tasks.check_task("nope", "x")
