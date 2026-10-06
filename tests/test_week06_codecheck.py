"""week_06/codecheck.py: hidden tests run in real child processes (no network)."""

from __future__ import annotations

import pytest

from week_06 import codecheck as cc

GOOD = """
def is_palindrome(text: str) -> bool:
    s = [c.lower() for c in text if c.isalnum()]
    return s == s[::-1]
"""

ASCII_ONLY = """
import re

def is_palindrome(text):
    s = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return s == s[::-1]
"""


def fenced(code: str, lang: str = "python") -> str:
    return f"Вот решение:\n```{lang}\n{code}\n```\nГотово."


def test_correct_function_passes_all_seven():
    result = cc.check_code_answer(fenced(GOOD))
    assert (result.verdict, result.passed, result.total) == ("ok", 7, 7)
    assert result.failed == []
    assert result.own_asserts == "none"


def test_ascii_only_regex_is_exactly_five_of_seven():
    result = cc.check_code_answer(fenced(ASCII_ONLY))
    assert result.verdict == "partial"
    assert (result.passed, result.total) == (5, 7)
    # empty string is a palindrome, so both Cyrillic negatives fail
    assert result.failed == ["Привет, мир", "Мама мыла раму"]
    assert result.label == "✗ 5/7"


def test_case_composition_is_stated():
    assert len(cc.CASES) == 7
    assert "2 латиница, 2 кириллица, 3 негатив" in cc.COMPOSITION


def test_no_code():
    assert cc.check_code_answer("просто текст").verdict == "no_code"
    two_plain = "```\na\n```\n```\nb\n```"
    assert cc.check_code_answer(two_plain).verdict == "no_code"


def test_last_python_block_wins_and_single_plain_block_accepted():
    text = fenced("def is_palindrome(t):\n    return False") + fenced(GOOD)
    assert cc.check_code_answer(text).verdict == "ok"
    assert cc.check_code_answer(fenced(GOOD, lang="")).verdict == "ok"


USAGE = "print(is_palindrome('abba'))\nprint(is_palindrome('abc'))"


def test_function_block_then_usage_block_picks_function():
    text = fenced(GOOD) + fenced(USAGE)
    assert "def is_palindrome" in cc.extract_code(text)
    result = cc.check_code_answer(text)
    assert (result.verdict, result.passed, result.total) == ("ok", 7, 7)


def test_usage_block_first_function_block_second():
    code = cc.extract_code(fenced(USAGE) + fenced(GOOD))
    assert "def is_palindrome" in code and "print(" not in code


def test_two_defining_blocks_last_wins():
    first = "def is_palindrome(t):\n    return False"
    text = fenced(first) + fenced(GOOD)
    assert cc.extract_code(text).strip() == GOOD.strip()


def test_no_function_falls_back_to_last_python_block():
    text = fenced("x = 1") + fenced("y = 2")
    assert cc.extract_code(text).strip() == "y = 2"
    assert cc.extract_code(fenced("z = 3", lang="")).strip() == "z = 3"


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef is_palindrome(t): return True",
        "from subprocess import run\ndef is_palindrome(t): return True",
        "def is_palindrome(t):\n    open('x', 'w')\n    return True",
        "def is_palindrome(t):\n    return eval('True')",
        "def is_palindrome(t):\n    return t.__class__ is str",
        # references, not only calls (review #1)
        "f = open\nf('x', 'w')\ndef is_palindrome(t): return True",
        "e = eval\ndef is_palindrome(t): return e('True')",
        "[exec][0]('1')\ndef is_palindrome(t): return True",
        "__builtins__\ndef is_palindrome(t): return True",
        "g = getattr\ndef is_palindrome(t): return True",
    ],
)
def test_unsafe_code_is_not_executed(code, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not spawn")

    monkeypatch.setattr(cc.subprocess, "Popen", boom)
    result = cc.run_hidden_tests(code)
    assert result.verdict == "unsafe" and not result.executed


def test_alias_to_open_does_not_write_outside(tmp_path):
    marker = tmp_path / "alias.txt"
    code = f"f = open\nf({str(marker)!r}, 'w')\ndef is_palindrome(t): return True"
    assert cc.run_hidden_tests(code).verdict == "unsafe"
    assert not marker.exists()


def test_re_compile_attribute_stays_allowed():
    code = (
        "import re\nP = re.compile(r'[^a-z]')\ndef is_palindrome(t):\n"
        "    s = P.sub('', t.lower())\n    return s == s[::-1]"
    )
    assert cc.unsafe_reason(code) is None


def test_unsafe_does_not_touch_disk(tmp_path, monkeypatch):
    marker = tmp_path / "pwned.txt"
    code = f"def is_palindrome(t):\n    open({str(marker)!r}, 'w')\n    return True"
    assert cc.run_hidden_tests(code).verdict == "unsafe"
    assert not marker.exists()


def test_infinite_loop_times_out():
    result = cc.run_hidden_tests("while True:\n    pass\n", timeout=1.5)
    assert result.verdict == "error" and "timeout" in result.error


def test_syntax_error_is_error_not_run():
    result = cc.run_hidden_tests("def is_palindrome(:\n")
    assert result.verdict == "error" and not result.executed


def test_function_missing():
    result = cc.run_hidden_tests("x = 1\n")
    assert result.verdict == "error" and "is_palindrome" in result.error


def test_own_asserts_none_failed_ok():
    assert cc.run_hidden_tests(GOOD).own_asserts == "none"
    ok = cc.run_hidden_tests(GOOD + "\nassert is_palindrome('aba')\n")
    assert ok.own_asserts == "ok" and ok.verdict == "ok"
    bad = cc.run_hidden_tests(GOOD + "\nassert not is_palindrome('aba')\n")
    # failed own assert does not hide the hidden tests
    assert bad.own_asserts == "failed" and bad.verdict == "ok" and bad.passed == 7


def test_assert_before_function_still_finds_function():
    code = "assert 1 == 2\n" + GOOD
    result = cc.run_hidden_tests(code)
    # exec stopped at the assert: the function never got defined
    assert result.own_asserts == "failed"
    assert result.verdict == "error"


def test_exception_outside_assert_after_function():
    code = GOOD + "\nraise ValueError('boom')\n"
    result = cc.run_hidden_tests(code)
    assert result.own_asserts == "error" and result.verdict == "ok"
    assert "ValueError" in result.error


def test_model_print_does_not_break_result():
    code = GOOD + "\nprint('{\"passed\": 99}')\nprint(is_palindrome('aba'))\n"
    result = cc.run_hidden_tests(code)
    assert result.verdict == "ok" and result.passed == 7


def test_child_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    env = cc._child_env()
    assert "MISTRAL_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert set(env) <= {"SYSTEMROOT", "PATH"}


def test_popen_receives_the_minimal_env(monkeypatch):
    """The env actually handed to Popen, not just what _child_env() returns."""
    monkeypatch.setenv("MISTRAL_API_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    seen = {}
    real = cc.subprocess.Popen

    def spy(*args, **kwargs):
        seen["env"] = kwargs.get("env")
        return real(*args, **kwargs)

    monkeypatch.setattr(cc.subprocess, "Popen", spy)
    assert cc.run_hidden_tests(GOOD).verdict == "ok"
    assert seen["env"] is not None
    assert "MISTRAL_API_KEY" not in seen["env"] and "GITHUB_TOKEN" not in seen["env"]
    assert set(seen["env"]) <= {"SYSTEMROOT", "PATH"}
