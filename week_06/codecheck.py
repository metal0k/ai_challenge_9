"""Hidden tests for model-written `is_palindrome`, run in a child process.

No real sandbox: a static ast gate, a minimal environment, a temp cwd, a
timeout and capped output. The residual risk is named in the README.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT_S = 10.0
OUTPUT_CAP = 64 * 1024
FUNC_NAME = "is_palindrome"

# (group, text, expected)
CASES: tuple[tuple[str, str, bool], ...] = (
    ("латиница", "A man, a plan, a canal: Panama", True),
    ("латиница", "Was it a car or a cat I saw?", True),
    ("кириллица", "А роза упала на лапу Азора", True),
    ("кириллица", "Я иду с мечем судия", True),
    ("негатив", "Привет, мир", False),
    ("негатив", "Мама мыла раму", False),
    ("негатив", "race a car", False),
)
COMPOSITION = "скрытые тесты: 2 латиница, 2 кириллица, 3 негатив"

ALLOWED_IMPORTS = frozenset({"re", "string", "unicodedata", "typing"})
# Any reference is refused, not only a call: `f = open; f(...)` must not pass.
BANNED_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "exit",
        "quit",
        "memoryview",
        "__builtins__",
    }
)
# `re.compile` is a normal regex call; only the bare builtin is dangerous.
SAFE_ATTRS = frozenset({"compile"})

VERDICT_OK = "ok"
VERDICT_PARTIAL = "partial"
VERDICT_ERROR = "error"
VERDICT_NO_CODE = "no_code"
VERDICT_UNSAFE = "unsafe"


@dataclass(slots=True)
class CodeCheck:
    verdict: str
    passed: int = 0
    total: int = len(CASES)
    failed: list[str] = field(default_factory=list)
    # ok | failed | error | none
    own_asserts: str = "none"
    error: str | None = None
    executed: bool = False

    @property
    def label(self) -> str:
        if self.verdict == VERDICT_OK:
            return f"✓ {self.passed}/{self.total}"
        if self.verdict == VERDICT_PARTIAL:
            return f"✗ {self.passed}/{self.total}"
        if self.verdict == VERDICT_NO_CODE:
            return "нет блока кода"
        if self.verdict == VERDICT_UNSAFE:
            return "небезопасный код, не исполнялось"
        return "ошибка исполнения"


_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_PY_LANGS = {"python", "py", "python3"}


_DEF_RE = re.compile(rf"^\s*def\s+{FUNC_NAME}\b", re.M)


def _defines_target(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(_DEF_RE.search(code))
    return any(isinstance(n, ast.FunctionDef) and n.name == FUNC_NAME for n in tree.body)


def extract_code(text: str) -> str | None:
    """Last block defining FUNC_NAME; else last python block; else the only block."""
    blocks = _FENCE_RE.findall(text)
    defining = [
        body
        for lang, body in blocks
        if (lang.lower() in _PY_LANGS or not lang) and _defines_target(body)
    ]
    if defining:
        return defining[-1]
    python = [body for lang, body in blocks if lang.lower() in _PY_LANGS]
    if python:
        return python[-1]
    if len(blocks) == 1:
        return blocks[0][1]
    return None


def unsafe_reason(code: str) -> str | None:
    """Why the code must not run, or None. Raises SyntaxError on bad syntax."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"импорт {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return f"импорт из {node.module or '.'!r}"
        elif isinstance(node, ast.Attribute):
            if "__" in node.attr:
                return f"атрибут {node.attr!r}"
            if node.attr in BANNED_NAMES and node.attr not in SAFE_ATTRS:
                return f"ссылка на {node.attr}"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return f"ссылка на {node.id}"
            if node.id.startswith("__") and node.id != "__name__":
                return f"имя {node.id!r}"
    return None


def _child_env() -> dict[str, str]:
    """Minimal environment: no API keys or anything else from .env."""
    env = {}
    for key in ("SYSTEMROOT", "PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


# Runs inside the child. The result goes to result.json, not stdout: the model's
# own print() must not be able to corrupt the channel.
_HARNESS = r"""
import ast, contextlib, json, sys
CAP = %(cap)d
CASES = %(cases)s

class Cap:
    def __init__(self):
        self.n = 0
    def write(self, s):
        self.n += len(s)
        return len(s)
    def flush(self):
        pass

src = open("candidate.py", encoding="utf-8").read()
has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(src)))
ns = {"__name__": "candidate"}
res = {"passed": 0, "failed": [], "own_asserts": "none", "error": None}
sink = Cap()
with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    try:
        exec(compile(src, "candidate.py", "exec"), ns)
        if has_assert:
            res["own_asserts"] = "ok"
    except AssertionError:
        res["own_asserts"] = "failed"
    except BaseException as exc:
        res["own_asserts"] = "error"
        res["error"] = type(exc).__name__ + ": " + str(exc)[:200]
    fn = ns.get("is_palindrome")
    if not callable(fn):
        if res["error"] is None:
            res["error"] = "функция is_palindrome не определена"
    else:
        for group, text, expected in CASES:
            try:
                r = fn(text)
                ok = isinstance(r, bool) and r == expected
            except BaseException as exc:
                ok = False
            if ok:
                res["passed"] += 1
            else:
                res["failed"].append(text)
res["defined"] = callable(ns.get("is_palindrome"))
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False)
"""


def run_hidden_tests(code: str, *, timeout: float = TIMEOUT_S) -> CodeCheck:
    """Gate, execute in a child, collect result.json."""
    try:
        reason = unsafe_reason(code)
    except SyntaxError as exc:
        return CodeCheck(VERDICT_ERROR, error=f"синтаксическая ошибка: {exc.msg}")
    if reason is not None:
        return CodeCheck(VERDICT_UNSAFE, error=reason)

    cases = [(g, t, e) for g, t, e in CASES]
    harness = _HARNESS % {"cap": OUTPUT_CAP, "cases": repr(cases)}
    with tempfile.TemporaryDirectory(prefix="w06check_") as tmp:
        work = Path(tmp)
        (work / "candidate.py").write_text(code, encoding="utf-8")
        (work / "harness.py").write_text(harness, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "harness.py"],
            cwd=work,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return CodeCheck(VERDICT_ERROR, error=f"timeout {timeout:g} с", executed=True)
        result_path = work / "result.json"
        if not result_path.exists():
            tail = (err or b"")[:OUTPUT_CAP].decode("utf-8", errors="replace").strip()
            tail = tail.splitlines()[-1] if tail else f"код возврата {proc.returncode}"
            return CodeCheck(VERDICT_ERROR, error=tail[:200], executed=True)
        data = json.loads(result_path.read_text(encoding="utf-8"))

    check = CodeCheck(
        VERDICT_ERROR,
        passed=data["passed"],
        failed=data["failed"],
        own_asserts=data["own_asserts"],
        error=data["error"],
        executed=True,
    )
    if not data["defined"]:
        return check
    check.verdict = VERDICT_OK if check.passed == check.total else VERDICT_PARTIAL
    return check


def check_code_answer(text: str) -> CodeCheck:
    """Extract the code block from a model answer and run the hidden tests."""
    code = extract_code(text)
    if code is None:
        return CodeCheck(VERDICT_NO_CODE)
    return run_hidden_tests(code)
