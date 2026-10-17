import os
import subprocess
import sys
from pathlib import Path

import pytest

from adventship import gates
from adventship.ctx import ShipError
from adventship.gates import confirm, tty_gate


def test_no_tty_fails_with_exit_8_and_never_reads():
    def read(_):
        raise AssertionError("must not prompt without a tty")

    with pytest.raises(ShipError) as caught:
        tty_gate("push?", "details", is_tty=lambda: False, read=read)
    assert caught.value.exit_code == 8


@pytest.mark.parametrize("answer", ["да", "Да", " y ", "YES", "д"])
def test_yes_answers_pass(answer):
    assert tty_gate("q", "d", is_tty=lambda: True, read=lambda _: answer) is True


@pytest.mark.parametrize("answer", ["нет", "", "n", "yes please", "ok"])
def test_anything_else_is_refusal_exit_9(answer):
    with pytest.raises(ShipError) as caught:
        tty_gate("q", "d", is_tty=lambda: True, read=lambda _: answer)
    assert caught.value.exit_code == 9


def test_eof_is_refusal():
    def read(_):
        raise EOFError

    with pytest.raises(ShipError) as caught:
        tty_gate("q", "d", is_tty=lambda: True, read=read)
    assert caught.value.exit_code == 9


def test_default_tty_check_uses_real_streams(monkeypatch):
    class NoTty:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NoTty())
    with pytest.raises(ShipError) as caught:
        tty_gate("q", "d")
    assert caught.value.exit_code == 8


class _Stream:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def test_real_tty_false_when_isatty_false():
    assert gates._is_real_tty(_Stream(False)) is False


def test_real_tty_false_when_not_a_console(monkeypatch):
    monkeypatch.setattr(gates.os, "name", "nt")
    monkeypatch.setattr(gates, "_is_console_handle", lambda s: False)
    assert gates._is_real_tty(_Stream(True)) is False


def test_real_tty_true_when_console(monkeypatch):
    monkeypatch.setattr(gates.os, "name", "nt")
    monkeypatch.setattr(gates, "_is_console_handle", lambda s: True)
    assert gates._is_real_tty(_Stream(True)) is True


# A true console cannot be provided under pytest, so only the refusals are tested live.
_CHILD = (
    "import sys\n"
    "from adventship.gates import tty_gate\n"
    "from adventship.ctx import ShipError\n"
    "try:\n"
    "    tty_gate('SECRETQUESTION', 'd')\n"
    "except ShipError as e:\n"
    "    sys.exit(e.exit_code)\n"
    "sys.exit(0)\n"
)


@pytest.mark.skipif(os.name != "nt", reason="NUL-as-tty is a Windows quirk")
@pytest.mark.parametrize("stdin", [subprocess.DEVNULL, subprocess.PIPE], ids=["devnull", "pipe"])
def test_child_without_console_stdin_exits_8(stdin):
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        stdin=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=root,
        timeout=60,
    )
    assert proc.returncode == 8, proc.stderr
    assert "SECRETQUESTION" not in proc.stdout + proc.stderr


def test_confirm_turns_falsy_gate_into_exit_9():
    class Ctx:
        gate = staticmethod(lambda q, d: False)

    with pytest.raises(ShipError) as caught:
        confirm(Ctx(), "q", "d")
    assert caught.value.exit_code == 9


def test_confirm_passes_on_true():
    class Ctx:
        gate = staticmethod(lambda q, d: True)

    confirm(Ctx(), "q", "d")


def test_no_yes_flag_exists():
    from typer.testing import CliRunner

    from adventship.cli import app

    result = CliRunner().invoke(app, ["run", "--day", "35", "--yes"])
    assert result.exit_code != 0
