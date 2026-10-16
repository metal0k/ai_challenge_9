import pytest

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
