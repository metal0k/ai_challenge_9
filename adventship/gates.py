"""Human gates. There is deliberately no way to auto-confirm: no `--yes`, no env override."""

from __future__ import annotations

import sys
from collections.abc import Callable

from advent_core import console
from adventship.ctx import EXIT_GATE_DENIED, EXIT_NO_TTY, ShipContext, ShipError

YES = {"да", "д", "y", "yes"}


def tty_gate(
    question: str,
    details: str,
    *,
    is_tty: Callable[[], bool] | None = None,
    read: Callable[[str], str] | None = None,
) -> bool:
    """Show `details`, ask `question`. True only on an explicit yes.

    Without a tty nothing is asked and nothing runs (exit 8); any other answer, EOF
    included, is a refusal (exit 9). `is_tty`/`read` are test seams.
    """
    tty_ok = is_tty() if is_tty else (sys.stdin.isatty() and sys.stderr.isatty())
    if not tty_ok:
        raise ShipError(
            "нужен tty для подтверждения",
            exit_code=EXIT_NO_TTY,
            hint="Gate нельзя пропустить: запусти команду в интерактивном терминале.",
        )
    console.err.print(details, markup=False)
    prompt = f"{question} [да/нет]: "
    try:
        answer = (read or input)(prompt)
    except EOFError:
        answer = ""
    if answer.strip().lower() not in YES:
        raise ShipError("отклонено на gate", exit_code=EXIT_GATE_DENIED)
    return True


def confirm(ctx: ShipContext, question: str, details: str) -> None:
    """Stage helper: a falsy gate answer from any gate implementation becomes exit 9."""
    if not ctx.gate(question, details):
        raise ShipError("отклонено на gate", exit_code=EXIT_GATE_DENIED)
