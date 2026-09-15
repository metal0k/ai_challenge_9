"""Interactive, deterministic, offline Day 11 memory demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.console import Console

from advent_core.console import force_utf8
from advent_core.memory import (
    MemoryDelta,
    MemoryOperation,
    MemorySnapshot,
    MemoryStore,
    ShortTermMemory,
    apply_delta,
    manual_set,
    memory_messages,
    render_memory,
)
from advent_core.session import Session, Turn

EVIDENCE = (
    "Подготовь migration с zero downtime; отвечай на русском prose с English "
    "technical terms; code word ORBIT."
)
CODE_WORD = "ORBIT"
COMMANDS = (
    "/turn",
    "/memory",
    "/switch B",
    "/switch A",
    "/pin-conflict",
    "/credential",
    "/new",
    "/storage",
)


class Demo:
    """Stateful operator surface used by both interactive and dry-run modes."""

    def __init__(self, root: Path, *, no_color: bool = False) -> None:
        self.console = Console(
            force_terminal=not no_color,
            color_system="standard" if not no_color else None,
            legacy_windows=False,
        )
        self.store = MemoryStore(root / "logs")
        self.session = Session.new("session-a", directory=root / "logs" / "sessions")
        self.a = MemorySnapshot(
            short_term=ShortTermMemory(
                ({"role": "user", "content": f"Кодовое слово: {CODE_WORD}. {EVIDENCE}"},), 0, 1
            )
        )
        self.b = MemorySnapshot()

    def out(self, text: str = "", style: str | None = None) -> None:
        # Memory keys and labels use square brackets; disable Rich markup so
        # the request blocks are displayed literally on camera.
        self.console.print(text, style=style, markup=False)

    def heading(self, text: str) -> None:
        self.out(f"\n━━ {text} ━━", "bold cyan")

    @staticmethod
    def answer(snapshot: MemorySnapshot, label: str) -> str:
        # Build the same protected request that the production agent sends;
        # the answer is intentionally derived from assembled messages rather
        # than reading three snapshot fields directly.
        request = memory_messages(snapshot)
        assembled = "\n".join(str(message.get("content", "")) for message in request)
        language = (
            "Russian prose with English technical terms"
            if "preferences.answer_language" in assembled
            else "<empty>"
        )
        goal = "Prepare the migration" if "goal.primary" in assembled else "<empty>"
        word = CODE_WORD if CODE_WORD in assembled else "<empty>"
        return f"{label}: language={language}; goal={goal}; code_word={word}"

    def turn(self, user_text: str | None = None) -> None:
        self.heading("1. Один turn: extractor → apply_delta → fake model")
        user_text = user_text or EVIDENCE
        if CODE_WORD not in user_text:
            user_text = user_text.rstrip(" .") + "; code word ORBIT."
        self.out("ты › " + user_text, "yellow")
        update = apply_delta(
            self.a,
            MemoryDelta(
                working=(
                    MemoryOperation("set", "goal", "primary", "Prepare the migration", user_text),
                    MemoryOperation("set", "constraints", "downtime", "zero downtime", user_text),
                ),
                long_term=(
                    MemoryOperation(
                        "set",
                        "preferences",
                        "answer_language",
                        "Russian prose with English technical terms",
                        user_text,
                    ),
                ),
            ),
            user_messages=({"role": "user", "content": user_text},),
            memory_upto=1,
        )
        self.a = update.snapshot
        self.store.save_working("session-a", self.a.working, 1)
        self.store.save_long_term(self.a.long_term)
        self.session.turns.append(Turn("user", user_text, datetime.now(UTC).isoformat()))
        self.session.turns.append(
            Turn("assistant", self.answer(self.a, "all layers"), datetime.now(UTC).isoformat())
        )
        self.session.save()
        self.out("✓ extractor: delta подтверждён exact user evidence", "green")
        self.out("✓ apply_delta: working + goal/constraint; long-term + preference", "green")
        self.out(self.answer(self.a, "assistant видит все три layers"), "bold white")
        order = " → ".join(m["content"].splitlines()[0] for m in memory_messages(self.a))
        self.out("request order: " + order, "magenta")

    def memory(self) -> None:
        self.heading("2. /memory: смотрим persisted layers")
        self.out("$ /memory", "yellow")
        self.out(render_memory(self.a), "white")

    def switch(self, target: str) -> None:
        target = target.upper()
        self.heading(f"3. /switch {target}: local layers и global layer")
        self.out(f"$ /switch {target}", "yellow")
        if target == "B":
            self.b = MemorySnapshot(long_term=self.store.load_long_term().value)
            self.out("session B загружена; working и short-term пусты", "cyan")
            self.out(self.answer(self.b, "B answer (только global)"), "bold white")
        else:
            self.a = MemorySnapshot(
                self.a.short_term,
                self.store.load_working("session-a", 1).value,
                self.store.load_long_term().value,
            )
            self.out("session A восстановлена из working file + transcript", "cyan")
            self.out(self.answer(self.a, "A answer (все layers восстановлены)"), "bold white")

    def pin_conflict(self) -> None:
        self.heading("4. Manual pin блокирует automatic conflict")
        self.out("$ /pin-conflict", "yellow")
        pinned = manual_set(self.a, "working", "goal.primary", "Pinned release goal")
        result = apply_delta(
            pinned,
            MemoryDelta(
                working=(
                    MemoryOperation(
                        "set", "goal", "primary", "Conflicting goal", "Prepare the migration"
                    ),
                )
            ),
            user_messages=({"role": "user", "content": "Prepare the migration"},),
        )
        self.a = result.snapshot
        self.out("manual set → goal.primary = Pinned release goal [PINNED]", "green")
        self.out("automatic set → goal.primary = Conflicting goal", "red")
        self.out("result: !goal.primary blocked by pin", "bold magenta")

    def credential(self) -> None:
        self.heading("5. Privacy boundary: credential отклонён")
        self.out("$ /credential", "yellow")
        try:
            apply_delta(
                self.a,
                MemoryDelta(
                    long_term=(
                        MemoryOperation(
                            "set",
                            "knowledge",
                            "api_token",
                            "secret-123",
                            "Remember api_token secret-123",
                        ),
                    )
                ),
                user_messages=({"role": "user", "content": "Remember api_token secret-123"},),
            )
        except ValueError as exc:
            self.out("rejected: " + str(exc).split(": ", 1)[-1], "bold red")

    def new(self) -> None:
        self.heading("6. /new: очищаем short-term + working, сохраняем global")
        self.out("$ /new", "yellow")
        self.a = MemorySnapshot(long_term=self.store.load_long_term().value)
        self.store.save_working("session-a", self.a.working, 0)
        self.session.turns.clear()
        self.session.save()
        self.out(self.answer(self.a, "new session"), "bold white")
        self.out("✓ global preference сохранён; session layers пусты", "green")

    def storage(self) -> None:
        self.heading("7. Storage locations: реальные JSON files")
        files = (
            self.session.path,
            self.store.working_path("session-a"),
            self.store.long_term_path,
        )
        for path in files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if path == self.session.path:
                    detail = f"turns={len(raw.get('turns', []))}"
                elif path == self.store.working_path("session-a"):
                    detail = f"entries={len(raw.get('entries', {}))}, upto={raw.get('upto')}"
                else:
                    detail = f"entries={len(raw.get('entries', {}))}"
                self.out(f"✓ {path.relative_to(self.store.root.parent)} ({detail})", "magenta")
            except (OSError, ValueError) as exc:
                self.out(f"! {path}: {exc}", "red")

    def dispatch(self, command: str) -> bool:
        command = command.strip()
        if not command or command.startswith("#"):
            return True
        if command in ("/exit", "/quit"):
            return False
        self.out("operator › " + command, "bold yellow")
        if command.startswith("/turn"):
            self.turn(command.partition(" ")[2] or None)
        elif command == "/memory":
            self.memory()
        elif command.startswith("/switch"):
            self.switch(command.split(maxsplit=1)[1] if " " in command else "A")
        elif command == "/pin-conflict":
            self.pin_conflict()
        elif command == "/credential":
            self.credential()
        elif command == "/new":
            self.new()
        elif command == "/storage":
            self.storage()
        else:
            self.out(f"unknown command: {command} (use /exit)", "red")
        return True


def run() -> None:
    force_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run the fixed offline scenario")
    parser.add_argument(
        "--interactive", action="store_true", help="read staged commands from UTF-8 stdin"
    )
    parser.add_argument("--no-color", action="store_true", help="disable Rich/ANSI colours")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="memory-demo-") as directory:
        demo = Demo(Path(directory), no_color=args.no_color)
        if args.interactive:
            demo.out("Демонстрация memory Day 11 — staged commands; /exit для выхода", "bold cyan")
            for raw in sys.stdin.buffer:
                if not demo.dispatch(raw.decode("utf-8").rstrip("\r\n")):
                    break
        else:
            for command in COMMANDS:
                if not demo.dispatch(command):
                    break


if __name__ == "__main__":
    run()
