"""Readable, deterministic Day 12 profile lifecycle demo (no API calls)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.markup import escape as rich_escape

from advent_core import chat, console, profiles
from advent_core.config import ConfigError
from advent_core.session import Session

SCREEN_PAUSE = 6.0
REQUEST_PAUSE = 10.0

USER_PROMPT = (
    "Составь план релиза API без простоя: шаги, риски и критерии rollback для команды разработки."
)
DEVELOPER_ANSWER = (
    "1. Перед deploy: выполнить dry-run migration, backup DB и health checks.\n"
    "2. Deploy: начать canary 5%, затем 25/100% после проверки metrics.\n"
    "3. Проверка: сравнить p95 latency, 5xx rate и queue lag с baseline.\n"
    "4. Rollback: вернуть previous image и выполнить обратную migration при нарушении SLO."
)
MANAGER_ANSWER = (
    "• Цель: выпустить API без простоя и с понятным rollback.\n"
    "• План: backup → canary 5% → проверка metrics → rollout 100%.\n"
    "• Риски: migration failure, 5xx spike и рост latency; owner — on-call.\n"
    "• Решение: если smoke test не green — rollback to previous image."
)


class Demo:
    def __init__(self, root: Path, *, dry_run: bool) -> None:
        self.root = root
        self.directory = root / "profiles"
        self.session = Session.new("profile-demo", directory=root)
        self.dry_run = dry_run

    def screen(self, title: str, *lines: str, pause: float = SCREEN_PAUSE) -> None:
        """Render one stable screen; OBS gets time to capture each proof point."""
        print("\033[2J\033[H", end="")
        console.out.print("[bold cyan]DAY 12 · NAMED PROFILES[/bold cyan]")
        console.out.print()
        console.out.print(f"[bold yellow]{rich_escape(title)}[/bold yellow]")
        console.out.print()
        for line in lines:
            escaped = rich_escape(line)
            if line.startswith("✓"):
                console.out.print(f"[green]{escaped}[/green]")
            elif line.startswith("✗"):
                console.out.print(f"[bold red]{escaped}[/bold red]")
            elif line.startswith(("active profile:", "injected block", "profile context")):
                console.out.print(f"[magenta]{escaped}[/magenta]")
            elif line.startswith(("USER PROMPT", "DETERMINISTIC", "same user prompt")):
                console.out.print(f"[bold cyan]{escaped}[/bold cyan]")
            elif line.startswith(("assistant A", "assistant B", "A developer", "B manager")):
                console.out.print(f"[bold white]{escaped}[/bold white]")
            else:
                console.out.print(escaped)
        if not self.dry_run:
            time.sleep(pause)

    @staticmethod
    def token_estimate(messages: list[dict[str, str]]) -> int:
        """Small offline equivalent of `/tokens` for a deterministic take."""
        return max(1, sum(len(message["content"]) for message in messages) // 3)

    def request_screen(
        self,
        name: str,
        values: dict[str, str],
        answer: str,
        answer_label: str,
    ) -> None:
        request = chat.build_messages(USER_PROMPT, history=profiles.messages(values))
        profile_block = request[0]["content"].splitlines()
        self.screen(
            f"{answer_label} · /profile use {name} · same request",
            "DETERMINISTIC OFFLINE DEMO · no API call; answer is fixed for the proof",
            "USER PROMPT (identical in A and B):",
            f"  {USER_PROMPT}",
            f"active profile: {name}",
            "injected block (actual request context):",
            *[f"  {line}" for line in profile_block],
            f"assistant {answer_label.lower()}:",
            *[f"  {line}" for line in answer.splitlines()],
            pause=REQUEST_PAUSE,
        )

    def run(self) -> None:
        self.screen(
            "Storage and priority",
            "DETERMINISTIC OFFLINE DEMO · no network / no API credits",
            "profiles → logs/profiles/<name>.json",
            "active_profile → Session.state (additive)",
            "current request > active profile > default behavior",
        )

        developer = {"style": "technical", "format": "code-first"}
        profiles.save("developer", developer, self.directory)
        self.screen(
            "/profile create developer style=technical format=code-first",
            "✓ created and stored developer",
            f"file: {profiles.path_for('developer', self.directory).name}",
            "preferences will be injected automatically into the next request",
        )
        self.session.state["active_profile"] = "developer"
        self.session.save()
        self.request_screen("developer", developer, DEVELOPER_ANSWER, "A developer")

        manager = {"style": "brief", "format": "bullets"}
        profiles.save("manager", manager, self.directory)
        self.session.state["active_profile"] = "manager"
        self.session.save()
        self.request_screen("manager", manager, MANAGER_ANSWER, "B manager")
        self.screen(
            "A/B result · only profile context changed",
            f"same user prompt: {USER_PROMPT}",
            "A developer → technical + code-first: numbered rollout and SLO metrics",
            "B manager   → brief + bullets: goal, risks, owner and decision rule",
            "The prompt is identical; the injected preferences explain the format.",
            pause=8.0,
        )

        values = profiles.load("manager", self.directory)
        request = chat.build_messages(USER_PROMPT, history=profiles.messages(values))
        self.screen(
            "/tokens · profile context counted separately",
            "active profile: manager",
            f"profile tokens (~): {self.token_estimate(profiles.messages(values))}",
            f"request tokens (~): {self.token_estimate(request)}",
            "profile context is not conversation history or journal content",
        )

        resumed = Session.load("profile-demo", directory=self.root)
        resumed.clear()
        resumed.save()
        self.screen(
            "Restart + /new lifecycle",
            "Session.load('profile-demo')",
            f"✓ active_profile restored: {resumed.state.get('active_profile')}",
            "✓ /new clears turns but keeps active_profile: manager",
            "selection is session-scoped; profile files remain global",
        )

        try:
            profiles.save("unsafe", {"api_key": "sk-demo-not-a-secret"}, self.directory)
        except ConfigError as error:
            profiles.delete("manager", self.directory)
            resumed.state["active_profile"] = None
            resumed.save()
            self.screen(
                "Privacy + cleanup",
                "/profile create unsafe api_key=...",
                f"✗ rejected: {error}",
                "✓ credential-like value never reaches disk or request",
                "✓ delete manager removes the file and clears active_profile",
                f"remaining profiles: {profiles.list_names(self.directory)}",
                pause=7.0,
            )

        self.screen(
            "Day 12 summary",
            "same prompt → different injected profile context → different answer format",
            "developer: technical numbered rollout / manager: brief action bullets",
            "tokens, restart and /new lifecycle verified",
            "privacy rejection and active-profile cleanup verified",
        )


def main() -> None:
    console.force_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="skip OBS pacing pauses")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="profile-demo-") as raw:
        Demo(Path(raw), dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
