"""Week 04 entry point `adventmcp`: connect to an MCP server and list its tools.

stdout is the product (tools table, verification verdict); the initialize
header, raw wire frames and errors go to stderr, as everywhere in this repo.
"""

from __future__ import annotations

import subprocess

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import console
from advent_core.errors import AdventError
from week_04 import client as mcp_client
from week_04.server import TOOL_NAMES

DEFAULT_TIMEOUT = 15.0
# A tools/list frame is kilobytes long; one screen line is what a viewer can read.
RAW_HEAD = 800
RAW_TAIL = 800
RAW_LINE_LIMIT = RAW_HEAD + RAW_TAIL

app = typer.Typer(
    help="Минимальный MCP-клиент: подключается к серверу и печатает его инструменты.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Keeps `tools` a real subcommand (typer collapses a lone command otherwise)."""


def _arg_line(arg: mcp_client.ToolArg) -> str:
    parts = [f"{arg.name}: {arg.type}", "обязательный" if arg.required else "опциональный"]
    if arg.default is not None:
        parts.append(f"по умолчанию {arg.default}")
    if arg.enum:
        parts.append("одно из " + " | ".join(arg.enum))
    return " · ".join(parts)


def tools_table(tools: list[mcp_client.ToolInfo]) -> Table:
    table = Table(title="Инструменты MCP-сервера", title_justify="left")
    table.add_column("Инструмент", no_wrap=True, style="bold")
    table.add_column("Описание")
    table.add_column("Аргументы")
    for tool in tools:
        args = "\n".join(_arg_line(a) for a in tool.args) or "—"
        table.add_row(rich_escape(tool.name), rich_escape(tool.description), rich_escape(args))
    return table


def verify(names: list[str], expected: list[str]) -> tuple[bool, str]:
    """Compare the advertised tool names with the expected ones."""
    got, want = set(names), set(expected)
    matched = len(got & want)
    if got == want:
        return True, f"✓ {matched} из {len(want)} инструментов совпадают с ожидаемыми"
    problems = [f"совпало {matched} из {len(want)}"]
    if want - got:
        problems.append("не хватает: " + ", ".join(sorted(want - got)))
    if got - want:
        problems.append("лишние: " + ", ".join(sorted(got - want)))
    return False, "✗ " + "; ".join(problems)


def _raw_sink(line: str) -> None:
    if len(line) > RAW_LINE_LIMIT:
        # Head and tail: the tool names sit in the middle of a tools/list frame.
        cut = len(line) - RAW_LINE_LIMIT
        line = f"{line[:RAW_HEAD]} … (−{cut} симв.) … {line[-RAW_TAIL:]}"
    console.err.print(rich_escape(line), soft_wrap=True, highlight=False)


def _parse_expect(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


@app.command("tools")
def tools_command(
    server: str | None = typer.Option(
        None,
        "--server",
        help='Команда запуска MCP-сервера по stdio, например "python -m my_server". '
        "По умолчанию — собственный сервер репозитория.",
    ),
    timeout: float = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", help="Секунд на handshake и на каждую страницу tools/list."
    ),
    raw: bool = typer.Option(
        False,
        "--raw/--no-raw",
        help=f"Печатать JSON-RPC кадры обоих направлений в stderr (→ клиент, ← сервер; "
        f"длиннее {RAW_LINE_LIMIT} символов сокращаются: голова и хвост).",
    ),
    expect: str | None = typer.Option(
        None,
        "--expect",
        help="Ожидаемые имена инструментов через запятую. Для собственного сервера "
        "по умолчанию — его три инструмента, для чужого проверки нет.",
    ),
) -> None:
    """Подключиться к MCP-серверу, выполнить handshake и вывести список инструментов."""
    command = mcp_client.split_command(server) if server is not None else None
    used_default = command is None
    if command is None:
        command = mcp_client.default_server_command()
    if expect is not None:
        expected: list[str] | None = _parse_expect(expect)
    else:
        expected = list(TOOL_NAMES) if used_default else None

    shown = "python -m week_04.server" if used_default else subprocess.list2cmdline(command)
    console.note(f"подключаюсь: {rich_escape(shown)}")
    result = mcp_client.connect_and_list(command, timeout=timeout, raw=raw, sink=_raw_sink)

    console.err.print(
        f"[bold]сервер:[/bold] {rich_escape(result.server_name)} "
        f"{rich_escape(result.server_version)} · протокол {rich_escape(result.protocol_version)}",
        highlight=False,
    )
    if result.instructions:
        console.err.print(f"[dim]инструкции: {rich_escape(result.instructions)}[/dim]")

    console.out.print(tools_table(result.tools))
    console.out.print(f"Инструментов получено: {len(result.tools)}")
    if expected is not None:
        ok, verdict = verify([t.name for t in result.tools], expected)
        console.out.print(f"[green]{verdict}[/green]" if ok else f"[bold red]{verdict}[/bold red]")
        if not ok:
            raise typer.Exit(1)


def main() -> None:
    """Entry point `adventmcp`: errors as text, not a traceback."""
    console.force_utf8()
    try:
        app()
    except AdventError as error:
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


# `python -m week_04.cli` is how the demo recorder launches this (Step.module).
if __name__ == "__main__":
    main()
