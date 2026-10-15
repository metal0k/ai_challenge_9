"""Stdio MCP server over repo files plus a per-run changeset (day 34).

Started as `python -m adventassistant.files_server --run-id ID`. The pipeline fixes
the run id and never lets the model choose it or the repo root; `--root` exists only
so tests can point the server at a temporary git repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from advent_core.config import load_env, redact
from advent_core.errors import AdventError
from adventassistant.files_store import Workspace

mcp = MCPServer(
    "advent-files",
    instructions=(
        "Поиск, чтение и подготовка изменений в tracked-файлах репозитория. "
        "Записи попадают только в changeset прогона, рабочее дерево не меняется."
    ),
    version="0.1.0",
    log_level="WARNING",
)

_workspace: Workspace | None = None


def configure(run_id: str, root: Path | None = None) -> None:
    """Bind the server to one run; raises AdventError if the run directory is invalid."""
    global _workspace
    _workspace = Workspace(run_id, root)


def _ws() -> Workspace:
    if _workspace is None:
        raise ToolError("Сервер не настроен: нет run-id.")
    return _workspace


def _guard(call):
    try:
        return call()
    except AdventError as exc:
        raise ToolError(redact(exc.message)) from None


@mcp.tool(
    description=(
        "Список файлов репозитория по шаблону fnmatch (регистр не важен, разделитель /), "
        "включая новые файлы текущего прогона. Не более 200 строк и общее число."
    )
)
def list_files(pattern: str = "*") -> str:
    return _guard(lambda: _ws().list_files(pattern))


@mcp.tool(
    description=(
        "Поиск текста по файлам репозитория: строка (без учёта регистра) или regex. "
        "Формат path:line: текст, не более 100 совпадений, затем общее число. "
        "glob ограничивает набор файлов."
    )
)
def search_text(query: str, glob: str | None = None, regex: bool = False) -> str:
    return _guard(lambda: _ws().search_text(query, glob, regex))


@mcp.tool(
    description=(
        "Прочитать файл с номерами строк: не более 400 строк за вызов, в ответе общее "
        "число строк. Видит изменения текущего прогона."
    )
)
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    return _guard(lambda: _ws().read_file(path, start_line, end_line))


@mcp.tool(
    description=(
        "Создать или полностью заменить файл .md/.txt в changeset прогона. "
        "Рабочее дерево не меняется; применяет изменения человек отдельной командой."
    )
)
def write_file(path: str, content: str) -> str:
    return _guard(lambda: _ws().write_file(path, content))


@mcp.tool(
    description=(
        "Заменить в файле ровно одно вхождение old на new (в changeset). "
        "Если old не найден или встречается несколько раз, вернётся ошибка с числом."
    )
)
def edit_file(path: str, old: str, new: str) -> str:
    return _guard(lambda: _ws().edit_file(path, old, new))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adventassistant.files_server")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=None, help="tests only")
    args = parser.parse_args(argv)
    load_env()  # the child does not inherit .env, and redact() reads secrets from the env
    try:
        configure(args.run_id, args.root)
    except AdventError as exc:
        print(redact(exc.message), file=sys.stderr)
        raise SystemExit(2) from None
    mcp.run()


if __name__ == "__main__":
    main()
