"""Facade MCP server `repo`: read-only repository tools (stdio).

Run: `python -m week_04.servers_repo`. stdout carries the protocol only.
"""

from __future__ import annotations

from week_04 import facade

SERVER_NAME = "advent-repo-facade"
TOOL_NAMES = ("list_days", "get_task", "git_log", "count_tokens")

mcp = facade.build(
    SERVER_NAME,
    "Read-only tools about the AI Advent coursework repository.",
    TOOL_NAMES,
)

if __name__ == "__main__":
    mcp.run()
