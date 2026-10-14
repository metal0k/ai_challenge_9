"""Read-only stdio MCP server for the fictional support fixture."""

from __future__ import annotations

import json

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from advent_core.errors import AdventError
from adventassistant.support_store import get_support_context as read_support_context

mcp = MCPServer(
    "advent-support",
    instructions="Scoped, read-only profile and ticket lookup for the local support demo.",
    version="0.1.0",
    log_level="WARNING",
)


@mcp.tool(description="Получить профиль выбранного пользователя и только его тикет.")
def get_support_context(user_id: str, ticket_id: str | None = None) -> str:
    try:
        context = read_support_context(user_id, ticket_id)
    except AdventError as exc:
        raise ToolError(exc.message) from None
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    mcp.run()
