"""Week 4's own bit: the path to its MCP server, plus a re-export of the transport.

The transport itself (spawn/handshake/list/call, error translation) lives in
`advent_core.mcp_client` — it is not specific to this week's server.
"""

from __future__ import annotations

import sys

from advent_core.mcp_client import (
    ListResult,
    ToolArg,
    ToolInfo,
    acall_tool,
    aconnect_and_list,
    call_tool_once,
    connect_and_list,
    split_command,
    summarize_args,
)

__all__ = [
    "ListResult",
    "ToolArg",
    "ToolInfo",
    "acall_tool",
    "aconnect_and_list",
    "call_tool_once",
    "connect_and_list",
    "default_server_command",
    "split_command",
    "summarize_args",
]


def default_server_command() -> list[str]:
    return [sys.executable, "-m", "week_04.server"]
