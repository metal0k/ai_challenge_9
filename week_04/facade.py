"""Builds a facade MCP server that re-exposes a subset of `week_04.server` tools.

`@mcp.tool` returns the function unchanged (day-19 fact), so a facade imports the
same callables and registers them on its own server: no logic is duplicated.
Descriptions are read from the original server, not retyped.
"""

from __future__ import annotations

from collections.abc import Sequence

import anyio
from mcp.server import MCPServer

from week_04 import server as base


def _descriptions() -> dict[str, str]:
    tools = anyio.run(base.mcp.list_tools)
    return {tool.name: tool.description or "" for tool in tools}


def build(name: str, instructions: str, tool_names: Sequence[str]) -> MCPServer:
    """New server exposing exactly `tool_names` of `week_04.server`."""
    descriptions = _descriptions()
    facade = MCPServer(name, instructions=instructions, version="0.1.0", log_level="WARNING")
    for tool_name in tool_names:
        # KeyError on a typo is deliberate: a facade must not silently lose a tool.
        facade.tool(description=descriptions[tool_name])(getattr(base, tool_name))
    return facade
