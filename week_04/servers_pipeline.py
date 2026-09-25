"""Facade MCP server `pipeline`: Mistral-backed summarizing tools (stdio).

`save_to_file` is deliberately not exposed: in the day-20 registry writing files
is the filesystem server's job. Run: `python -m week_04.servers_pipeline`.
"""

from __future__ import annotations

from week_04 import facade

SERVER_NAME = "advent-pipeline-facade"
TOOL_NAMES = ("summarize_text", "commit_digest")

mcp = facade.build(
    SERVER_NAME,
    "Text summarizing and the commit-digest pipeline.",
    TOOL_NAMES,
)

if __name__ == "__main__":
    mcp.run()
