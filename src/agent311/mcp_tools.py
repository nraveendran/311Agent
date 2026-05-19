"""MCP tool loading and validation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


logger = logging.getLogger(__name__)


async def load_mcp_tools(
    *,
    mcp_url: str,
    mcp_transport: str,
    classification_tool_name: str | None = None,
    required_tool_name: str | None = None,
    server_name: str = "department_classifier",
):
    required_name = required_tool_name if required_tool_name is not None else classification_tool_name
    logger.info("Connecting to MCP server: name=%s url=%s transport=%s", server_name, mcp_url, mcp_transport)
    connections: dict[str, Any] = {
        server_name: {
            "url": mcp_url,
            "transport": mcp_transport,
        }
    }
    mcp_client = MultiServerMCPClient(
        connections
    )

    try:
        tools = await mcp_client.get_tools()
    except Exception:
        logger.error("Failed to load MCP tools from %s", mcp_url)
        raise

    validate_mcp_tools(
        tools=tools,
        mcp_url=mcp_url,
        required_tool_name=required_name,
    )
    logger.info("Discovered MCP tool(s): %s", ", ".join(tool.name for tool in tools))
    return tools


def validate_mcp_tools(*, tools, mcp_url: str, required_tool_name: str | None) -> None:
    if not tools:
        logger.error("No MCP tools were discovered at %s", mcp_url)
        raise RuntimeError(f"No MCP tools were discovered at {mcp_url}.")

    if not required_tool_name:
        logger.info("No required MCP tool name configured; accepting discovered MCP tools")
        return

    tool_names = {tool.name for tool in tools}
    if required_tool_name in tool_names:
        logger.info("Required MCP tool is available: %s", required_tool_name)
        return

    available_tools = ", ".join(sorted(tool_names))
    logger.error(
        "Required MCP tool is missing: required=%s available=%s",
        required_tool_name,
        available_tools or "none",
    )
    raise RuntimeError(
        f"Required MCP tool '{required_tool_name}' was not discovered. "
        f"Available tools: {available_tools or 'none'}."
    )
