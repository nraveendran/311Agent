"""MCP tool loading and validation."""

from __future__ import annotations

import logging

from langchain_mcp_adapters.client import MultiServerMCPClient


logger = logging.getLogger(__name__)


async def load_mcp_tools(
    *,
    mcp_url: str,
    mcp_transport: str,
    classification_tool_name: str | None,
):
    logger.info("Connecting to MCP server: url=%s transport=%s", mcp_url, mcp_transport)
    mcp_client = MultiServerMCPClient(
        {
            "department_classifier": {
                "url": mcp_url,
                "transport": mcp_transport,
            }
        }
    )

    try:
        tools = await mcp_client.get_tools()
    except Exception:
        logger.error("Failed to load MCP tools from %s", mcp_url)
        raise

    validate_mcp_tools(
        tools=tools,
        mcp_url=mcp_url,
        classification_tool_name=classification_tool_name,
    )
    logger.info("Discovered MCP tool(s): %s", ", ".join(tool.name for tool in tools))
    return tools


def validate_mcp_tools(*, tools, mcp_url: str, classification_tool_name: str | None) -> None:
    if not tools:
        logger.error("No MCP tools were discovered at %s", mcp_url)
        raise RuntimeError(f"No MCP tools were discovered at {mcp_url}.")

    if not classification_tool_name:
        logger.info("No required classification tool name configured; accepting discovered MCP tools")
        return

    tool_names = {tool.name for tool in tools}
    if classification_tool_name in tool_names:
        logger.info("Required MCP classification tool is available: %s", classification_tool_name)
        return

    available_tools = ", ".join(sorted(tool_names))
    logger.error(
        "Required MCP classification tool is missing: required=%s available=%s",
        classification_tool_name,
        available_tools or "none",
    )
    raise RuntimeError(
        f"Required MCP classification tool '{classification_tool_name}' was not discovered. "
        f"Available tools: {available_tools or 'none'}."
    )
