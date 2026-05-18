"""High-level orchestration for classifying 311 tickets."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agent311.config import load_config
from agent311.mcp_tools import load_mcp_tools
from agent311.models import DepartmentDecision, Ticket
from agent311.prompts import ticket_prompt
from agent311.response_parser import parse_department_decision
from agent311.routing_graph import build_routing_graph


logger = logging.getLogger(__name__)


class TicketRoutingAgent:
    """Coordinates MCP tools, LangGraph execution, and final decision parsing."""

    def __init__(
        self,
        *,
        mcp_url: str,
        mcp_transport: str,
        model: str,
        openai_api_key: str | None,
        recursion_limit: int,
        classification_tool_name: str | None,
    ) -> None:
        self.mcp_url = mcp_url
        self.mcp_transport = mcp_transport
        self.model = model
        self.openai_api_key = openai_api_key
        self.recursion_limit = recursion_limit
        self.classification_tool_name = classification_tool_name

    async def classify(self, ticket: Ticket) -> DepartmentDecision:
        logger.info("Starting 311 ticket classification")
        logger.info("Ticket address: %s", ticket.address)

        tools = await load_mcp_tools(
            mcp_url=self.mcp_url,
            mcp_transport=self.mcp_transport,
            classification_tool_name=self.classification_tool_name,
        )
        graph = build_routing_graph(
            tools=tools,
            model=self.model,
            openai_api_key=self.openai_api_key,
        )

        logger.info("LangGraph compiled; invoking agent")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=ticket_prompt(ticket))]},
            {"recursion_limit": self.recursion_limit},
        )
        logger.info("LangGraph invocation finished")

        final_message = result["messages"][-1]
        return parse_department_decision(final_message)


async def classify_ticket(
    *,
    description: str,
    address: str,
    mcp_url: str | None = None,
    mcp_transport: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
    recursion_limit: int | None = None,
    classification_tool_name: str | None = None,
    config_path: str | None = None,
) -> DepartmentDecision:
    """Classify a 311 ticket using LangGraph and tools from an HTTP MCP server."""

    config = load_config(config_path)
    logger.info("Loaded configuration")
    logger.info(
        "Resolved settings: mcp_url=%s mcp_transport=%s model=%s recursion_limit=%s "
        "classification_tool_name=%s openai_api_key_configured=%s",
        mcp_url or config.mcp_url,
        mcp_transport or config.mcp_transport,
        model or config.model,
        recursion_limit or config.recursion_limit,
        classification_tool_name or config.classification_tool_name or "any",
        bool(openai_api_key or config.openai_api_key),
    )

    agent = TicketRoutingAgent(
        mcp_url=mcp_url or config.mcp_url,
        mcp_transport=mcp_transport or config.mcp_transport,
        model=model or config.model,
        openai_api_key=openai_api_key or config.openai_api_key,
        recursion_limit=recursion_limit or config.recursion_limit,
        classification_tool_name=classification_tool_name or config.classification_tool_name,
    )
    return await agent.classify(Ticket(description=description, address=address))
