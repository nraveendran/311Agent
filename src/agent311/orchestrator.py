"""High-level orchestration for classifying 311 tickets."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agent311.config import load_config
from agent311.department_registry import load_department_registry
from agent311.mcp_tools import load_mcp_tools
from agent311.models import DepartmentDecision, InternalTicketContext, Ticket
from agent311.prompts import ticket_prompt
from agent311.response_parser import parse_department_decision
from agent311.routing_graph import build_routing_graph
from agent311.ticket_creator import create_department_ticket, create_internal_ticket


logger = logging.getLogger(__name__)
LOW_CONFIDENCE_THRESHOLD = 0.7


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
        mcp_registry_path: str | None,
    ) -> None:
        self.mcp_url = mcp_url
        self.mcp_transport = mcp_transport
        self.model = model
        self.openai_api_key = openai_api_key
        self.recursion_limit = recursion_limit
        self.classification_tool_name = classification_tool_name
        self.mcp_registry_path = mcp_registry_path

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
        decision = parse_department_decision(final_message)
        decision, department_mcp_server_found = await self._create_department_ticket_if_configured(
            ticket,
            decision,
        )
        return await self._create_internal_ticket(
            ticket=ticket,
            decision=decision,
            department_mcp_server_found=department_mcp_server_found,
        )

    async def _create_department_ticket_if_configured(
        self,
        ticket: Ticket,
        decision: DepartmentDecision,
    ) -> tuple[DepartmentDecision, bool]:
        registry = load_department_registry(self.mcp_registry_path)
        department_server = registry.find(decision.department)
        if department_server is None:
            logger.info("No department MCP server registered for department=%s", decision.department)
            return decision, False

        if self._classification_confidence_low(decision):
            logger.info(
                "Skipping department ticket creation because classification confidence is low: "
                "department=%s confidence=%s threshold=%s",
                decision.department,
                decision.confidence,
                LOW_CONFIDENCE_THRESHOLD,
            )
            return decision, True

        logger.info(
            "Department MCP server found: department=%s url=%s transport=%s",
            department_server.department,
            department_server.url,
            department_server.transport,
        )
        ticket_creation = await create_department_ticket(
            ticket=ticket,
            decision=decision,
            mcp_url=department_server.url,
            mcp_transport=department_server.transport,
            model=self.model,
            openai_api_key=self.openai_api_key,
            recursion_limit=self.recursion_limit,
        )
        return decision.model_copy(update={"ticket_creation": ticket_creation}), True

    async def _create_internal_ticket(
        self,
        *,
        ticket: Ticket,
        decision: DepartmentDecision,
        department_mcp_server_found: bool,
    ) -> DepartmentDecision:
        logger.info("Creating internal 311 system ticket through primary MCP server")
        internal_ticket_creation = await create_internal_ticket(
            ticket=ticket,
            decision=decision,
            department_ticket_creation=decision.ticket_creation,
            internal_context=self._internal_ticket_context(
                decision=decision,
                department_mcp_server_found=department_mcp_server_found,
            ),
            mcp_url=self.mcp_url,
            mcp_transport=self.mcp_transport,
            model=self.model,
            openai_api_key=self.openai_api_key,
            recursion_limit=self.recursion_limit,
        )
        return decision.model_copy(update={"internal_ticket_creation": internal_ticket_creation})

    @staticmethod
    def _classification_confidence_low(decision: DepartmentDecision) -> bool:
        return decision.confidence is None or decision.confidence < LOW_CONFIDENCE_THRESHOLD

    def _internal_ticket_context(
        self,
        *,
        decision: DepartmentDecision,
        department_mcp_server_found: bool,
    ) -> InternalTicketContext:
        department_ticket_created = bool(
            decision.ticket_creation is not None and decision.ticket_creation.created
        )
        return InternalTicketContext(
            department_mcp_server_found=department_mcp_server_found,
            department_ticket_created=department_ticket_created,
            department_ticket_skipped_reason=self._department_ticket_skipped_reason(
                decision=decision,
                department_mcp_server_found=department_mcp_server_found,
                department_ticket_created=department_ticket_created,
            ),
            classification_confidence=decision.confidence,
            classification_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
            classification_confidence_low=self._classification_confidence_low(decision),
        )

    def _department_ticket_skipped_reason(
        self,
        *,
        decision: DepartmentDecision,
        department_mcp_server_found: bool,
        department_ticket_created: bool,
    ) -> str | None:
        if department_ticket_created:
            return None
        if not department_mcp_server_found:
            return "no_department_mcp_server"
        if self._classification_confidence_low(decision):
            return "low_classification_confidence"
        return "department_ticket_not_created"


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
    mcp_registry_path: str | None = None,
    config_path: str | None = None,
) -> DepartmentDecision:
    """Classify a 311 ticket using LangGraph and tools from an HTTP MCP server."""

    config = load_config(config_path)
    logger.info("Loaded configuration")
    logger.info(
        "Resolved settings: mcp_url=%s mcp_transport=%s model=%s recursion_limit=%s "
        "classification_tool_name=%s mcp_registry_path=%s openai_api_key_configured=%s",
        mcp_url or config.mcp_url,
        mcp_transport or config.mcp_transport,
        model or config.model,
        recursion_limit or config.recursion_limit,
        classification_tool_name or config.classification_tool_name or "any",
        mcp_registry_path or config.mcp_registry_path or "default",
        bool(openai_api_key or config.openai_api_key),
    )

    agent = TicketRoutingAgent(
        mcp_url=mcp_url or config.mcp_url,
        mcp_transport=mcp_transport or config.mcp_transport,
        model=model or config.model,
        openai_api_key=openai_api_key or config.openai_api_key,
        recursion_limit=recursion_limit or config.recursion_limit,
        classification_tool_name=classification_tool_name or config.classification_tool_name,
        mcp_registry_path=mcp_registry_path or config.mcp_registry_path,
    )
    return await agent.classify(Ticket(description=description, address=address))
