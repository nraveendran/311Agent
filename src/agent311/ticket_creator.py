"""Create department tickets through department-level MCP tools."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent311.models import DepartmentDecision, InternalTicketContext, Ticket, TicketCreationResult
from agent311.mcp_tools import load_mcp_tools
from agent311.routing_graph import build_tool_graph


logger = logging.getLogger(__name__)


async def create_department_ticket(
    *,
    ticket: Ticket,
    decision: DepartmentDecision,
    mcp_url: str,
    mcp_transport: str,
    model: str,
    openai_api_key: str | None,
    recursion_limit: int,
) -> TicketCreationResult:
    """Use department-level MCP tools to create a service ticket."""

    tools = await load_mcp_tools(
        mcp_url=mcp_url,
        mcp_transport=mcp_transport,
        required_tool_name=None,
        server_name="department_ticketing",
    )
    return await create_ticket_with_tools(
        ticket=ticket,
        decision=decision,
        tools=tools,
        model=model,
        openai_api_key=openai_api_key,
        recursion_limit=recursion_limit,
        scope="department",
        system_name=decision.department,
        existing_ticket_creation=None,
        internal_context=None,
    )


async def create_internal_ticket(
    *,
    ticket: Ticket,
    decision: DepartmentDecision,
    department_ticket_creation: TicketCreationResult | None,
    internal_context: InternalTicketContext,
    mcp_url: str,
    mcp_transport: str,
    model: str,
    openai_api_key: str | None,
    recursion_limit: int,
) -> TicketCreationResult:
    """Use the primary MCP server to create an internal 311 tracking ticket."""

    tools = await load_mcp_tools(
        mcp_url=mcp_url,
        mcp_transport=mcp_transport,
        required_tool_name=None,
        server_name="internal_ticketing",
    )
    return await create_ticket_with_tools(
        ticket=ticket,
        decision=decision,
        tools=tools,
        model=model,
        openai_api_key=openai_api_key,
        recursion_limit=recursion_limit,
        scope="internal",
        system_name="internal 311 system",
        existing_ticket_creation=department_ticket_creation,
        internal_context=internal_context,
    )


async def create_ticket_with_tools(
    *,
    ticket: Ticket,
    decision: DepartmentDecision,
    tools,
    model: str,
    openai_api_key: str | None,
    recursion_limit: int,
    scope: str,
    system_name: str,
    existing_ticket_creation: TicketCreationResult | None,
    internal_context: InternalTicketContext | None,
) -> TicketCreationResult:
    graph = build_tool_graph(
        tools=tools,
        model=model,
        openai_api_key=openai_api_key,
        system_content=ticket_creation_system_prompt(scope=scope, system_name=system_name),
        require_tool_result=True,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=ticket_creation_prompt(
                        ticket=ticket,
                        decision=decision,
                        scope=scope,
                        system_name=system_name,
                        existing_ticket_creation=existing_ticket_creation,
                        internal_context=internal_context,
                    )
                )
            ]
        },
        {"recursion_limit": recursion_limit},
    )
    final_content = result["messages"][-1].content
    return parse_ticket_creation_result(
        final_content,
        fallback_scope=scope,
        fallback_department=decision.department,
    )


def ticket_creation_system_prompt(*, scope: str, system_name: str) -> str:
    return (
        f"You are creating a 311 service ticket in the {system_name}. "
        "Use the available MCP ticketing tool to create the ticket. Do not claim a ticket "
        "was created unless an MCP tool call created it. Return valid JSON only with keys: "
        "created, scope, department, ticket_id, status, tool_name, message. "
        f"The scope value must be {scope!r}."
    )


def ticket_creation_prompt(
    *,
    ticket: Ticket,
    decision: DepartmentDecision,
    scope: str,
    system_name: str,
    existing_ticket_creation: TicketCreationResult | None,
    internal_context: InternalTicketContext | None,
) -> str:
    existing_ticket_text = (
        existing_ticket_creation.model_dump_json()
        if existing_ticket_creation is not None
        else "None"
    )
    internal_context_text = (
        internal_context.model_dump_json()
        if internal_context is not None
        else "None"
    )
    return (
        f"Create a {scope} ticket in the {system_name} for this classified 311 request.\n\n"
        f"Department: {decision.department}\n"
        f"Category: {decision.category or 'Unknown'}\n"
        f"Confidence: {decision.confidence if decision.confidence is not None else 'Unknown'}\n"
        f"Routing reasoning: {decision.reasoning}\n\n"
        f"Citizen complaint description: {ticket.description}\n"
        f"Address: {ticket.address}\n\n"
        f"Existing department ticket creation result: {existing_ticket_text}\n\n"
        f"Internal ticket creation context: {internal_context_text}\n\n"
        "You must call the appropriate MCP ticket creation tool. Use the tool metadata, "
        "including tool descriptions and input parameters attached to the tool call, to populate valid fields. "
        "Carefully inspect the input parameters to determine how status should be created. "
        "Do not invent status values when the tool parameters define how status should be set. "
        "Return only JSON."
    )


def parse_ticket_creation_result(
    content: object,
    *,
    fallback_scope: str | None = None,
    fallback_department: str = "Unknown",
) -> TicketCreationResult:
    if not isinstance(content, str):
        content = json.dumps(content)

    try:
        payload = _extract_json(str(content))
        return TicketCreationResult.model_validate(payload)
    except json.JSONDecodeError:
        logger.warning("Ticket creation response was not parseable JSON")
        return TicketCreationResult(
            created=False,
            scope=fallback_scope,
            department=fallback_department,
            ticket_id=None,
            status=None,
            tool_name=None,
            message=f"Ticket tool ran, but final response was not parseable JSON: {str(content).strip()}",
        )
    except (TypeError, ValidationError) as error:
        logger.warning("Ticket creation response did not match expected schema: %s", error)
        return TicketCreationResult(
            created=False,
            scope=fallback_scope,
            department=fallback_department,
            ticket_id=None,
            status=None,
            tool_name=None,
            message=f"Ticket tool ran, but final response did not match expected schema: {error}",
        )


def _extract_json(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)
