"""Prompt text for the 311 routing agent."""

from __future__ import annotations

from agent311.models import Ticket


def ticket_prompt(ticket: Ticket) -> str:
    return (
        "Classify this 311 ticket.\n\n"
        f"Citizen complaint description: {ticket.description}\n"
        f"Address: {ticket.address}\n\n"
        "You must call the available MCP ticket classification tool before making a routing decision. "
        "Do not classify this ticket from your own knowledge. "
        "Return only JSON with these keys: department, confidence, category, reasoning, tools_used. "
        "The tools_used value must be a JSON array of MCP tool names you called."
    )


def system_prompt() -> str:
    return (
        "You are a 311 customer service routing agent. Your job is to route each ticket "
        "to the correct city department using the complaint description, address, and MCP "
        "classification tools. You must not classify from your own knowledge. If no MCP "
        "classification tool result is available, do not produce a department decision. "
        "The final response must be valid JSON only and must be based on MCP tool output."
    )
