"""Domain and LangGraph state models for the 311 agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class Ticket(BaseModel):
    """Input ticket fields received from the 311 intake channel."""

    description: str = Field(..., min_length=1)
    address: str = Field(..., min_length=1)


class TicketCreationResult(BaseModel):
    """Normalized result from creating a ticket in a department MCP server."""

    created: bool = Field(..., description="Whether the department MCP tool created a ticket.")
    scope: str | None = Field(default=None, description="Ticketing system scope, such as department or internal.")
    department: str = Field(..., description="Department where ticket creation was attempted.")
    ticket_id: str | None = Field(default=None, description="Created ticket identifier, when available.")
    status: str | None = Field(default=None, description="Ticket status selected from MCP tool metadata.")
    tool_name: str | None = Field(default=None, description="MCP tool used to create the ticket.")
    message: str = Field(..., description="Short status message from the ticket creation flow.")

    @field_validator("ticket_id", mode="before")
    @classmethod
    def normalize_ticket_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value)


class InternalTicketContext(BaseModel):
    """Context passed to internal ticket creation after department routing."""

    department_mcp_server_found: bool = Field(
        ...,
        description="Whether the classified department had a registered MCP ticket server.",
    )
    department_ticket_created: bool = Field(
        ...,
        description="Whether a department ticket was successfully created.",
    )
    department_ticket_skipped_reason: str | None = Field(
        default=None,
        description="Reason department ticket creation was skipped, when applicable.",
    )
    classification_confidence: float | None = Field(
        default=None,
        description="Confidence returned by the classification step.",
    )
    classification_confidence_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold below which classification is considered low confidence.",
    )
    classification_confidence_low: bool = Field(
        ...,
        description="Whether classification confidence was missing or below threshold.",
    )


class DepartmentDecision(BaseModel):
    """Normalized routing decision returned by the orchestrator."""

    department: str = Field(..., description="Department that should receive the ticket.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score from 0 to 1.",
    )
    category: str | None = Field(
        default=None,
        description="Optional service request category or subtype.",
    )
    reasoning: str = Field(
        ...,
        description="Short explanation citing the ticket facts and MCP tool findings.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of MCP tools used while reaching the decision.",
    )
    ticket_creation: TicketCreationResult | None = Field(
        default=None,
        description="Department ticket creation result when a department MCP server is configured.",
    )
    internal_ticket_creation: TicketCreationResult | None = Field(
        default=None,
        description="Internal 311 system ticket creation result from the primary MCP server.",
    )


class AgentState(TypedDict):
    """LangGraph state shared between the model node and tool node."""

    messages: Annotated[list[BaseMessage], add_messages]
