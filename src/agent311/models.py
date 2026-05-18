"""Domain and LangGraph state models for the 311 agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Input ticket fields received from the 311 intake channel."""

    description: str = Field(..., min_length=1)
    address: str = Field(..., min_length=1)


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


class AgentState(TypedDict):
    """LangGraph state shared between the model node and tool node."""

    messages: Annotated[list[BaseMessage], add_messages]
