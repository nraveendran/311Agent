"""FastAPI application for invoking the 311 routing agent."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent311.config import load_config
from agent311.models import DepartmentDecision
from agent311.orchestrator import classify_ticket


class ClassifyTicketRequest(BaseModel):
    """REST request body for a 311 ticket."""

    description: str = Field(..., min_length=1)
    address: str = Field(..., min_length=1)


class ClassifyTicketResponse(BaseModel):
    """REST response with a user-facing explanation and raw decision details."""

    summary: str
    decision: DepartmentDecision


load_dotenv()
config = load_config()
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(
    title="311 Agent API",
    description="Classify 311 tickets and create internal and department tickets through MCP tools.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tickets", response_model=ClassifyTicketResponse)
async def create_ticket(request: ClassifyTicketRequest) -> ClassifyTicketResponse:
    try:
        decision = await classify_ticket(
            description=request.description,
            address=request.address,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return ClassifyTicketResponse(
        summary=build_user_summary(decision),
        decision=decision,
    )


def build_user_summary(decision: DepartmentDecision) -> str:
    confidence_text = (
        f"{decision.confidence:.1%}"
        if decision.confidence is not None
        else "unknown"
    )
    category_text = f" in the {decision.category} category" if decision.category else ""
    parts = [
        f"I classified the ticket to {decision.department}{category_text} with {confidence_text} confidence."
    ]

    context = decision.internal_ticket_context
    department_ticket = decision.ticket_creation
    if context is None:
        parts.append("I did not receive internal context about department ticket handling.")
    elif department_ticket and department_ticket.created:
        parts.append(
            "Because the confidence was high enough and this department has an MCP server, "
            f"I created a department ticket with ID {department_ticket.ticket_id or 'unknown'}"
            f" and status {department_ticket.status or 'unknown'}."
        )
    elif not context.department_mcp_server_found:
        parts.append(
            "I did not create a department ticket because this department does not have a registered MCP server."
        )
    elif context.classification_confidence_low:
        parts.append(
            "I did not create a department ticket because the classification confidence was below "
            f"the {context.classification_confidence_threshold:.0%} threshold."
        )
    else:
        parts.append("I attempted department ticket handling, but no department ticket was created.")

    internal_ticket = decision.internal_ticket_creation
    if internal_ticket and internal_ticket.created:
        parts.append(
            f"I also created an internal 311 ticket with ID {internal_ticket.ticket_id or 'unknown'}"
            f" and status {internal_ticket.status or 'unknown'}."
        )
    elif internal_ticket:
        parts.append(f"I attempted to create an internal 311 ticket, but it was not created: {internal_ticket.message}")
    else:
        parts.append("I did not receive a result for internal 311 ticket creation.")

    return " ".join(parts)
