# AI-Powered 311 Service Routing & Ticket Creation Platform

## Overview

This project explores how AI agents can improve municipal 311 workflows by
classifying citizen-reported issues, routing them to the right department, and
creating follow-up tickets through MCP-enabled operational systems.

The platform is designed for complaints that may be vague, incomplete, or
written in everyday language, such as:

- "There is peeling paint and broken windows on an abandoned building."
- "Trash has not been picked up for two weeks."
- "Streetlight is out near the alley."
- "There is a large pothole in the right lane."

For each request, the agent can produce:

- Responsible department
- Service category
- Classification confidence
- Internal 311 tracking ticket
- Department-specific ticket when a department MCP server is available
- A user-friendly explanation of what happened

## How AI Is Used

The platform does not route tickets with simple keyword rules.

Instead, it uses an LLM-driven LangGraph agent that must call MCP tools before
making a routing decision. The classification result is parsed into a structured
decision object containing the department, category, confidence, reasoning, and
tools used.

After classification, deterministic orchestration logic decides what happens
next:

- If confidence is below `0.7`, the agent does not create a department ticket.
- If confidence is high enough and the department has a registered MCP server,
  the agent creates a department ticket.
- The agent always attempts to create an internal 311 tracking ticket through
  the primary MCP server.
- Internal ticket creation receives context about confidence, department server
  availability, and whether department ticket creation succeeded or was skipped.

Ticket status is not hard-coded by the client. The prompt instructs the model to
use the tool metadata, tool descriptions, and input parameters exposed by the MCP
ticketing tools to determine the correct status.

## Business Value / ROI

This type of platform can reduce operational overhead by:

- Reducing manual triage of citizen service requests
- Improving consistency in department routing
- Creating internal and department tickets from one intake flow
- Preserving confidence and routing rationale for auditability
- Preventing low-confidence classifications from automatically creating
  department-level tickets
- Speeding up dispatcher and customer-service workflows
- Making department-specific integrations modular through an MCP registry

## High-Level Architecture

```text
Citizen Complaint
        |
        v
FastAPI / CLI / Python API
        |
        v
LangGraph Routing Agent
        |
        v
Primary MCP Classification Tool
        |
        v
Structured Department Decision
        |
        +--> Confidence Gate
        |        |
        |        v
        |   Department MCP Registry
        |        |
        |        v
        |   Department Ticket Creation
        |
        v
Internal 311 Ticket Creation via Primary MCP Server
        |
        v
User-Friendly Summary + Structured JSON Response
```

## Current Implementation

The current implementation uses:

- Python
- FastAPI + Uvicorn for REST access
- LangGraph for agent orchestration
- LangChain MCP adapters for MCP tool access
- OpenAI chat models for tool-driven reasoning
- YAML-based department MCP registry
- Pydantic models for structured responses

The default department registry includes one department-level MCP server:

```yaml
departments:
  Code Compliance:
    url: localhost:8082
    transport: streamable_http
```

Bare registry URLs such as `localhost:8082` are normalized to
`http://localhost:8082/mcp`.

## REST API

Start the FastAPI app:

```bash
uvicorn agent311.api:app --host 0.0.0.0 --port 8000 --reload
```

Create a ticket:

```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "description": "There is peeling paint and broken windows on an abandoned building.",
    "address": "123 Main St, Chicago, IL"
  }'
```

The response includes:

- `summary`: a user-friendly explanation of classification and ticket creation
- `decision`: the full structured routing and ticket-creation result

Example summary:

```text
I classified the ticket to Code Compliance in the Code Concern - CCS category
with 72.3% confidence. Because the confidence was high enough and this
department has an MCP server, I created a department ticket with ID 2 and status
Received. I also created an internal 311 ticket with ID 2 and status Received.
```

Health check:

```bash
curl http://localhost:8000/health
```

## Command Line Usage

```bash
classify-311 \
  --description "There is a large pothole in the right lane." \
  --address "123 Main St, Chicago, IL"
```

Optional flags:

```bash
classify-311 \
  --mcp-url http://localhost:8080/mcp \
  --mcp-transport streamable_http \
  --model gpt-4.1-mini \
  --classification-tool-name classify_ticket \
  --mcp-registry mcp_registry.yaml \
  --log-level INFO \
  --config config.properties \
  --description "Trash has not been picked up for two weeks." \
  --address "25 N State St, Chicago, IL"
```

## Configuration

Edit `config.properties`:

```properties
MCP_URL = http://localhost:8080/mcp
MCP_TRANSPORT = streamable_http
MODEL = gpt-4.1-mini
OPENAI_API_KEY = your-api-key
RECURSION_LIMIT = 12
CLASSIFICATION_TOOL_NAME =
MCP_REGISTRY_PATH = mcp_registry.yaml
LOG_LEVEL = INFO
```

Environment variable `OPENAI_API_KEY` overrides the value in
`config.properties`.

Set `CLASSIFICATION_TOOL_NAME` when startup should require one specific MCP
classification tool. Leave it blank to accept any discovered MCP tool while
still requiring the model to call an MCP tool before returning a department.

Use `MCP_TRANSPORT = http` instead of `streamable_http` if your installed
`langchain-mcp-adapters` version expects the newer HTTP transport label.

## Prerequisites

- Python 3.11+
- A primary MCP server running at `http://localhost:8080/mcp`
- Optional department-level MCP servers registered in `mcp_registry.yaml`
- `OPENAI_API_KEY` set in `config.properties` or in your environment

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use From Python

```python
import asyncio
from agent311.orchestrator import classify_ticket

result = asyncio.run(
    classify_ticket(
        description="Streetlight is out near the alley.",
        address="500 W Madison St, Chicago, IL",
    )
)

print(result.model_dump_json(indent=2))
```
