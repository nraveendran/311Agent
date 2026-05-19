# 311 Agent

LangGraph-based orchestrator for routing 311 tickets to the right department and creating tickets using tools exposed by MCP servers.

## Prerequisites

- Python 3.11+
- A classification MCP server running at `http://localhost:8080/mcp`
- Optional department-level MCP servers registered in `mcp_registry.yaml`
- `OPENAI_API_KEY` set in `config.properties` or in your environment

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Edit `config.properties` first:

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

Environment variable `OPENAI_API_KEY` overrides the value in `config.properties`.
Set `CLASSIFICATION_TOOL_NAME` when you want startup to require one specific MCP
tool name. Leave it blank to accept any discovered MCP tool, while still requiring
the model to call an MCP tool before it can return a department.

Department ticket creation is configured with `mcp_registry.yaml`:

```yaml
departments:
  Code Compliance:
    url: localhost:8082
    transport: streamable_http
```

After classification, the agent creates an internal 311 tracking ticket through
the tools exposed by `MCP_URL`. It passes the MCP tool metadata, including tool
descriptions and input schemas, into the ticket-creation prompt so statuses are
chosen from the server-provided metadata instead of hard-coded in the client.

When classification returns `Code Compliance`, the agent also loads that
department MCP server and asks its available tool to create the department
ticket. Registry URLs may be full MCP URLs such as `http://localhost:8082/mcp`;
bare host and port values are normalized to `http://<host>:<port>/mcp`.

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

Use `--mcp-transport http` instead if your installed `langchain-mcp-adapters`
version expects the newer HTTP transport label.

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
