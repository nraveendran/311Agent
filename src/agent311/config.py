"""Configuration loading for the 311 agent."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Runtime settings for the 311 routing agent."""

    mcp_url: str = Field(..., min_length=1)
    mcp_transport: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    openai_api_key: str | None = None
    recursion_limit: int = Field(..., ge=1)
    classification_tool_name: str | None = None
    log_level: str = "INFO"


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.properties"


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    """Load agent settings from a properties file.

    `AGENT311_CONFIG_PATH` can point to a different properties file.
    `OPENAI_API_KEY` overrides the properties file value when present.
    """

    path = Path(config_path or os.getenv("AGENT311_CONFIG_PATH") or _default_config_path())
    if not path.exists():
        raise FileNotFoundError(f"Could not read config file: {path}")

    properties = _read_properties(path)

    config = AgentConfig(
        mcp_url=properties.get("MCP_URL", ""),
        mcp_transport=properties.get("MCP_TRANSPORT", ""),
        model=properties.get("MODEL", ""),
        openai_api_key=(properties.get("OPENAI_API_KEY") or None),
        recursion_limit=int(properties.get("RECURSION_LIMIT", "12")),
        classification_tool_name=(properties.get("CLASSIFICATION_TOOL_NAME") or None),
        log_level=properties.get("LOG_LEVEL", "INFO"),
    )

    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        config.openai_api_key = env_api_key

    return config


def _read_properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        separator = "=" if "=" in stripped else ":"
        if separator not in stripped:
            continue
        key, value = stripped.split(separator, 1)
        properties[key.strip()] = value.strip()
    return properties
