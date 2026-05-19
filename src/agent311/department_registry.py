"""YAML-backed registry for department-level MCP servers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field


class DepartmentMCPServer(BaseModel):
    """Connection settings for one department MCP server."""

    department: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    transport: str = Field(default="streamable_http", min_length=1)


class DepartmentMCPRegistry(BaseModel):
    """Lookup table for department MCP servers."""

    departments: dict[str, DepartmentMCPServer] = Field(default_factory=dict)

    def find(self, department: str) -> DepartmentMCPServer | None:
        normalized_department = normalize_department_name(department)
        for registered_department, server in self.departments.items():
            if normalize_department_name(registered_department) == normalized_department:
                return server
        return None


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "mcp_registry.yaml"


def load_department_registry(registry_path: str | Path | None = None) -> DepartmentMCPRegistry:
    path = Path(registry_path) if registry_path else default_registry_path()
    if not path.exists():
        return DepartmentMCPRegistry()

    raw_registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_departments = raw_registry.get("departments") or {}
    if not isinstance(raw_departments, dict):
        raise ValueError(f"Department MCP registry must contain a 'departments' mapping: {path}")

    departments: dict[str, DepartmentMCPServer] = {}
    for department, raw_server in raw_departments.items():
        if isinstance(raw_server, str):
            raw_server = {"url": raw_server}
        if not isinstance(raw_server, dict):
            raise ValueError(f"Registry entry for department '{department}' must be a mapping or URL.")

        server = DepartmentMCPServer(
            department=str(department),
            url=normalize_mcp_url(str(raw_server.get("url", ""))),
            transport=str(raw_server.get("transport", "streamable_http")),
        )
        departments[str(department)] = server

    return DepartmentMCPRegistry(departments=departments)


def normalize_department_name(department: str) -> str:
    return " ".join(department.strip().casefold().split())


def normalize_mcp_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        return stripped

    if "://" not in stripped:
        stripped = f"http://{stripped}"

    parsed = urlparse(stripped)
    if parsed.path in ("", "/"):
        return stripped.rstrip("/") + "/mcp"
    return stripped
