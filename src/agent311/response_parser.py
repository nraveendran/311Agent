"""Parsing helpers for final LLM responses."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import BaseMessage

from agent311.models import DepartmentDecision


logger = logging.getLogger(__name__)


def parse_department_decision(message: BaseMessage) -> DepartmentDecision:
    content = message_content_as_text(message)
    try:
        decision = DepartmentDecision.model_validate(extract_json(content))
        logger.info(
            "Classification complete: department=%s confidence=%s category=%s",
            decision.department,
            decision.confidence,
            decision.category,
        )
        return decision
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Final model response was not parseable JSON; returning fallback decision")
        return fallback_decision(content)


def message_content_as_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content)


def extract_json(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def fallback_decision(content: str) -> DepartmentDecision:
    return DepartmentDecision(
        department="Unknown",
        confidence=None,
        category=None,
        reasoning=(
            "The agent did not return parseable JSON. Raw final response: "
            f"{content.strip()}"
        ),
        tools_used=[],
    )
