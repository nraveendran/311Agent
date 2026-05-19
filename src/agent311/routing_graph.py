"""LangGraph construction for 311 ticket routing."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from agent311.models import AgentState
from agent311.prompts import system_prompt


logger = logging.getLogger(__name__)

AGENT_NODE = "agent"
TOOLS_NODE = "tools"
END_NODE = "__end__"


def build_routing_graph(*, tools, model: str, openai_api_key: str | None):
    return build_tool_graph(
        tools=tools,
        model=model,
        openai_api_key=openai_api_key,
        system_content=system_prompt(),
        require_tool_result=True,
    )


def build_tool_graph(
    *,
    tools,
    model: str,
    openai_api_key: str | None,
    system_content: str,
    require_tool_result: bool,
):
    logger.info("Building ChatOpenAI model: model=%s", model)
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=SecretStr(openai_api_key) if openai_api_key else None,
    ).bind_tools(tools)

    async def call_model(state: AgentState):
        logger.info("Calling LLM with %d message(s)", len(state["messages"]))
        response = await llm.ainvoke(
            [SystemMessage(content=system_content), *state["messages"]]
        )
        if isinstance(response, AIMessage) and response.tool_calls:
            tool_names = [tool_call.get("name", "unknown") for tool_call in response.tool_calls]
            logger.info("LLM requested MCP tool call(s): %s", ", ".join(tool_names))
        else:
            logger.info("LLM returned a final response")
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node(AGENT_NODE, call_model)
    graph.add_node(TOOLS_NODE, ToolNode(tools))
    graph.set_entry_point(AGENT_NODE)
    graph.add_conditional_edges(
        AGENT_NODE,
        lambda state: next_step(state, require_tool_result=require_tool_result),
        {TOOLS_NODE: TOOLS_NODE, END_NODE: END_NODE},
    )
    graph.add_edge(TOOLS_NODE, AGENT_NODE)
    return graph.compile()


def next_step(state: AgentState, *, require_tool_result: bool = True) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Routing graph to MCP tools")
        return TOOLS_NODE

    has_tool_result = any(isinstance(message, ToolMessage) for message in state["messages"])
    if require_tool_result and not has_tool_result:
        logger.error("LLM attempted to classify without an MCP tool result")
        raise RuntimeError(
            "The model attempted to finish without calling a required MCP tool."
        )

    logger.info("MCP tool result found; ending graph")
    return END_NODE
