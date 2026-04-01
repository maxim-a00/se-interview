"""LangGraph workflow assembly and routing policy.

Design note:
- This module is responsible for orchestration, not tool implementation.
- The workflow was intentionally constrained after Phoenix evaluation showed
  repeated tool calls and inefficient search loops.
- Travel requests now follow a deliberate sequence: search only when current
  information is needed, call the itinerary tool once, then finalize without
  more tool use.
- The goal was to solve a real behavior problem without over-engineering the
  project with additional layers that are unnecessary for the assignment.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.config import get_config
from app.tools import build_travel_itinerary, get_tools


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TRAVEL_TERMS = {
    "travel",
    "trip",
    "vacation",
    "honeymoon",
    "hotel",
    "hotels",
    "flight",
    "flights",
    "attraction",
    "attractions",
    "beach",
    "beaches",
    "itinerary",
    "destination",
    "destinations",
    "resort",
    "resorts",
}

LIVE_SEARCH_TERMS = {
    "current",
    "latest",
    "today",
    "tonight",
    "right now",
    "available",
    "availability",
    "bookable",
    "booking",
    "price",
    "prices",
    "cost",
    "flight",
    "flights",
    "check-in",
    "check out",
}


def _message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content)


def _latest_user_message(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _called_tool_names(messages: list[AnyMessage]) -> list[str]:
    tool_names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_names.append(tool_call["name"])
    return tool_names


def _is_travel_request(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(term in lowered for term in TRAVEL_TERMS)


def _needs_live_search(user_message: str) -> bool:
    lowered = user_message.lower()
    has_date_hint = any(char.isdigit() for char in lowered) and any(
        month in lowered
        for month in (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "2026",
            "2027",
        )
    )
    return has_date_hint or any(term in lowered for term in LIVE_SEARCH_TERMS)


def _workflow_stage(messages: list[AnyMessage]) -> str:
    user_message = _latest_user_message(messages)
    tool_names = _called_tool_names(messages)
    used_search = "duckduckgo_search" in tool_names
    used_itinerary = "build_travel_itinerary" in tool_names

    if _is_travel_request(user_message):
        if used_itinerary:
            return "travel_finalize"
        if _needs_live_search(user_message) and not used_search:
            return "travel_search"
        return "travel_structure"

    if used_search:
        return "general_finalize"
    return "general_search"


def create_model():
    config = get_config()
    return ChatOpenAI(model=config.openai_model, temperature=config.openai_temperature)


def llm_call(state: MessagesState, model, tools_by_name: dict[str, object]) -> dict:
    """Call the LLM with a stage-specific prompt and constrained tool choice."""
    stage = _workflow_stage(state["messages"])

    if stage == "travel_search":
        system_prompt = (
            "You are gathering live travel facts for a later structured answer. "
            "Call duckduckgo_search exactly once with the single best query for the user's request. "
            "Do not call any other tool in this step."
        )
        runnable = model.bind_tools([tools_by_name["duckduckgo_search"]])
    elif stage == "travel_structure":
        system_prompt = (
            "You are preparing the structured travel answer now. "
            "Call build_travel_itinerary exactly once in this step. "
            "Use any earlier search results if they exist. "
            "Do not call duckduckgo_search in this step. "
            "If a booking link or other detail is unknown, leave it blank or mark it as not provided rather than searching again."
        )
        runnable = model.bind_tools([tools_by_name["build_travel_itinerary"]])
    elif stage == "travel_finalize":
        system_prompt = (
            "You already have the structured travel output you need. "
            "Write the final answer using the itinerary tool result. "
            "Do not call any more tools."
        )
        runnable = model
    elif stage == "general_finalize":
        system_prompt = (
            "You already have the search results you need. "
            "Write the final answer using those results. "
            "Do not call any more tools."
        )
        runnable = model
    else:
        system_prompt = (
            "You are a helpful assistant. "
            "If the question needs current information, call duckduckgo_search exactly once. "
            "Otherwise, answer directly."
        )
        runnable = model.bind_tools([tools_by_name["duckduckgo_search"]])

    return {
        "messages": [
            runnable.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        ]
    }


def tool_node(state: MessagesState, tools_by_name: dict[str, object]) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    tools = get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    model = create_model()

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", lambda state: llm_call(state, model, tools_by_name))
    graph_builder.add_node("tool_node", lambda state: tool_node(state, tools_by_name))

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    return graph_builder.compile()
