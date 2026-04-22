from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .nodes import (
    extract_fields_node,
    generic_suggest_node,
    load_task_node,
    material_agent_node,
    parse_pdf_node,
    persist_node,
    policy_agent_node,
    repair_agent_node,
    supervisor_node,
    travel_agent_node,
)
from .state import FinanceGraphState

logger = logging.getLogger("finance.graph")

_compiled_graph = None


def route_supervisor(state: FinanceGraphState) -> str:
    task_type = str(state.get("task_type") or "generic").strip().lower()
    if task_type == "material":
        return "material_agent"
    if task_type == "travel":
        return "travel_agent"
    if task_type == "policy":
        return "policy_agent"
    return "generic_suggest"


def route_task_type(state: FinanceGraphState) -> str:
    # Backward-compatible alias used by smoke tests and legacy callers.
    return route_supervisor(state)


def route_material_after_repair(state: FinanceGraphState) -> str:
    # Backward-compatible alias for older tests.
    return route_material_after_agent(state)


def route_material_after_agent(state: FinanceGraphState) -> str:
    next_action = str(state.get("next_action") or "persist").strip().lower()
    if next_action == "repair":
        logger.info("material -> repair")
        return "repair_agent"
    logger.info("material -> persist")
    return "persist"


def route_travel_after_agent(state: FinanceGraphState) -> str:
    next_action = str(state.get("next_action") or "persist").strip().lower()
    needs_policy = bool(state.get("needs_policy"))
    if next_action == "policy" or needs_policy:
        logger.info("travel -> policy")
        return "policy_agent"
    return "persist"


def build_finance_graph():
    graph_builder = StateGraph(FinanceGraphState)

    graph_builder.add_node("load_task", load_task_node)
    graph_builder.add_node("parse_pdf", parse_pdf_node)
    graph_builder.add_node("extract_fields", extract_fields_node)
    graph_builder.add_node("supervisor", supervisor_node)

    graph_builder.add_node("material_agent", material_agent_node)
    graph_builder.add_node("repair_agent", repair_agent_node)

    graph_builder.add_node("travel_agent", travel_agent_node)
    graph_builder.add_node("policy_agent", policy_agent_node)
    graph_builder.add_node("generic_suggest", generic_suggest_node)
    graph_builder.add_node("persist", persist_node)

    graph_builder.add_edge(START, "load_task")
    graph_builder.add_edge("load_task", "parse_pdf")
    graph_builder.add_edge("parse_pdf", "extract_fields")
    graph_builder.add_edge("extract_fields", "supervisor")

    graph_builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "material_agent": "material_agent",
            "travel_agent": "travel_agent",
            "policy_agent": "policy_agent",
            "generic_suggest": "generic_suggest",
        },
    )

    graph_builder.add_conditional_edges(
        "material_agent",
        route_material_after_agent,
        {
            "repair_agent": "repair_agent",
            "persist": "persist",
        },
    )
    graph_builder.add_edge("repair_agent", "persist")

    graph_builder.add_conditional_edges(
        "travel_agent",
        route_travel_after_agent,
        {
            "policy_agent": "policy_agent",
            "persist": "persist",
        },
    )

    graph_builder.add_edge("policy_agent", "persist")
    graph_builder.add_edge("generic_suggest", "persist")
    graph_builder.add_edge("persist", END)

    return graph_builder.compile()


def get_finance_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_finance_graph()
    return _compiled_graph
