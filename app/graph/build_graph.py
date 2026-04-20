from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    classify_task_node,
    extract_fields_node,
    generic_suggest_node,
    load_task_node,
    material_prepare_node,
    material_repair_node,
    material_validate_node,
    parse_pdf_node,
    persist_node,
    travel_assign_node,
    travel_prepare_node,
    travel_qa_context_node,
)
from .state import FinanceGraphState

_compiled_graph = None


def route_task_type(state: FinanceGraphState) -> str:
    task_type = str(state.get("task_type") or "generic").strip().lower()
    if task_type == "material":
        return "material_prepare"
    if task_type == "travel":
        return "travel_prepare"
    return "generic_suggest"


def route_material_after_repair(state: FinanceGraphState) -> str:
    review_items = list(state.get("review_items") or [])
    if review_items:
        return "material_validate"
    return "material_validate"


def build_finance_graph():
    graph_builder = StateGraph(FinanceGraphState)

    graph_builder.add_node("load_task", load_task_node)
    graph_builder.add_node("parse_pdf", parse_pdf_node)
    graph_builder.add_node("extract_fields", extract_fields_node)
    graph_builder.add_node("classify_task", classify_task_node)

    graph_builder.add_node("material_prepare", material_prepare_node)
    graph_builder.add_node("material_repair", material_repair_node)
    graph_builder.add_node("material_validate", material_validate_node)

    graph_builder.add_node("travel_prepare", travel_prepare_node)
    graph_builder.add_node("travel_assign", travel_assign_node)
    graph_builder.add_node("travel_qa_context", travel_qa_context_node)

    graph_builder.add_node("generic_suggest", generic_suggest_node)
    graph_builder.add_node("persist", persist_node)

    graph_builder.add_edge(START, "load_task")
    graph_builder.add_edge("load_task", "parse_pdf")
    graph_builder.add_edge("parse_pdf", "extract_fields")
    graph_builder.add_edge("extract_fields", "classify_task")

    graph_builder.add_conditional_edges(
        "classify_task",
        route_task_type,
        {
            "material_prepare": "material_prepare",
            "travel_prepare": "travel_prepare",
            "generic_suggest": "generic_suggest",
        },
    )

    graph_builder.add_edge("material_prepare", "material_repair")
    graph_builder.add_conditional_edges(
        "material_repair",
        route_material_after_repair,
        {
            "material_validate": "material_validate",
        },
    )
    graph_builder.add_edge("material_validate", "persist")

    graph_builder.add_edge("travel_prepare", "travel_assign")
    graph_builder.add_edge("travel_assign", "travel_qa_context")
    graph_builder.add_edge("travel_qa_context", "persist")

    graph_builder.add_edge("generic_suggest", "persist")
    graph_builder.add_edge("persist", END)

    return graph_builder.compile()


def get_finance_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_finance_graph()
    return _compiled_graph

