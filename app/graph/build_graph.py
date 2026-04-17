from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import extract_fields_node, load_task_node, parse_pdf_node, persist_node, suggest_node
from .state import FinanceGraphState

_compiled_graph = None


def build_finance_graph():
    graph_builder = StateGraph(FinanceGraphState)
    graph_builder.add_node("load_task", load_task_node)
    graph_builder.add_node("parse_pdf", parse_pdf_node)
    graph_builder.add_node("extract_fields", extract_fields_node)
    graph_builder.add_node("suggest", suggest_node)
    graph_builder.add_node("persist", persist_node)

    graph_builder.add_edge(START, "load_task")
    graph_builder.add_edge("load_task", "parse_pdf")
    graph_builder.add_edge("parse_pdf", "extract_fields")
    graph_builder.add_edge("extract_fields", "suggest")
    graph_builder.add_edge("suggest", "persist")
    graph_builder.add_edge("persist", END)

    return graph_builder.compile()


def get_finance_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_finance_graph()
    return _compiled_graph
