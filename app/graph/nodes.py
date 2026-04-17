from __future__ import annotations

from app.db import repo
from app.services import extractor, parser, validator

from .state import FinanceGraphState


def load_task_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    if not task:
        raise ValueError(f"Task not found: {state['task_id']}")
    repo.set_task_status(db, task.id, "processing")
    return {}


def parse_pdf_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    if not task:
        raise ValueError(f"Task not found: {state['task_id']}")
    raw_text = parser.parse_pdf_text(task.stored_path)
    return {"raw_text": raw_text}


def extract_fields_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    pdf_path = task.stored_path if task else None
    extracted_data = extractor.extract_invoice_fields(state.get("raw_text", ""), pdf_path=pdf_path)
    return {"extracted_data": extracted_data}


def suggest_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    extracted_data = state.get("extracted_data", {})
    policies = repo.list_policy_documents(db, limit=100)
    historical_samples = repo.get_historical_samples(
        db,
        extracted_data.get("bill_type"),
        item_content=extracted_data.get("item_content"),
        limit=200,
    )
    suggestion_data = validator.suggest_processing(
        extracted_data=extracted_data,
        raw_text=state.get("raw_text", ""),
        policies=policies,
        historical_samples=historical_samples,
    )
    return {"suggestion_data": suggestion_data}


def persist_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    repo.save_processing_result(
        db=db,
        task_id=state["task_id"],
        raw_text=state.get("raw_text", ""),
        extracted_data=state.get("extracted_data", {}),
        suggestion_data=state.get("suggestion_data", {}),
    )
    return {}
