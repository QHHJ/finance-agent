from __future__ import annotations

from typing import Any, TypedDict

from sqlalchemy.orm import Session


class FinanceGraphState(TypedDict, total=False):
    task_id: str
    db: Session
    raw_text: str
    extracted_data: dict[str, Any]
    suggestion_data: dict[str, Any]
    final_data: dict[str, Any]

    task_type: str
    review_items: list[dict[str, Any]]
    material_fix_result: dict[str, Any]
    travel_context: dict[str, Any]
    route_reason: str
    llm_error: str | None
