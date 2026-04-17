from __future__ import annotations

from typing import Any, TypedDict

from sqlalchemy.orm import Session


class FinanceGraphState(TypedDict, total=False):
    task_id: str
    db: Session
    raw_text: str
    extracted_data: dict[str, Any]
    suggestion_data: dict[str, Any]
