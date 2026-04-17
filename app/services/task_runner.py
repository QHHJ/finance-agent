from __future__ import annotations

from sqlalchemy.orm import Session

from app.db import repo
from app.graph.build_graph import get_finance_graph


def run_task_pipeline(db: Session, task_id: str):
    graph = get_finance_graph()
    try:
        graph.invoke({"task_id": task_id, "db": db})
    except Exception as exc:
        repo.set_task_status(db, task_id, "failed", error_message=str(exc))
        raise
    return repo.get_task(db, task_id)
