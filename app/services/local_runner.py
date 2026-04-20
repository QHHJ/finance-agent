from __future__ import annotations

"""Backward-compatible wrappers.

This module is kept for compatibility with old callers. The actual business
orchestration lives in app.usecases.task_orchestration.
"""

from app.usecases import task_orchestration as task_ops


def upload_policy_pdf(filename: str, content: bytes):
    return task_ops.upload_policy_pdf(filename, content)


def create_and_process_task(
    filename: str,
    content: bytes,
    auto_process: bool = True,
    auto_export: bool = True,
):
    return task_ops.create_and_process_task(
        filename=filename,
        content=content,
        auto_process=auto_process,
        auto_export=auto_export,
    )


def process_task(task_id: str):
    return task_ops.process_task(task_id)


def list_tasks(limit: int = 200):
    return task_ops.list_tasks(limit=limit)


def get_task(task_id: str):
    return task_ops.get_task(task_id)


def list_policies(limit: int = 200):
    return task_ops.list_policies(limit=limit)


def rebuild_policy_rag_index(limit: int = 500) -> int:
    return task_ops.rebuild_policy_rag_index(limit=limit)


def delete_policy(policy_id: int) -> bool:
    return task_ops.delete_policy(policy_id)


def apply_corrections(task_id: str, corrections: dict):
    return task_ops.apply_corrections(task_id, corrections)


def export_task(task_id: str, export_format: str = "both") -> dict[str, str | None]:
    return task_ops.export_task(task_id, export_format=export_format)
