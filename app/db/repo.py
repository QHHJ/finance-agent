from __future__ import annotations

import json
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CorrectionLog, ExpenseTask, PolicyDocument, RagVectorDocument


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def create_task(db: Session, original_filename: str, stored_path: str) -> ExpenseTask:
    task = ExpenseTask(original_filename=original_filename, stored_path=stored_path, status="uploaded")
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: str) -> ExpenseTask | None:
    return db.get(ExpenseTask, task_id)


def list_tasks(db: Session, limit: int = 100) -> list[ExpenseTask]:
    stmt = select(ExpenseTask).order_by(ExpenseTask.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def set_task_status(db: Session, task_id: str, status: str, error_message: str | None = None) -> ExpenseTask:
    task = get_task(db, task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    task.status = status
    task.error_message = error_message
    db.commit()
    db.refresh(task)
    return task


def save_processing_result(
    db: Session,
    task_id: str,
    raw_text: str,
    extracted_data: dict[str, Any],
    suggestion_data: dict[str, Any],
) -> ExpenseTask:
    task = get_task(db, task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")

    task.raw_text = raw_text
    task.extracted_data = extracted_data
    task.suggestion_data = suggestion_data
    task.final_data = {
        "expense_category": suggestion_data.get("expense_category"),
        "required_materials": suggestion_data.get("required_materials", []),
        "risk_points": suggestion_data.get("risk_points", []),
        "policy_references": suggestion_data.get("policy_references", []),
        "similar_case_refs": suggestion_data.get("similar_case_refs", []),
        "rationale": suggestion_data.get("rationale", ""),
        "rag_trace": suggestion_data.get("rag_trace", {}),
        "extracted_fields": extracted_data,
    }
    task.status = "completed"
    task.error_message = None

    db.commit()
    db.refresh(task)
    return task


def save_export_paths(
    db: Session,
    task_id: str,
    excel_path: str | None = None,
    text_path: str | None = None,
) -> ExpenseTask:
    task = get_task(db, task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if excel_path:
        task.export_excel_path = excel_path
    if text_path:
        task.export_text_path = text_path
    db.commit()
    db.refresh(task)
    return task


def apply_corrections(db: Session, task: ExpenseTask, corrections: dict[str, Any]) -> ExpenseTask:
    extracted_data = dict(task.extracted_data or {})
    final_data = dict(task.final_data or {})

    extracted_fields = corrections.get("extracted_fields")
    if extracted_fields:
        for field_name, new_value in extracted_fields.items():
            if new_value is None:
                continue
            old_value = extracted_data.get(field_name)
            if old_value == new_value:
                continue
            extracted_data[field_name] = new_value
            db.add(
                CorrectionLog(
                    task_id=task.id,
                    field_name=f"extracted.{field_name}",
                    old_value=_to_text(old_value),
                    new_value=_to_text(new_value),
                )
            )
        final_data["extracted_fields"] = extracted_data

    if "expense_category" in corrections and corrections["expense_category"] is not None:
        new_value = corrections["expense_category"]
        old_value = final_data.get("expense_category")
        if old_value != new_value:
            final_data["expense_category"] = new_value
            db.add(
                CorrectionLog(
                    task_id=task.id,
                    field_name="expense_category",
                    old_value=_to_text(old_value),
                    new_value=_to_text(new_value),
                )
            )

    if "required_materials" in corrections and corrections["required_materials"] is not None:
        new_value = corrections["required_materials"]
        old_value = final_data.get("required_materials")
        if old_value != new_value:
            final_data["required_materials"] = new_value
            db.add(
                CorrectionLog(
                    task_id=task.id,
                    field_name="required_materials",
                    old_value=_to_text(old_value),
                    new_value=_to_text(new_value),
                )
            )

    if "risk_points" in corrections and corrections["risk_points"] is not None:
        new_value = corrections["risk_points"]
        old_value = final_data.get("risk_points")
        if old_value != new_value:
            final_data["risk_points"] = new_value
            db.add(
                CorrectionLog(
                    task_id=task.id,
                    field_name="risk_points",
                    old_value=_to_text(old_value),
                    new_value=_to_text(new_value),
                )
            )

    task.extracted_data = extracted_data
    task.final_data = final_data
    task.status = "corrected"

    db.commit()
    db.refresh(task)
    return task


def create_policy_document(
    db: Session,
    name: str,
    stored_path: str,
    content_hash: str,
    raw_text: str,
) -> PolicyDocument:
    policy = PolicyDocument(
        name=name,
        stored_path=stored_path,
        content_hash=content_hash,
        raw_text=raw_text,
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy


def list_policy_documents(db: Session, limit: int = 50) -> list[PolicyDocument]:
    stmt = select(PolicyDocument).order_by(PolicyDocument.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def delete_policy_document(db: Session, policy_id: int) -> str | None:
    policy = db.get(PolicyDocument, policy_id)
    if policy is None:
        return None
    stored_path = policy.stored_path
    rag_stmt = select(RagVectorDocument).where(
        RagVectorDocument.source_type == "policy",
        RagVectorDocument.source_id == str(policy_id),
    )
    for rag_doc in db.scalars(rag_stmt):
        db.delete(rag_doc)
    db.delete(policy)
    db.commit()
    return stored_path


def upsert_rag_document(
    db: Session,
    *,
    source_type: str,
    source_id: str,
    doc_key: str,
    content: str,
    embedding: list[float],
    title: str | None = None,
    metadata_json: dict[str, Any] | None = None,
) -> RagVectorDocument:
    stmt = select(RagVectorDocument).where(RagVectorDocument.doc_key == doc_key)
    doc = db.scalar(stmt)
    if doc is None:
        doc = RagVectorDocument(
            source_type=source_type,
            source_id=source_id,
            doc_key=doc_key,
            title=title,
            content=content,
            metadata_json=metadata_json or {},
            embedding=embedding,
        )
        db.add(doc)
    else:
        doc.source_type = source_type
        doc.source_id = source_id
        doc.title = title
        doc.content = content
        doc.metadata_json = metadata_json or {}
        doc.embedding = embedding

    db.flush()
    db.refresh(doc)
    return doc


def delete_rag_documents(
    db: Session,
    *,
    source_type: str | None = None,
    source_id: str | None = None,
    doc_key_prefix: str | None = None,
) -> int:
    stmt = select(RagVectorDocument)
    if source_type:
        stmt = stmt.where(RagVectorDocument.source_type == source_type)
    if source_id:
        stmt = stmt.where(RagVectorDocument.source_id == source_id)
    if doc_key_prefix:
        stmt = stmt.where(RagVectorDocument.doc_key.like(f"{doc_key_prefix}%"))

    docs = list(db.scalars(stmt))
    for doc in docs:
        db.delete(doc)
    db.flush()
    return len(docs)


def list_rag_documents(
    db: Session,
    *,
    source_types: list[str] | None = None,
    limit: int = 5000,
) -> list[RagVectorDocument]:
    stmt = select(RagVectorDocument).order_by(RagVectorDocument.updated_at.desc()).limit(limit)
    if source_types:
        stmt = stmt.where(RagVectorDocument.source_type.in_(source_types))
    return list(db.scalars(stmt))


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _is_similar_item(current_item: str | None, historical_item: str | None) -> bool:
    cur = _normalize_text(current_item)
    hist = _normalize_text(historical_item)
    if not cur:
        return True
    if not hist:
        return False
    return cur in hist or hist in cur


def get_historical_samples(
    db: Session,
    bill_type: str | None,
    item_content: str | None = None,
    limit: int = 200,
) -> list[dict[str, str]]:
    if not bill_type:
        return []

    stmt = select(ExpenseTask).order_by(ExpenseTask.updated_at.desc()).limit(limit)
    tasks = list(db.scalars(stmt))

    samples: list[dict[str, str]] = []
    for task in tasks:
        extracted = task.extracted_data or {}
        task_bill_type = extracted.get("bill_type")
        task_item_content = extracted.get("item_content")
        category = (task.final_data or {}).get("expense_category")
        if task_bill_type == bill_type and _is_similar_item(item_content, task_item_content) and category:
            samples.append(
                {
                    "task_id": task.id,
                    "bill_type": task_bill_type,
                    "item_content": task_item_content or "",
                    "expense_category": category,
                }
            )
    return samples


def summarize_historical_preference(samples: list[dict[str, str]]) -> tuple[str | None, int]:
    if not samples:
        return None, 0
    counter = Counter(sample["expense_category"] for sample in samples if sample.get("expense_category"))
    if not counter:
        return None, 0
    category, count = counter.most_common(1)[0]
    return category, count
