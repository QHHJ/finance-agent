from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.db import repo
from app.db.session import SessionLocal
from app.runtime import EXPORT_DIR, POLICY_DIR, UPLOAD_DIR, init_runtime
from app.services import exporter, learning, parser, rag_ingest, task_runner


def _save_uploaded_bytes(filename: str, content: bytes, folder: Path) -> Path:
    safe_name = Path(filename).name if filename else "uploaded.pdf"
    stored_name = f"{uuid4().hex}_{safe_name}"
    target_path = folder / stored_name
    target_path.write_bytes(content)
    return target_path.resolve()


def upload_policy_pdf(filename: str, content: bytes):
    init_runtime()
    stored_path = _save_uploaded_bytes(filename, content, POLICY_DIR)
    raw_text = parser.parse_file_text(stored_path, max_pages=0)
    content_hash = parser.compute_file_sha256(stored_path)

    db = SessionLocal()
    try:
        policy = repo.create_policy_document(
            db=db,
            name=filename,
            stored_path=str(stored_path),
            content_hash=content_hash,
            raw_text=raw_text,
        )
        try:
            rag_ingest.sync_policy_document(policy)
        except Exception:
            # Keep policy upload available even if vector ingestion is temporarily unavailable.
            pass
        return policy
    finally:
        db.close()


def create_and_process_task(
    filename: str,
    content: bytes,
    auto_process: bool = True,
    auto_export: bool = True,
):
    init_runtime()
    stored_path = _save_uploaded_bytes(filename, content, UPLOAD_DIR)

    db = SessionLocal()
    try:
        task = repo.create_task(db, original_filename=filename, stored_path=str(stored_path))
        if auto_process:
            task = task_runner.run_task_pipeline(db, task.id)
            if auto_export and task is not None and task.status in {"completed", "corrected"}:
                excel_path = exporter.export_to_excel(
                    task_id=task.id,
                    extracted_data=task.extracted_data or {},
                    suggestion_data=task.suggestion_data or {},
                    final_data=task.final_data or {},
                    export_dir=EXPORT_DIR,
                )
                text_path = exporter.export_to_text(
                    task_id=task.id,
                    extracted_data=task.extracted_data or {},
                    suggestion_data=task.suggestion_data or {},
                    final_data=task.final_data or {},
                    export_dir=EXPORT_DIR,
                )
                repo.save_export_paths(db, task_id=task.id, excel_path=excel_path, text_path=text_path)
        return task
    finally:
        db.close()


def process_task(task_id: str):
    init_runtime()
    db = SessionLocal()
    try:
        return task_runner.run_task_pipeline(db, task_id)
    finally:
        db.close()


def list_tasks(limit: int = 200):
    init_runtime()
    db = SessionLocal()
    try:
        return repo.list_tasks(db, limit=limit)
    finally:
        db.close()


def get_task(task_id: str):
    init_runtime()
    db = SessionLocal()
    try:
        return repo.get_task(db, task_id)
    finally:
        db.close()


def list_policies(limit: int = 200):
    init_runtime()
    db = SessionLocal()
    try:
        return repo.list_policy_documents(db, limit=limit)
    finally:
        db.close()


def rebuild_policy_rag_index(limit: int = 500) -> int:
    init_runtime()
    db = SessionLocal()
    try:
        policies = repo.list_policy_documents(db, limit=limit)
    finally:
        db.close()

    indexed = 0
    for policy in policies:
        try:
            indexed += rag_ingest.sync_policy_document(policy)
        except Exception:
            continue
    return indexed


def delete_policy(policy_id: int) -> bool:
    init_runtime()
    db = SessionLocal()
    try:
        stored_path = repo.delete_policy_document(db, policy_id)
    finally:
        db.close()

    if stored_path is None:
        return False
    file_path = Path(stored_path)
    if file_path.exists():
        file_path.unlink()
    return True


def apply_corrections(task_id: str, corrections: dict):
    init_runtime()
    db = SessionLocal()
    try:
        task = repo.get_task(db, task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        updated_task = repo.apply_corrections(db, task, corrections)
        try:
            learning.learn_from_material_task(updated_task)
        except Exception:
            # Learning should not block correction save.
            pass
        return updated_task
    finally:
        db.close()


def export_task(task_id: str, export_format: str = "both") -> dict[str, str | None]:
    init_runtime()
    db = SessionLocal()
    try:
        task = repo.get_task(db, task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        if task.status not in {"completed", "corrected"}:
            raise ValueError("Task is not completed yet.")

        excel_path = None
        text_path = None
        if export_format in {"excel", "both"}:
            excel_path = exporter.export_to_excel(
                task_id=task.id,
                extracted_data=task.extracted_data or {},
                suggestion_data=task.suggestion_data or {},
                final_data=task.final_data or {},
                export_dir=EXPORT_DIR,
            )
        if export_format in {"text", "both"}:
            text_path = exporter.export_to_text(
                task_id=task.id,
                extracted_data=task.extracted_data or {},
                suggestion_data=task.suggestion_data or {},
                final_data=task.final_data or {},
                export_dir=EXPORT_DIR,
            )
        repo.save_export_paths(db, task_id=task.id, excel_path=excel_path, text_path=text_path)
        return {"excel_path": excel_path, "text_path": text_path}
    finally:
        db.close()
