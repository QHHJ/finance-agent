from __future__ import annotations

from typing import Any

from app.usecases import material_agent as material_usecase
from app.usecases import task_orchestration as task_ops

from .base import BaseAgent
from .contracts import AgentCommand, AgentResult, AgentTask


class ExecutionAgent(BaseAgent):
    name = "execution_agent"

    def run(self, task: AgentTask) -> AgentResult:
        objective = str(task.objective or "").strip()
        if objective != "execute_command":
            return self._result(ok=False, summary=f"Unsupported execution objective: {objective}")
        command = task.payload.get("command")
        if not isinstance(command, AgentCommand):
            return self._result(ok=False, summary="Missing AgentCommand payload.")
        return self.execute(command)

    def execute(self, command: AgentCommand) -> AgentResult:
        command_type = str(command.command_type or "").strip()
        if command_type == "upload_policy_pdf":
            policy = task_ops.upload_policy_pdf(
                filename=str(command.payload.get("filename") or ""),
                content=bytes(command.payload.get("content") or b""),
            )
            return self._result(
                summary="Policy PDF uploaded.",
                payload={"policy": policy},
                events=[self._event("policy_uploaded", "Policy PDF uploaded.", {"policy_id": getattr(policy, "id", None)})],
            )
        if command_type == "create_and_process_task":
            task = task_ops.create_and_process_task(
                filename=str(command.payload.get("filename") or ""),
                content=bytes(command.payload.get("content") or b""),
                auto_process=bool(command.payload.get("auto_process", True)),
                auto_export=bool(command.payload.get("auto_export", True)),
            )
            return self._result(
                summary="Material task created and processed.",
                payload={"task": task},
                events=[self._event("task_processed", "Material task created and processed.", {"task_id": getattr(task, "id", None)})],
            )
        if command_type == "process_task":
            task = task_ops.process_task(str(command.payload.get("task_id") or ""))
            return self._result(
                summary="Task processed.",
                payload={"task": task},
                events=[self._event("task_processed", "Task processed.", {"task_id": getattr(task, "id", None)})],
            )
        if command_type == "apply_material_corrections":
            task = task_ops.apply_corrections(
                task_id=str(command.payload.get("task_id") or ""),
                corrections=dict(command.payload.get("corrections") or {}),
            )
            return self._result(
                summary="Material corrections applied.",
                payload={"task": task},
                events=[self._event("material_corrections_applied", "Material corrections applied.", {"task_id": getattr(task, "id", None)})],
            )
        if command_type == "material_apply_updates":
            result = material_usecase.apply_updates(
                task_id=str(command.payload.get("task_id") or ""),
                fields=dict(command.payload.get("fields") or {}),
            )
            return self._result(
                ok=bool(result.ok),
                summary=result.message or ("Material updates applied." if result.ok else "Material update failed."),
                payload={"operation_result": result},
                events=[
                    self._event(
                        "material_updates_applied" if result.ok else "material_updates_failed",
                        result.message or ("Material updates applied." if result.ok else "Material update failed."),
                        {"task_id": str(command.payload.get("task_id") or "")},
                    )
                ],
            )
        if command_type == "material_reprocess_and_export":
            result = material_usecase.reprocess_and_export(str(command.payload.get("task_id") or ""))
            return self._result(
                ok=bool(result.ok),
                summary=result.message or ("Material task reprocessed." if result.ok else "Material task reprocess failed."),
                payload={"operation_result": result},
                events=[
                    self._event(
                        "material_reprocessed" if result.ok else "material_reprocess_failed",
                        result.message or ("Material task reprocessed." if result.ok else "Material task reprocess failed."),
                        {"task_id": str(command.payload.get("task_id") or "")},
                    )
                ],
            )
        if command_type == "export_task":
            output = task_ops.export_task(
                task_id=str(command.payload.get("task_id") or ""),
                export_format=str(command.payload.get("export_format") or "both"),
            )
            return self._result(
                summary="Task exported.",
                payload={"export": output},
                events=[self._event("task_exported", "Task exported.", {"task_id": str(command.payload.get("task_id") or "")})],
            )
        if command_type == "rebuild_policy_rag_index":
            indexed = task_ops.rebuild_policy_rag_index(limit=int(command.payload.get("limit") or 500))
            return self._result(
                summary=f"Policy RAG index rebuilt: {indexed}",
                payload={"indexed": indexed},
                events=[self._event("policy_rag_rebuilt", f"Policy RAG index rebuilt: {indexed}", {"indexed": indexed})],
            )
        if command_type == "delete_policy":
            deleted = task_ops.delete_policy(int(command.payload.get("policy_id") or 0))
            return self._result(
                ok=bool(deleted),
                summary="Policy deleted." if deleted else "Policy delete failed.",
                payload={"deleted": deleted},
                events=[self._event("policy_deleted" if deleted else "policy_delete_failed", "Policy deleted." if deleted else "Policy delete failed.")],
            )
        if command_type == "material_batch_process":
            result = material_usecase.process_uploaded_files(list(command.payload.get("uploaded_files") or []))
            return self._result(
                summary=f"Material batch prepared: {len(result.task_ids)} tasks.",
                payload={"batch_result": result},
                events=[self._event("material_batch_processed", f"Material batch prepared: {len(result.task_ids)} tasks.", {"task_count": len(result.task_ids)})],
            )
        return self._result(ok=False, summary=f"Unsupported command type: {command_type}")
