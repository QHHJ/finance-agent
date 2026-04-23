from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseAgent
from .contracts import AgentCommand, AgentResult, AgentTask


class ExecutionAgent(BaseAgent):
    name = "execution_agent"

    @staticmethod
    def _load_task_ops():
        return import_module("app.usecases.task_orchestration")

    @staticmethod
    def _load_material_usecase():
        return import_module("app.usecases.material_agent")

    def _missing_dependency(self, exc: ModuleNotFoundError, scope: str) -> AgentResult:
        dependency = getattr(exc, "name", None) or "unknown"
        summary = f"{scope} unavailable: missing dependency {dependency}."
        return self._result(
            ok=False,
            summary=summary,
            events=[self._event("dependency_missing", summary, {"scope": scope, "dependency": dependency})],
        )

    @staticmethod
    def _callable(payload: dict[str, Any], key: str):
        value = payload.get(key)
        return value if callable(value) else None

    def _execute_travel_edit_pipeline(
        self,
        *,
        command_text: str,
        payload: dict[str, Any],
        reason: str,
    ) -> AgentResult:
        profiles = list(payload.get("profiles") or [])
        manual_overrides = payload.get("manual_overrides")
        manual_slot_overrides = payload.get("manual_slot_overrides")
        reclassify_fn = self._callable(payload, "reclassify_fn")
        slot_fn = self._callable(payload, "slot_fn")
        relabel_fn = self._callable(payload, "relabel_fn")
        amount_fn = self._callable(payload, "amount_fn")
        build_assignment_fn = self._callable(payload, "build_assignment_fn")
        remember_overrides_fn = self._callable(payload, "remember_overrides_fn")
        sync_slot_overrides_fn = self._callable(payload, "sync_slot_overrides_fn")
        learn_fn = self._callable(payload, "learn_fn")

        if not command_text:
            return self._result(ok=False, summary="缺少可执行的差旅修改内容。")
        if not all([reclassify_fn, slot_fn, relabel_fn, amount_fn, build_assignment_fn]):
            return self._result(ok=False, summary="差旅执行命令缺少必要处理器。")

        recheck_count, _, recheck_error = reclassify_fn(
            command_text,
            profiles,
            manual_overrides=manual_overrides,
            manual_slot_overrides=manual_slot_overrides,
        )
        if recheck_error:
            return self._result(
                ok=False,
                summary=str(recheck_error),
                payload={"profiles": profiles, "result_type": "error"},
                events=[self._event("travel_edit_failed", str(recheck_error), {"reason": reason})],
            )

        slot_changed_count, slot_changed_names, target_slot = slot_fn(command_text, profiles)
        changed_count, changed_names, target_doc_type = relabel_fn(command_text, profiles)
        amount_changed_count, amount_changed_names, manual_amount, amount_error = amount_fn(command_text, profiles)
        if amount_error:
            return self._result(
                ok=False,
                summary=str(amount_error),
                payload={"profiles": profiles, "result_type": "error"},
                events=[self._event("travel_edit_failed", str(amount_error), {"reason": reason})],
            )

        total_changed = int(recheck_count) + int(slot_changed_count) + int(changed_count) + int(amount_changed_count)
        if total_changed <= 0:
            return self._result(
                ok=False,
                summary="确认后未命中可执行变更，请补充更具体的目标。",
                payload={
                    "profiles": profiles,
                    "result_type": "no_change",
                    "target_slot": target_slot,
                    "target_doc_type": target_doc_type,
                },
                events=[self._event("travel_edit_noop", "No travel change was applied.", {"reason": reason})],
            )

        if callable(remember_overrides_fn):
            remember_overrides_fn(manual_overrides, profiles)
        if callable(sync_slot_overrides_fn):
            sync_slot_overrides_fn(manual_slot_overrides, profiles)

        assignment = build_assignment_fn(profiles)
        if callable(learn_fn):
            try:
                learn_fn(profiles, assignment, reason=reason)
            except Exception:
                pass

        summary = f"已执行 {total_changed} 项差旅调整。"
        if slot_changed_count > 0 and target_slot:
            summary += f" 槽位已调整到 {target_slot}。"
        elif changed_count > 0 and target_doc_type:
            summary += f" 分类已调整到 {target_doc_type}。"
        elif amount_changed_count > 0 and manual_amount is not None:
            summary += f" 金额已修正为 {manual_amount}。"

        return self._result(
            summary=summary,
            payload={
                "assignment": assignment,
                "profiles": profiles,
                "result_type": "changed",
                "total_changed": total_changed,
                "recheck_count": recheck_count,
                "slot_changed_count": slot_changed_count,
                "slot_changed_names": slot_changed_names,
                "target_slot": target_slot,
                "changed_count": changed_count,
                "changed_names": changed_names,
                "target_doc_type": target_doc_type,
                "amount_changed_count": amount_changed_count,
                "amount_changed_names": amount_changed_names,
                "manual_amount": manual_amount,
            },
            events=[self._event("travel_edit_applied", summary, {"reason": reason, "total_changed": total_changed})],
        )

    def _execute_travel_pending_action(self, command: AgentCommand) -> AgentResult:
        payload = dict(command.payload or {})
        action = dict(payload.get("action") or {})
        action_type = str(action.get("action_type") or "")
        if action_type == "travel_reorganize":
            organize_fn = self._callable(payload, "organize_fn")
            if not callable(organize_fn):
                return self._result(ok=False, summary="差旅重组命令缺少整理器。")
            assignment, profiles = organize_fn(
                list(payload.get("pool_list") or []),
                manual_overrides=payload.get("manual_overrides"),
                manual_slot_overrides=payload.get("manual_slot_overrides"),
            )
            return self._result(
                summary="已完成重新归并，并刷新去程/返程/酒店分配。",
                payload={"assignment": assignment, "profiles": profiles, "result_type": "reorganized"},
                events=[self._event("travel_reorganized", "Travel materials reorganized.")],
            )
        if action_type == "travel_export":
            return self._result(
                summary="已确认导出。请在下方“差旅材料打包导出”点击导出按钮。",
                payload={"result_type": "export_confirmed", "export_confirmed": True},
                events=[self._event("travel_export_confirmed", "Travel export confirmed.")],
            )
        if action_type == "travel_apply_all":
            return self._result(
                summary="已确认批量应用请求。若有待确认分类动作，会逐条执行。",
                payload={"result_type": "batch_confirmed"},
                events=[self._event("travel_batch_confirmed", "Travel batch confirmation accepted.")],
            )
        if action_type == "travel_manual_confirm":
            action_payload = dict(action.get("payload") or {})
            command_text = str(action_payload.get("command") or action.get("target") or "").strip()
            return self._execute_travel_edit_pipeline(
                command_text=command_text,
                payload=payload,
                reason="pending_confirm",
            )
        return self._result(ok=False, summary=f"暂不支持的差旅动作类型：{action_type}")

    def _execute_material_light_edit(self, command: AgentCommand) -> AgentResult:
        payload = dict(command.payload or {})
        handler = self._callable(payload, "handler")
        if not callable(handler):
            return self._result(ok=False, summary="材料费轻修正命令缺少处理器。")
        handled, reply, updated_task, updated_fields = handler(
            str(payload.get("user_text") or "").strip(),
            payload.get("task"),
            dict(payload.get("fields") or {}),
        )
        ok = bool(handled)
        summary = str(reply or ("已执行材料费轻修正。" if handled else "未命中材料费修改动作。"))
        return self._result(
            ok=ok,
            summary=summary,
            payload={
                "handled": handled,
                "task": updated_task,
                "fields": updated_fields,
            },
            events=[self._event("material_edit_applied" if handled else "material_edit_skipped", summary)],
        )

    def _execute_material_pending_action(self, command: AgentCommand) -> AgentResult:
        payload = dict(command.payload or {})
        handler = self._callable(payload, "handler")
        set_export_confirmed = self._callable(payload, "set_export_confirmed")
        if not callable(handler):
            return self._result(ok=False, summary="材料费待确认命令缺少处理器。")
        ok, msg, updated_task, updated_fields = handler(
            dict(payload.get("action") or {}),
            payload.get("task"),
            dict(payload.get("fields") or {}),
        )
        if callable(set_export_confirmed) and str((payload.get("action") or {}).get("action_type") or "") == "material_export":
            export_flag_key = str(payload.get("export_flag_key") or "")
            if export_flag_key:
                set_export_confirmed(export_flag_key)
        return self._result(
            ok=bool(ok),
            summary=str(msg or ""),
            payload={"task": updated_task, "fields": updated_fields},
            events=[self._event("material_pending_executed" if ok else "material_pending_failed", str(msg or ""))],
        )

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
        if command_type == "travel_light_edit":
            return self._execute_travel_edit_pipeline(
                command_text=str(command.payload.get("user_text") or "").strip(),
                payload=dict(command.payload or {}),
                reason="chat_light_edit",
            )
        if command_type == "travel_pending_action":
            return self._execute_travel_pending_action(command)
        if command_type == "material_light_edit":
            return self._execute_material_light_edit(command)
        if command_type == "material_pending_action":
            return self._execute_material_pending_action(command)
        if command_type == "upload_policy_pdf":
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
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
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
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
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
            task = task_ops.process_task(str(command.payload.get("task_id") or ""))
            return self._result(
                summary="Task processed.",
                payload={"task": task},
                events=[self._event("task_processed", "Task processed.", {"task_id": getattr(task, "id", None)})],
            )
        if command_type == "apply_material_corrections":
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
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
            try:
                material_usecase = self._load_material_usecase()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "material specialist")
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
            try:
                material_usecase = self._load_material_usecase()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "material specialist")
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
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
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
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
            indexed = task_ops.rebuild_policy_rag_index(limit=int(command.payload.get("limit") or 500))
            return self._result(
                summary=f"Policy RAG index rebuilt: {indexed}",
                payload={"indexed": indexed},
                events=[self._event("policy_rag_rebuilt", f"Policy RAG index rebuilt: {indexed}", {"indexed": indexed})],
            )
        if command_type == "delete_policy":
            try:
                task_ops = self._load_task_ops()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "task orchestration")
            deleted = task_ops.delete_policy(int(command.payload.get("policy_id") or 0))
            return self._result(
                ok=bool(deleted),
                summary="Policy deleted." if deleted else "Policy delete failed.",
                payload={"deleted": deleted},
                events=[self._event("policy_deleted" if deleted else "policy_delete_failed", "Policy deleted." if deleted else "Policy delete failed.")],
            )
        if command_type == "material_batch_process":
            try:
                material_usecase = self._load_material_usecase()
            except ModuleNotFoundError as exc:
                return self._missing_dependency(exc, "material specialist")
            result = material_usecase.process_uploaded_files(list(command.payload.get("uploaded_files") or []))
            return self._result(
                summary=f"Material batch prepared: {len(result.task_ids)} tasks.",
                payload={"batch_result": result},
                events=[self._event("material_batch_processed", f"Material batch prepared: {len(result.task_ids)} tasks.", {"task_count": len(result.task_ids)})],
            )
        return self._result(ok=False, summary=f"Unsupported command type: {command_type}")
