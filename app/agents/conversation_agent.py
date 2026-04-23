from __future__ import annotations

from typing import Any

from app.usecases import dto as usecase_dto

from .base import BaseAgent
from .contracts import AgentCommand, AgentTask


class ConversationAgent(BaseAgent):
    name = "conversation_agent"

    @staticmethod
    def _normalize_intent(value: Any) -> usecase_dto.IntentParseResult:
        if isinstance(value, usecase_dto.IntentParseResult):
            return value
        if isinstance(value, dict):
            return usecase_dto.IntentParseResult(
                intent_type=str(value.get("intent_type") or "chat"),
                is_actionable=bool(value.get("is_actionable")),
                risk_level=str(value.get("risk_level") or "low"),
                needs_confirmation=bool(value.get("needs_confirmation")),
                reason=str(value.get("reason") or ""),
            )
        return usecase_dto.IntentParseResult(intent_type="chat", reason="fallback_chat")

    @staticmethod
    def _guide_flow_label(flow: str) -> str:
        value = str(flow or "").strip()
        if value == "travel":
            return "差旅"
        if value == "material":
            return "材料费"
        return "当前"

    @staticmethod
    def _direct_home_flow_request(text: str) -> str | None:
        source = str(text or "").strip()
        if not source:
            return None
        lowered = source.lower()
        enter_tokens = ["进入", "进去", "进到", "切到", "直接进", "开始", "走"]
        wants_enter = any(token in source for token in enter_tokens) or any(token in lowered for token in ["enter", "start", "go"])
        if not wants_enter:
            return None
        if any(token in source for token in ["差旅", "出差", "机票", "酒店"]) or "travel" in lowered:
            return "travel"
        if any(token in source for token in ["材料费", "材料", "采购", "发票"]) or "material" in lowered:
            return "material"
        return None

    @staticmethod
    def _replace_last_assistant_message(state: dict[str, Any], reply: str) -> dict[str, Any]:
        updated = dict(state or {})
        history = list(updated.get("conversation_history") or [])
        for idx in range(len(history) - 1, -1, -1):
            item = history[idx]
            if isinstance(item, dict) and str(item.get("role") or "") == "assistant":
                history[idx] = {"role": "assistant", "content": str(reply or "").strip()}
                updated["conversation_history"] = history
                return updated
        if reply:
            history.append({"role": "assistant", "content": str(reply or "").strip()})
        updated["conversation_history"] = history
        return updated

    def _plan_travel_turn(self, task: AgentTask):
        payload = dict(task.payload or {})
        user_text = str(payload.get("user_text") or "").strip()
        if not user_text:
            return self._result(
                ok=False,
                summary="缺少用户输入。",
                payload={"intent": usecase_dto.IntentParseResult(intent_type="chat", reason="empty_message").to_dict()},
            )

        intent_parser = payload.get("intent_parser")
        if callable(intent_parser):
            intent = self._normalize_intent(intent_parser(user_text, dict(payload.get("intent_context") or {})))
        else:
            intent = usecase_dto.IntentParseResult(intent_type="chat", reason="missing_intent_parser")

        commands: list[AgentCommand] = []
        reply = ""
        pending_action_spec: dict[str, Any] | None = None
        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            pending_builder = payload.get("pending_action_builder")
            if callable(pending_builder):
                pending_action_spec = pending_builder(user_text)
            action_name = str((pending_action_spec or {}).get("summary") or "待确认动作")
            reply = f"这步会改当前结果，我先放到右侧待确认：{action_name}。你点确认后我再执行。"
        elif intent.intent_type == "light_edit":
            commands.append(
                AgentCommand(
                    command_type="travel_light_edit",
                    payload={
                        "user_text": user_text,
                        **dict(payload.get("execution_payload") or {}),
                    },
                    summary="执行差旅轻修正",
                    risk_level="low",
                    requires_confirmation=False,
                    created_by=self.name,
                )
            )
        elif intent.intent_type == "ambiguous":
            summary_text = str(payload.get("summary_text") or "").strip()
            reply = f"{summary_text} 你直接告诉我具体文件名和目标类型，我就能继续处理。".strip()
        else:
            reply_llm = payload.get("reply_llm")
            reply_rule = payload.get("reply_rule")
            generated = None
            if callable(reply_llm):
                generated = reply_llm(
                    user_text,
                    dict(payload.get("assignment") or {}),
                    dict(payload.get("status") or {}),
                    list(payload.get("profiles") or []),
                    list(payload.get("messages") or []),
                )
            if not generated and callable(reply_rule):
                generated = reply_rule(
                    user_text,
                    dict(payload.get("assignment") or {}),
                    dict(payload.get("status") or {}),
                    list(payload.get("profiles") or []),
                )
            reply = str(generated or "").strip()

        return self._result(
            summary=f"Travel turn planned: {intent.intent_type}",
            payload={
                "intent": intent.to_dict(),
                "reply": reply,
                "pending_action_spec": dict(pending_action_spec or {}),
                "user_text": user_text,
            },
            commands=commands,
            events=[self._event("travel_turn_planned", f"Travel turn planned: {intent.intent_type}", {"intent": intent.to_dict()})],
        )

    def _compose_travel_edit_reply(self, task: AgentTask):
        payload = dict(task.payload or {})
        execution_ok = bool(payload.get("execution_ok"))
        execution_summary = str(payload.get("execution_summary") or "").strip()
        result_type = str(payload.get("result_type") or "")
        total_changed = int(payload.get("total_changed") or 0)
        slot_changed_count = int(payload.get("slot_changed_count") or 0)
        slot_changed_names = list(payload.get("slot_changed_names") or [])
        target_slot_label = str(payload.get("target_slot_label") or "")
        changed_count = int(payload.get("changed_count") or 0)
        changed_names = list(payload.get("changed_names") or [])
        target_doc_type_label = str(payload.get("target_doc_type_label") or "")
        amount_changed_count = int(payload.get("amount_changed_count") or 0)
        amount_changed_names = list(payload.get("amount_changed_names") or [])
        manual_amount_text = str(payload.get("manual_amount_text") or "")

        if not execution_ok and result_type != "no_change":
            reply = f"这次没改成功：{execution_summary}。你换个更完整的文件名，或者让我先把当前分配列出来。"
            return self._result(summary="Travel edit reply composed.", payload={"reply": reply})

        if execution_ok and total_changed > 0:
            slot_preview = "、".join(slot_changed_names[:3]) if slot_changed_names else ""
            if slot_changed_count > 3:
                slot_preview += f" 等{slot_changed_count}个文件"
            changed_preview = "、".join(changed_names[:3]) if changed_names else ""
            if changed_count > 3:
                changed_preview += f" 等{changed_count}个文件"
            amount_preview = "、".join(amount_changed_names[:3]) if amount_changed_names else ""
            if amount_changed_count > 3:
                amount_preview += f" 等{amount_changed_count}个文件"
            change_text = f"我已完成 {total_changed} 项轻量修正。"
            if slot_changed_count > 0 and target_slot_label:
                change_text += f" 已把 {slot_changed_count} 份材料调整到{target_slot_label}"
                if slot_preview:
                    change_text += f"（{slot_preview}）"
                change_text += "。"
            if changed_preview:
                change_text += f" 主要调整：{changed_preview}。"
            if target_doc_type_label:
                change_text += f" 目标类型：{target_doc_type_label}。"
            if amount_changed_count > 0 and manual_amount_text:
                change_text += (
                    f" 已把 {amount_changed_count} 份材料金额修正为 {manual_amount_text}"
                    + (f"（{amount_preview}）" if amount_preview else "")
                    + "。"
                )
            reply = f"已改好。{change_text} 如需恢复，我可以撤销刚才这一步。"
            return self._result(summary="Travel edit reply composed.", payload={"reply": reply})

        reply = f"{execution_summary or '这次没有命中可执行变更。'} 你可以继续说“改成返程机票明细”这类更具体的目标。"
        return self._result(summary="Travel edit reply composed.", payload={"reply": reply})

    def _plan_material_turn(self, task: AgentTask):
        payload = dict(task.payload or {})
        user_text = str(payload.get("user_text") or "").strip()
        if not user_text:
            return self._result(ok=False, summary="缺少用户输入。")

        intent_parser = payload.get("intent_parser")
        if callable(intent_parser):
            intent = self._normalize_intent(intent_parser(user_text, dict(payload.get("intent_context") or {})))
        else:
            intent = usecase_dto.IntentParseResult(intent_type="chat", reason="missing_intent_parser")

        commands: list[AgentCommand] = []
        reply = ""
        pending_action_spec: dict[str, Any] | None = None
        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            pending_builder = payload.get("pending_action_builder")
            if callable(pending_builder):
                pending_action_spec = pending_builder(user_text, payload.get("task"), dict(payload.get("fields") or {}))
            action_name = str((pending_action_spec or {}).get("summary") or "待确认动作")
            reply = f"这步影响比较大，我先放到右侧待确认：{action_name}。你确认后我再执行。"
        elif intent.intent_type == "light_edit":
            commands.append(
                AgentCommand(
                    command_type="material_light_edit",
                    payload={
                        "user_text": user_text,
                        **dict(payload.get("execution_payload") or {}),
                    },
                    summary="执行材料费轻修正",
                    risk_level="low",
                    requires_confirmation=False,
                    created_by=self.name,
                )
            )
        elif intent.intent_type == "ambiguous":
            row_count = int(payload.get("row_count") or 0)
            quality_hint_count = int(payload.get("quality_hint_count") or 0)
            pending_count = int(payload.get("pending_count") or 0)
            reply = f"目前明细 {row_count} 行，质量提示 {quality_hint_count} 条，待确认动作 {pending_count} 条。你告诉我具体行号和字段，我就能继续处理。"
        else:
            reply_llm = payload.get("reply_llm")
            generated = None
            if callable(reply_llm):
                generated = reply_llm(
                    user_text,
                    payload.get("task"),
                    dict(payload.get("fields") or {}),
                    list(payload.get("messages") or []),
                )
            reply = str(generated or "").strip()
            if not reply:
                reply = "我先解释当前判断。你可以继续追问原因，或者直接告诉我想改成什么。"

        return self._result(
            summary=f"Material turn planned: {intent.intent_type}",
            payload={
                "intent": intent.to_dict(),
                "reply": reply,
                "pending_action_spec": dict(pending_action_spec or {}),
                "user_text": user_text,
            },
            commands=commands,
            events=[self._event("material_turn_planned", f"Material turn planned: {intent.intent_type}", {"intent": intent.to_dict()})],
        )

    def _compose_material_edit_reply(self, task: AgentTask):
        payload = dict(task.payload or {})
        execution_ok = bool(payload.get("execution_ok"))
        execution_summary = str(payload.get("execution_summary") or "").strip()
        if execution_ok:
            reply = f"已改好。{execution_summary or '已更新当前表格。'} 如需恢复，我可以撤销刚才这一步。"
        else:
            reply = f"{execution_summary or '这次没有命中可执行动作。'} 你可以更具体地告诉我哪一行、哪个字段、希望改成什么。"
        return self._result(summary="Material edit reply composed.", payload={"reply": reply})

    def _run_home_turn(self, task: AgentTask):
        payload = dict(task.payload or {})
        processor = payload.get("turn_processor")
        if not callable(processor):
            return self._result(ok=False, summary="缺少首页引导处理器。")
        state, reply = processor(
            payload.get("state"),
            user_message=str(payload.get("user_message") or ""),
            uploaded_files=list(payload.get("uploaded_files") or []),
        )
        state = dict(state or {})
        user_message = str(payload.get("user_message") or "").strip()
        uploaded_files = list(payload.get("uploaded_files") or [])
        enter_flow = self._direct_home_flow_request(user_message)
        recommended_flow = str(state.get("recommended_flow") or "unknown")
        file_count = len(uploaded_files)
        if enter_flow in {"travel", "material"}:
            flow_label = self._guide_flow_label(enter_flow)
            if file_count > 0:
                reply = f"可以，已切到{flow_label}工作台，并带入当前 {file_count} 份材料。"
            else:
                reply = f"可以，已切到{flow_label}工作台。你可以在那边继续上传材料。"
        elif recommended_flow in {"travel", "material"} and user_message:
            flow_label = self._guide_flow_label(recommended_flow)
            if file_count > 0:
                base = str(reply or "").strip()
                if base and "进入" not in base:
                    reply = f"{base}\n\n你可以直接进入{flow_label}工作台继续处理。"
                elif not base:
                    reply = f"这批材料更像{flow_label}报销。你可以直接进入{flow_label}工作台继续处理。"
            elif any(token in user_message for token in ["差旅", "材料", "材料费"]):
                base = str(reply or "").strip()
                if base and "进入" not in base:
                    reply = f"{base}\n\n如果你确定方向，可以直接进入{flow_label}工作台。"
                elif not base:
                    reply = f"先按{flow_label}报销理解。你现在可以直接进入{flow_label}工作台，或先上传 1-2 份材料让我预检。"
        state = self._replace_last_assistant_message(state, str(reply or "").strip())
        return self._result(
            summary="Home guide turn processed.",
            payload={"state": state, "reply": str(reply or ""), "enter_flow": enter_flow},
            events=[self._event("home_turn_processed", "Home guide turn processed.")],
        )

    def run(self, task: AgentTask):
        objective = str(task.objective or "").strip()
        if objective == "plan_travel_turn":
            return self._plan_travel_turn(task)
        if objective == "compose_travel_edit_reply":
            return self._compose_travel_edit_reply(task)
        if objective == "plan_material_turn":
            return self._plan_material_turn(task)
        if objective == "compose_material_edit_reply":
            return self._compose_material_edit_reply(task)
        if objective == "run_home_turn":
            return self._run_home_turn(task)
        return self._result(ok=False, summary=f"Unsupported conversation objective: {objective}")
