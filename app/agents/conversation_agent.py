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
        return "当前流程"

    @staticmethod
    def _direct_home_flow_request(text: str) -> str | None:
        source = str(text or "").strip()
        if not source:
            return None
        lowered = source.lower()
        enter_tokens = ["进入", "进到", "切到", "切换到", "直接去", "开始", "先走", "走", "转到"]
        goal_tokens = ["我要", "我想", "帮我", "现在就", "直接"]
        scene_tokens = ["报销", "流程", "处理", "整理"]
        travel_tokens = ["差旅", "出差", "机票", "高铁", "火车", "酒店", "travel", "trip"]
        material_tokens = ["材料费", "材料", "采购", "元器件", "发票", "material", "purchase"]

        wants_enter = any(token in source for token in enter_tokens) or any(token in lowered for token in ["enter", "start", "go", "switch"])
        goal_style = any(token in source for token in goal_tokens) and any(token in source for token in scene_tokens)

        asks_travel = any(token in source for token in travel_tokens) or any(token in lowered for token in ["travel", "trip"])
        asks_material = any(token in source for token in material_tokens) or any(token in lowered for token in ["material", "purchase"])

        if asks_travel and (wants_enter or goal_style):
            return "travel"
        if asks_material and (wants_enter or goal_style):
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
        reply_source = "none"
        pending_action_spec: dict[str, Any] | None = None

        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            pending_builder = payload.get("pending_action_builder")
            if callable(pending_builder):
                pending_action_spec = pending_builder(user_text)
            action_name = str((pending_action_spec or {}).get("summary") or "待确认动作")
            reply = f"这个操作会影响当前结果，我先放入待确认：{action_name}。确认后我再执行。"
            reply_source = "system"
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
            reply = f"{summary_text} 你直接告诉我“文件名 + 目标类型”，我就能继续处理。".strip()
            reply_source = "system"
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
                if generated:
                    reply_source = "llm"
            if not generated and callable(reply_rule):
                generated = reply_rule(
                    user_text,
                    dict(payload.get("assignment") or {}),
                    dict(payload.get("status") or {}),
                    list(payload.get("profiles") or []),
                )
                if generated:
                    reply_source = "rule"
            reply = str(generated or "").strip()
            if not reply:
                reply = "我先按当前结果回答。你可以继续问“还缺什么”或直接说要改哪一条。"
                reply_source = "system"

        return self._result(
            summary=f"Travel turn planned: {intent.intent_type}",
            payload={
                "intent": intent.to_dict(),
                "reply": reply,
                "reply_source": reply_source,
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
            reply = f"这次没有改成功：{execution_summary}。你可以补充完整文件名，或先让我列出当前分配。"
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

            lines: list[str] = [f"已完成 {total_changed} 项轻量修正。"]
            if slot_changed_count > 0 and target_slot_label:
                detail = f"- 槽位调整：{slot_changed_count} 份 -> {target_slot_label}"
                if slot_preview:
                    detail += f"（{slot_preview}）"
                lines.append(detail)
            if changed_preview:
                lines.append(f"- 主要变更文件：{changed_preview}")
            if target_doc_type_label:
                lines.append(f"- 目标类型：{target_doc_type_label}")
            if amount_changed_count > 0 and manual_amount_text:
                detail = f"- 金额修正：{amount_changed_count} 份 -> {manual_amount_text}"
                if amount_preview:
                    detail += f"（{amount_preview}）"
                lines.append(detail)
            lines.append("如需恢复，我可以撤销上一步。")
            return self._result(summary="Travel edit reply composed.", payload={"reply": "\n".join(lines)})

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
        reply_source = "none"
        pending_action_spec: dict[str, Any] | None = None

        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            pending_builder = payload.get("pending_action_builder")
            if callable(pending_builder):
                pending_action_spec = pending_builder(user_text, payload.get("task"), dict(payload.get("fields") or {}))
            action_name = str((pending_action_spec or {}).get("summary") or "待确认动作")
            reply = f"这步影响范围较大，我先放到待确认：{action_name}。你确认后我再执行。"
            reply_source = "system"
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
            reply = f"当前明细 {row_count} 行，质量提示 {quality_hint_count} 条，待确认 {pending_count} 条。你给我“行号 + 字段 + 目标值”，我就继续处理。"
            reply_source = "system"
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
                if generated:
                    reply_source = "llm"
            reply = str(generated or "").strip()
            if not reply:
                reply = "我先解释当前判断，不会直接改数据。你可以继续追问原因，或告诉我希望改成什么。"
                reply_source = "system"

        return self._result(
            summary=f"Material turn planned: {intent.intent_type}",
            payload={
                "intent": intent.to_dict(),
                "reply": reply,
                "reply_source": reply_source,
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
            reply = f"已改好。{execution_summary or '已更新当前表格。'} 如需恢复，我可以撤销上一步。"
        else:
            reply = f"{execution_summary or '这次没有命中可执行动作。'} 你可以更具体地说“第几行、哪个字段、改成什么”。"
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
        reply_source = "rule"

        if enter_flow in {"travel", "material"}:
            flow_label = self._guide_flow_label(enter_flow)
            if file_count > 0:
                reply = f"可以，已切到{flow_label}工作台，并带入当前 {file_count} 份材料。"
            else:
                reply = f"可以，已切到{flow_label}工作台。你可以继续上传材料。"
        elif recommended_flow in {"travel", "material"} and user_message:
            flow_label = self._guide_flow_label(recommended_flow)
            base = str(reply or "").strip()
            if file_count > 0:
                if base and "进入" not in base:
                    reply = f"{base}\n\n你可以直接进入{flow_label}工作台继续处理。"
                elif not base:
                    reply = f"这批材料更像{flow_label}报销。你可以直接进入{flow_label}工作台继续处理。"
            elif any(token in user_message for token in ["差旅", "材料", "材料费", "报销"]):
                if base and "进入" not in base:
                    reply = f"{base}\n\n如果方向已确定，可以直接进入{flow_label}工作台。"
                elif not base:
                    reply = f"先按{flow_label}报销理解。你现在可以直接进入{flow_label}工作台，或先上传1-2份材料让我预检。"

        # Keep routing deterministic, but let LLM make chat responses more natural.
        reply_llm = payload.get("reply_llm")
        if enter_flow not in {"travel", "material"} and callable(reply_llm):
            generated = reply_llm(user_message, dict(state or {}), list(uploaded_files or []))
            generated_text = str(generated or "").strip()
            if generated_text:
                reply = generated_text
                reply_source = "llm"

        state = self._replace_last_assistant_message(state, str(reply or "").strip())
        return self._result(
            summary="Home guide turn processed.",
            payload={"state": state, "reply": str(reply or ""), "enter_flow": enter_flow, "reply_source": reply_source},
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
