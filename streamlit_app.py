from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

from app.agents import AgentCommand, AgentTask, ReimbursementAgentOrchestrator
from app.services.ollama_config import (
    chat_model as _chat_model,
    render_model_runtime_panel as _render_model_runtime_panel,
)
from app.services import travel_processing
from app.ui.agent_metrics import (
    record_action_outcome as _record_action_outcome,
    record_llm_outcome as _record_llm_outcome,
)
from app.ui.chat_widgets import (
    inject_ui_styles as _inject_ui_styles,
)
from app.ui import home_router, material_workbench, task_hub, travel_workbench, workbench
from app.usecases import dto as usecase_dto
from app.usecases import material_agent as material_usecase
from app.usecases import task_orchestration
from app.utils.json_tools import parse_json_object_loose as _parse_json_object_loose

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), encoding="utf-8-sig")

UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
_AGENT_ORCHESTRATOR: ReimbursementAgentOrchestrator | None = None


def _get_agent_orchestrator() -> ReimbursementAgentOrchestrator:
    global _AGENT_ORCHESTRATOR
    if _AGENT_ORCHESTRATOR is None:
        _AGENT_ORCHESTRATOR = ReimbursementAgentOrchestrator()
    return _AGENT_ORCHESTRATOR


def _run_travel_specialist_task(objective: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    result = _get_agent_orchestrator().run_task(
        AgentTask(
            agent="travel_specialist_agent",
            objective=str(objective or "").strip(),
            payload=dict(payload or {}),
        )
    )
    return bool(result.ok), dict(result.payload or {}), str(result.summary or "")


def _run_conversation_agent_task(
    objective: str,
    payload: dict[str, Any],
) -> tuple[bool, dict[str, Any], str, list[AgentCommand]]:
    result = _get_agent_orchestrator().run_task(
        AgentTask(
            agent="conversation_agent",
            objective=str(objective or "").strip(),
            payload=dict(payload or {}),
        )
    )
    return bool(result.ok), dict(result.payload or {}), str(result.summary or ""), list(result.commands or [])


def _execute_agent_command(command: AgentCommand, *, scope: str | None = None) -> tuple[bool, dict[str, Any], str]:
    result = _get_agent_orchestrator().execute_command(command)
    command_type = str(getattr(command, "command_type", "") or "").strip()
    metric_scope = str(scope or "").strip().lower()
    if not metric_scope:
        if command_type.startswith("travel_"):
            metric_scope = "travel"
        elif command_type.startswith("material_"):
            metric_scope = "material"
        else:
            metric_scope = "global"
    _record_action_outcome(metric_scope, bool(result.ok))
    return bool(result.ok), dict(result.payload or {}), str(result.summary or "")


def _infer_intent_with_llm(message: str, domain: str) -> usecase_dto.IntentParseResult | None:
    text = str(message or "").strip()
    if not text:
        return None
    metric_scope = str(domain or "generic").strip().lower() or "generic"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()
    prompt = (
        "你是报销Agent的对话意图理解器。仅返回JSON对象，不要输出其他内容。\n"
        "intent_type 只能是: chat, light_edit, strong_action, ambiguous。\n"
        "chat: 用户在问问题、要求解释、问还缺什么、问哪个文件是什么。\n"
        "light_edit: 用户明确要求修改某个文件的分类、槽位、金额、字段，或说“xxx文件是去程明细/酒店明细”。\n"
        "strong_action: 用户要求导出、批量应用、覆盖结果、重新归并、删除大量内容等高影响操作。\n"
        "ambiguous: 用户只说不对、有问题、再看看，但没有明确文件或目标。\n"
        "risk_level 只能是: low, medium, high。needs_confirmation 和 is_actionable 为布尔值。\n"
        "差旅例子：\n"
        "- “现在还缺什么文件” => chat\n"
        "- “A.jpg是去程明细，B.jpg是返程明细” => light_edit\n"
        "- “333.jpg是酒店明细” => light_edit\n"
        "- “应用全部建议/导出结果/重新归并” => strong_action, needs_confirmation=true\n"
        "材料费例子：\n"
        "- “最后一行金额改为7792” => light_edit\n"
        "- “为什么金额不一致” => chat\n"
        "返回示例: {\"intent_type\":\"chat\",\"is_actionable\":false,\"risk_level\":\"low\",\"needs_confirmation\":false,\"reason\":\"...\"}\n"
        f"domain={domain}\n"
        f"message={text}\n"
    )
    content = ""
    try:
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "你是稳定的JSON分类器。"},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(5, 20))
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
    except Exception:
        content = ""

    if not content:
        # Fallback for models/versions where /api/chat json mode is unstable.
        prompt_lines = [
            "你是稳定的JSON分类器，只输出JSON对象。",
            "intent_type: chat/light_edit/strong_action/ambiguous。",
            "risk_level: low/medium/high。",
            f"domain={domain}",
            f"message={text}",
        ]
        fallback_prompt = "\n".join(prompt_lines)
        try:
            payload = {
                "model": model,
                "stream": False,
                "prompt": fallback_prompt,
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(5, 20))
            resp.raise_for_status()
            content = str(resp.json().get("response") or "")
        except Exception:
            _record_llm_outcome(metric_scope, False)
            return None

    parsed = _parse_json_object_loose(content)
    if not parsed:
        _record_llm_outcome(metric_scope, False)
        return None
    intent = str(parsed.get("intent_type") or "").strip().lower()
    if intent not in {"chat", "light_edit", "strong_action", "ambiguous"}:
        _record_llm_outcome(metric_scope, False)
        return None
    risk = str(parsed.get("risk_level") or "low").strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "low"
    _record_llm_outcome(metric_scope, True)
    return usecase_dto.IntentParseResult(
        intent_type=intent,
        is_actionable=bool(parsed.get("is_actionable")),
        risk_level=risk,
        needs_confirmation=bool(parsed.get("needs_confirmation")),
        reason=str(parsed.get("reason") or ""),
    )


def classify_user_message_intent(message: str, context: dict[str, Any] | None = None) -> usecase_dto.IntentParseResult:
    text = str(message or "").strip()
    domain = str((context or {}).get("domain") or "generic")
    if not text:
        return usecase_dto.IntentParseResult(intent_type="chat", reason="empty_message")

    llm_guess = _infer_intent_with_llm(text, domain)
    if llm_guess is not None:
        return llm_guess

    return usecase_dto.IntentParseResult(
        intent_type="chat",
        is_actionable=False,
        risk_level="low",
        needs_confirmation=False,
        reason="llm_unavailable_chat_fallback",
    )


PAGE_HOME_GUIDE = home_router.PAGE_HOME_GUIDE
PAGE_TRAVEL_FLOW = home_router.PAGE_TRAVEL_FLOW
PAGE_MATERIAL_FLOW = home_router.PAGE_MATERIAL_FLOW


def _ensure_router_state() -> None:
    home_router.ensure_router_state()


def _set_current_page(page: str, *, pause_auto_route: bool = False, flash_message: str = "") -> None:
    home_router.set_current_page(page, pause_auto_route=pause_auto_route, flash_message=flash_message)


def _pop_router_flash_message() -> str:
    return home_router.pop_router_flash_message()


def _render_flow_back_to_home(flow: str) -> None:
    home_router.render_flow_back_to_home(flow)


def _get_guide_handoff_for_flow(flow: str) -> tuple[dict[str, Any], list[Any]]:
    return home_router.get_guide_handoff_for_flow(flow)


def _render_home_guide_agent() -> None:
    home_router.render_home_guide_agent(UPLOAD_TYPES)


def _list_material_sidebar_tasks(limit: int = 20) -> list[Any]:
    try:
        return list(task_orchestration.list_tasks(limit=limit) or [])
    except Exception:
        return []


def _handle_workbench_sidebar_action(action: dict[str, Any] | None) -> None:
    if not isinstance(action, dict):
        return
    action_name = str(action.get("action") or "").strip()
    if action_name == "open_home":
        _set_current_page(PAGE_HOME_GUIDE, flash_message="已返回报销任务立案页。")
        st.rerun()
    if action_name == "new_travel":
        task_id = task_hub.create_travel_task(title=f"差旅任务 {datetime.now().strftime('%m-%d %H:%M')}")
        task_hub.set_active_travel_task(task_id)
        _set_current_page(PAGE_TRAVEL_FLOW, flash_message="已新建差旅任务。")
        st.rerun()
    if action_name == "new_material":
        task_hub.set_selected_material_task("")
        _set_current_page(PAGE_MATERIAL_FLOW, flash_message="已进入材料工作台，可上传新发票开始处理。")
        st.rerun()
    if action_name == "open_travel":
        task_id = str(action.get("task_id") or "")
        if task_id:
            task_hub.set_active_travel_task(task_id)
            _set_current_page(PAGE_TRAVEL_FLOW)
            st.rerun()
    if action_name == "open_material":
        task_id = str(action.get("task_id") or "")
        if task_id:
            task_hub.set_selected_material_task(task_id)
            _set_current_page(PAGE_MATERIAL_FLOW)
            st.rerun()


def _configure_travel_processing() -> None:
    travel_processing.configure_travel_processing(
        run_travel_specialist_task=_run_travel_specialist_task,
        execute_agent_command=_execute_agent_command,
    )


def _configure_travel_workbench() -> None:
    _configure_travel_processing()
    travel_workbench.configure_travel_workbench(
        run_conversation_agent_task=_run_conversation_agent_task,
        execute_agent_command=_execute_agent_command,
        classify_user_message_intent=classify_user_message_intent,
        build_travel_execution_payload=travel_processing.build_travel_execution_payload,
        append_travel_pending_action_from_spec=travel_processing.append_travel_pending_action_from_spec,
        apply_manual_overrides_to_profiles=travel_processing.apply_manual_overrides_to_profiles,
        apply_manual_slot_overrides_to_profiles=travel_processing.apply_manual_slot_overrides_to_profiles,
        as_uploaded_list=travel_processing.as_uploaded_list,
        build_assignment_from_profiles=travel_processing.build_assignment_from_profiles,
        build_travel_agent_status=travel_processing.build_travel_agent_status,
        build_travel_file_profile=travel_processing.build_travel_file_profile,
        build_travel_handoff_status_reply=travel_processing.build_travel_handoff_status_reply,
        clone_travel_profile=travel_processing.clone_travel_profile,
        doc_type_label=travel_processing.doc_type_label,
        files_signature=travel_processing.files_signature,
        format_amount=travel_processing.format_amount,
        generate_travel_agent_reply_llm=travel_processing.generate_travel_agent_reply_llm,
        generate_travel_agent_reply_rule=travel_processing.generate_travel_agent_reply_rule,
        merge_uploaded_lists=travel_processing.merge_uploaded_lists,
        profile_file_key=travel_processing.profile_file_key,
        prune_manual_overrides=travel_processing.prune_manual_overrides,
        prune_manual_slot_overrides=travel_processing.prune_manual_slot_overrides,
        safe_float=travel_processing.safe_float,
        slot_label=travel_processing.slot_label,
        travel_execute_pending_action=travel_processing.travel_execute_pending_action,
        travel_pending_action_spec_from_text=travel_processing.travel_pending_action_spec_from_text,
        travel_pop_undo_snapshot=travel_processing.travel_pop_undo_snapshot,
        travel_push_undo_snapshot=travel_processing.travel_push_undo_snapshot,
        travel_restore_undo_snapshot=travel_processing.travel_restore_undo_snapshot,
        travel_scope_name=travel_processing.travel_scope_name,
        travel_undo_stack_key=travel_processing.travel_undo_stack_key,
        uploaded_file_key=travel_processing.uploaded_file_key,
    )


def _render_travel_workbench() -> dict[str, Any]:
    _configure_travel_workbench()
    return travel_workbench.render_travel_workbench()


def _render_travel_conversation_agent() -> dict[str, Any]:
    _configure_travel_workbench()
    return travel_workbench.render_travel_conversation_agent()


def _render_export_download(task, key_scope: str = "default") -> None:
    excel_path = task.export_excel_path
    text_path = task.export_text_path

    if not excel_path and not text_path:
        st.info("当前任务还没有导出文件。可点击“重新处理”或“导出 Excel + 文本”。")
        return

    st.markdown("**导出文件**")
    cols = st.columns(2)

    if excel_path:
        excel_file = Path(excel_path)
        if excel_file.exists():
            with excel_file.open("rb") as fp:
                cols[0].download_button(
                    label="下载 Excel",
                    data=fp.read(),
                    file_name=excel_file.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_excel_{key_scope}_{task.id}",
                )
        else:
            cols[0].warning("Excel 路径已记录，但文件不存在。")

    if text_path:
        text_file = Path(text_path)
        if text_file.exists():
            with text_file.open("rb") as fp:
                cols[1].download_button(
                    label="下载文本",
                    data=fp.read(),
                    file_name=text_file.name,
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_text_{key_scope}_{task.id}",
                )
        else:
            cols[1].warning("文本路径已记录，但文件不存在。")


def _render_material_conversation_agent() -> None:
    material_workbench.configure_material_workbench(
        run_conversation_agent_task=_run_conversation_agent_task,
        execute_agent_command=_execute_agent_command,
        classify_user_message_intent=classify_user_message_intent,
        get_guide_handoff_for_flow=_get_guide_handoff_for_flow,
        render_export_download=_render_export_download,
        render_included_file_list=travel_processing.render_included_file_list,
    )
    material_workbench.render_material_conversation_agent()


def _render_material_flow() -> None:
    _render_flow_back_to_home("material")
    _render_material_conversation_agent()


def _render_travel_flow() -> None:
    _render_flow_back_to_home("travel")
    _render_travel_workbench()
    with st.expander("兼容入口：手工槽位校对（可选）", expanded=False):
        go_section = travel_processing.render_travel_transport_section("1) 出差去程交通报销", "travel_go")
        return_section = travel_processing.render_travel_transport_section("2) 出差返程交通报销", "travel_return")
        hotel_section = travel_processing.render_travel_hotel_section("travel_hotel")
        travel_processing.render_travel_summary(go_section, return_section, hotel_section)


def main() -> None:
    st.set_page_config(page_title="Finance Agent", layout="wide")
    _inject_ui_styles()
    workbench.inject_workbench_styles()
    st.title("报销 Agent Workbench")
    st.caption("左侧切任务，中间继续对话和上传，右侧查看当前结果与待确认动作。")

    material_usecase.init_app_runtime()
    _ensure_router_state()
    task_hub.ensure_task_hub_state()
    sidebar_action = task_hub.render_task_sidebar(
        current_page=str(st.session_state.get("current_page") or PAGE_HOME_GUIDE),
        material_tasks=_list_material_sidebar_tasks(),
    )
    _handle_workbench_sidebar_action(sidebar_action)
    _render_model_runtime_panel()
    flash_message = _pop_router_flash_message()
    if flash_message:
        st.success(flash_message)

    current_page = str(st.session_state.get("current_page") or PAGE_HOME_GUIDE)
    if current_page == PAGE_HOME_GUIDE:
        _render_home_guide_agent()
    elif current_page == PAGE_MATERIAL_FLOW:
        _render_material_flow()
    elif current_page == PAGE_TRAVEL_FLOW:
        _render_travel_flow()
    else:
        _set_current_page(PAGE_HOME_GUIDE)
        st.rerun()


if __name__ == "__main__":
    main()
