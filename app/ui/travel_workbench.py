from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import streamlit as st

from app.agents import AgentCommand
from app.ui import task_hub, workbench
from app.ui.agent_metrics import render_agent_metric_caption as _render_agent_metric_caption
from app.ui.chat_widgets import (
    compose_three_stage_reply as _compose_three_stage_reply,
    render_chat_messages as _render_chat_messages,
    travel_chat_input_with_files as _travel_chat_input_with_files,
)
from app.ui.pending_actions import (
    clear_last_applied_action as _clear_last_applied_action,
    clear_pending_actions as _clear_pending_actions,
    get_pending_actions as _get_pending_actions,
    record_last_applied_action as _record_last_applied_action,
    remove_pending_action as _remove_pending_action,
    update_pending_action as _update_pending_action,
)
from app.usecases import travel_agent as travel_usecase
from app.usecases import travel_chat_service as travel_chat_usecase


_DEPENDENCIES: dict[str, Callable[..., Any]] = {}


def configure_travel_workbench(**dependencies: Callable[..., Any]) -> None:
    _DEPENDENCIES.update({key: value for key, value in dependencies.items() if callable(value)})


def _require_dependency(name: str) -> Callable[..., Any]:
    dependency = _DEPENDENCIES.get(name)
    if not callable(dependency):
        raise RuntimeError(f"travel_workbench dependency is not configured: {name}")
    return dependency


def _run_conversation_agent_task(objective: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str, list[Any]]:
    return _require_dependency("run_conversation_agent_task")(objective, payload)


def _execute_agent_command(command: AgentCommand, *, scope: str | None = None) -> tuple[bool, dict[str, Any], str]:
    return _require_dependency("execute_agent_command")(command, scope=scope)


def classify_user_message_intent(message: str, context: dict[str, Any] | None = None) -> Any:
    return _require_dependency("classify_user_message_intent")(message, context)


def _build_travel_execution_payload(**kwargs: Any) -> dict[str, Any]:
    return _require_dependency("build_travel_execution_payload")(**kwargs)


def _append_travel_pending_action_from_spec(scope_name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    return _require_dependency("append_travel_pending_action_from_spec")(scope_name, spec)


def _apply_manual_overrides_to_profiles(profiles: list[dict[str, Any]], manual_overrides: dict[str, str]) -> int:
    return _require_dependency("apply_manual_overrides_to_profiles")(profiles, manual_overrides)


def _apply_manual_slot_overrides_to_profiles(profiles: list[dict[str, Any]], manual_slot_overrides: dict[str, str]) -> int:
    return _require_dependency("apply_manual_slot_overrides_to_profiles")(profiles, manual_slot_overrides)


def _as_uploaded_list(uploaded_value: Any) -> list[Any]:
    return _require_dependency("as_uploaded_list")(uploaded_value)


def _build_assignment_from_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return _require_dependency("build_assignment_from_profiles")(profiles)


def _build_travel_agent_status(assignment: dict[str, Any]) -> dict[str, Any]:
    return _require_dependency("build_travel_agent_status")(assignment)


def _build_travel_file_profile(uploaded_file: Any, index: int) -> dict[str, Any]:
    return _require_dependency("build_travel_file_profile")(uploaded_file, index)


def _build_travel_handoff_status_reply(*, profiles: list[dict[str, Any]], status: dict[str, Any], guide_files: list[Any]) -> str:
    return _require_dependency("build_travel_handoff_status_reply")(profiles=profiles, status=status, guide_files=guide_files)


def _clone_travel_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return _require_dependency("clone_travel_profile")(profile)


def _doc_type_label(doc_type: str) -> str:
    return _require_dependency("doc_type_label")(doc_type)


def _files_signature(files: list[Any]) -> str:
    return _require_dependency("files_signature")(files)


def _format_amount(value: float | None) -> str:
    return _require_dependency("format_amount")(value)


def _generate_travel_agent_reply_llm(user_text: str, assignment: dict[str, Any], status: dict[str, Any], profiles: list[dict[str, Any]], messages: list[dict[str, Any]]) -> str | None:
    return _require_dependency("generate_travel_agent_reply_llm")(user_text, assignment, status, profiles, messages)


def _generate_travel_agent_reply_rule(user_text: str, assignment: dict[str, Any], status: dict[str, Any], profiles: list[dict[str, Any]]) -> str:
    return _require_dependency("generate_travel_agent_reply_rule")(user_text, assignment, status, profiles)


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    return _require_dependency("merge_uploaded_lists")(first, second)


def _profile_file_key(profile: dict[str, Any]) -> str:
    return _require_dependency("profile_file_key")(profile)


def _prune_manual_overrides(manual_overrides: dict[str, str], pool_files: list[Any]) -> None:
    _require_dependency("prune_manual_overrides")(manual_overrides, pool_files)


def _prune_manual_slot_overrides(manual_slot_overrides: dict[str, str], pool_files: list[Any]) -> None:
    _require_dependency("prune_manual_slot_overrides")(manual_slot_overrides, pool_files)


def _safe_float(value: Any) -> float | None:
    return _require_dependency("safe_float")(value)


def _slot_label(slot: str) -> str:
    return _require_dependency("slot_label")(slot)


def _travel_execute_pending_action(
    action: dict[str, Any],
    pool_list: list[Any],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]]]:
    return _require_dependency("travel_execute_pending_action")(
        action,
        pool_list,
        assignment,
        profiles,
        manual_overrides,
        manual_slot_overrides,
    )


def _travel_pending_action_spec_from_text(user_text: str) -> dict[str, Any] | None:
    return _require_dependency("travel_pending_action_spec_from_text")(user_text)


def _travel_pop_undo_snapshot(task_id: str | None = None) -> dict[str, Any] | None:
    return _require_dependency("travel_pop_undo_snapshot")(task_id)


def _travel_push_undo_snapshot(
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
    *,
    task_id: str | None = None,
) -> None:
    _require_dependency("travel_push_undo_snapshot")(
        assignment,
        profiles,
        manual_overrides,
        manual_slot_overrides,
        task_id=task_id,
    )


def _travel_restore_undo_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, str]]:
    return _require_dependency("travel_restore_undo_snapshot")(snapshot)


def _travel_scope_name(task_id: str | None = None) -> str:
    return _require_dependency("travel_scope_name")(task_id)


def _travel_undo_stack_key(task_id: str | None = None) -> str:
    return _require_dependency("travel_undo_stack_key")(task_id)


def _uploaded_file_key(uploaded_file: Any) -> str:
    return _require_dependency("uploaded_file_key")(uploaded_file)


def _render_travel_conversation_agent() -> dict[str, Any]:
    return _render_travel_workbench()


def _travel_stage_label(pool_list: list[Any], status: dict[str, Any], pending_actions: list[dict[str, Any]]) -> str:
    if not pool_list:
        return "草稿"
    if pending_actions:
        return "待确认"
    if bool(status.get("complete")):
        return "可导出"
    return "处理中"


def _travel_summary_text(profiles: list[dict[str, Any]], status: dict[str, Any], pool_list: list[Any]) -> str:
    if not pool_list:
        return "还没有上传材料。"
    return (
        f"已识别 {len(profiles)} 份材料，缺件 {len(status.get('missing') or [])} 项，"
        f"异常 {len(status.get('issues') or [])} 项。"
    )


def _travel_issue_text(status: dict[str, Any]) -> str:
    if status.get("missing"):
        return "当前缺件：" + "、".join(list(status.get("missing") or [])[:3])
    if status.get("issues"):
        return "当前异常：" + "；".join(list(status.get("issues") or [])[:2])
    return ""


def _travel_next_step_text(pool_list: list[Any], status: dict[str, Any], pending_actions: list[dict[str, Any]]) -> str:
    if not pool_list:
        return "先上传材料，然后点“重新整理材料”。"
    if pending_actions:
        return "右侧还有待确认动作，确认后我会同步更新结果。"
    if status.get("complete"):
        return "右侧结果已经完整，可以直接导出当前整理结果。"
    return "先补齐缺件或继续指出错误分类，我会重新整理。"


def _save_travel_workspace_snapshot(
    task_id: str,
    *,
    workspace: dict[str, Any],
    pool_list: list[Any],
    messages: list[dict[str, Any]],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
    current_signature: str,
    guide_payload: dict[str, Any],
) -> None:
    workspace["files"] = list(pool_list)
    workspace["messages"] = list(messages)
    workspace["assignment"] = dict(assignment or {})
    workspace["profiles"] = list(profiles or [])
    workspace["manual_overrides"] = dict(manual_overrides or {})
    workspace["manual_slot_overrides"] = dict(manual_slot_overrides or {})
    in_flight = dict(workspace.get("recognition_job") or {})
    in_flight_signature = str(in_flight.get("signature") or "")
    try:
        in_flight_total = int(in_flight.get("total") or 0)
    except (TypeError, ValueError):
        in_flight_total = 0
    try:
        in_flight_next = int(in_flight.get("next_index") or 0)
    except (TypeError, ValueError):
        in_flight_next = 0
    if (
        in_flight_signature
        and in_flight_signature == str(current_signature or "")
        and in_flight_total > 0
        and in_flight_next < in_flight_total
    ):
        workspace["pool_signature"] = str(workspace.get("pool_signature") or "")
    else:
        workspace["pool_signature"] = str(current_signature or "")
    workspace["guide_payload"] = dict(guide_payload or {})
    task_hub.save_travel_workspace(task_id, workspace)


def _travel_profile_preview_key(profile: dict[str, Any], index: int) -> str:
    key = _profile_file_key(profile)
    if key:
        return key
    return f"{index}:{str(profile.get('name') or '')}:{str(profile.get('profile_id') or '')}"


def _render_travel_file_preview(profile: dict[str, Any]) -> None:
    file_obj = profile.get("file")
    name = str(profile.get("name") or getattr(file_obj, "name", "") or "未命名文件")
    if file_obj is None or not hasattr(file_obj, "getvalue"):
        st.info("当前文件对象不在会话里，重新上传后可以预览。")
        return

    try:
        file_bytes = file_obj.getvalue()
    except Exception:
        file_bytes = b""
    if not file_bytes:
        st.info("当前文件没有可预览内容。")
        return

    suffix = Path(name).suffix.lower()
    st.caption(f"{name} · {_doc_type_label(str(profile.get('doc_type') or 'unknown'))} · {_slot_label(str(profile.get('slot') or 'unknown'))}")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        st.image(file_bytes, caption=name, use_column_width=True)
        return
    if suffix == ".pdf":
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f"""
<iframe
  title="{name}"
  src="data:application/pdf;base64,{encoded}"
  width="100%"
  height="520"
  style="border: 1px solid #e5e7eb; border-radius: 12px;"
></iframe>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "下载 PDF",
            data=file_bytes,
            file_name=name,
            mime="application/pdf",
            use_container_width=True,
            key=f"travel_preview_download_{hashlib.sha1(file_bytes[:1024]).hexdigest()}",
        )
        return

    st.download_button(
        "下载文件",
        data=file_bytes,
        file_name=name,
        use_container_width=True,
        key=f"travel_preview_download_{hashlib.sha1(file_bytes[:1024]).hexdigest()}",
    )


def _render_travel_profile_preview_table(
    profiles: list[dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    if not profiles:
        st.info("当前还没有文件结果。")
        return

    preview_state_key = f"{key_prefix}_selected_preview"
    selected_key = str(st.session_state.get(preview_state_key) or "")
    selected_profile: dict[str, Any] | None = None
    for idx, profile in enumerate(profiles):
        if _travel_profile_preview_key(profile, idx) == selected_key:
            selected_profile = profile
            break

    if selected_profile:
        with st.container(border=True):
            top_left, top_right = st.columns([5, 1])
            top_left.markdown(f"**预览：{selected_profile.get('name') or '文件'}**")
            if top_right.button("关闭", key=f"{key_prefix}_close_preview", use_container_width=True):
                st.session_state.pop(preview_state_key, None)
                st.rerun()
            _render_travel_file_preview(selected_profile)

    st.markdown('<div class="travel-preview-table">', unsafe_allow_html=True)
    header_cols = st.columns([0.72, 2.5, 1.35, 1.75, 0.9, 1.15, 1.25])
    for col, label in zip(header_cols, ["预览", "文件名", "识别类型", "分配槽位", "金额", "日期", "识别来源"], strict=False):
        col.markdown(f'<div class="travel-preview-head">{label}</div>', unsafe_allow_html=True)

    for idx, profile in enumerate(profiles):
        st.markdown('<div class="travel-preview-row">', unsafe_allow_html=True)
        row_cols = st.columns([0.72, 2.5, 1.35, 1.75, 0.9, 1.15, 1.25])
        profile_key = _travel_profile_preview_key(profile, idx)
        if row_cols[0].button("预览", key=f"{key_prefix}_preview_{idx}_{hashlib.sha1(profile_key.encode('utf-8', errors='ignore')).hexdigest()[:10]}"):
            st.session_state[preview_state_key] = profile_key
            st.rerun()
        row_cols[1].markdown(str(profile.get("name") or ""))
        row_cols[2].markdown(_doc_type_label(str(profile.get("doc_type") or "unknown")))
        row_cols[3].markdown(_slot_label(str(profile.get("slot") or "unknown")))
        row_cols[4].markdown(_format_amount(_safe_float(profile.get("amount"))))
        row_cols[5].markdown(str(profile.get("date") or ""))
        row_cols[6].markdown(str(profile.get("source") or ""))
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_travel_workbench() -> dict[str, Any]:
    active_task_id = task_hub.get_active_travel_task_id()
    if not active_task_id:
        active_task_id = task_hub.create_travel_task(title=f"差旅任务 {datetime.now().strftime('%m-%d %H:%M')}")
    task_meta = next(
        (item for item in task_hub.list_travel_tasks() if str(item.get("task_id") or "") == active_task_id),
        {},
    )
    workspace = task_hub.get_or_create_travel_workspace(active_task_id)
    assignment = dict(workspace.get("assignment") or {})
    profiles = list(workspace.get("profiles") or [])
    messages = list(workspace.get("messages") or [])
    manual_overrides = dict(workspace.get("manual_overrides") or {})
    manual_slot_overrides = dict(workspace.get("manual_slot_overrides") or {})
    guide_payload = dict(workspace.get("guide_payload") or {})
    scope_name = _travel_scope_name(active_task_id)

    upload_key = f"travel_agent_pool_files_{active_task_id}"
    user_input, attached_files = _travel_chat_input_with_files(key=f"travel_agent_chat_input_{active_task_id}")
    center_col, right_col = st.columns([1.7, 1], gap="large")
    with center_col:
        st.markdown('<span class="travel-workbench-anchor"></span>', unsafe_allow_html=True)
        page_uploaded_files = _as_uploaded_list(st.session_state.get(upload_key))
        stored_files = list(workspace.get("files") or [])
        pool_list = _merge_uploaded_lists(stored_files, page_uploaded_files)
        current_signature = _files_signature(pool_list)
        _prune_manual_overrides(manual_overrides, pool_list)
        _prune_manual_slot_overrides(manual_slot_overrides, pool_list)

        action_left, action_mid, action_right = st.columns([1, 1, 1])
        refresh_clicked = action_left.button("重新整理材料", use_container_width=True, key=f"travel_agent_refresh_{active_task_id}")
        clear_chat_clicked = action_mid.button("清空对话", use_container_width=True, key=f"travel_agent_clear_chat_{active_task_id}")
        with action_right:
            with st.popover("更多操作", use_container_width=True):
                clear_cache_clicked = st.button("清空识别缓存", use_container_width=True, key=f"travel_agent_clear_cache_{active_task_id}")
        recognition_progress_placeholder = st.empty()
        recognition_log_placeholder = st.empty()

    if clear_chat_clicked:
        messages = []
    if clear_cache_clicked:
        st.cache_data.clear()
        assignment = {}
        profiles = []
        manual_overrides = {}
        manual_slot_overrides = {}
        workspace.pop("recognition_job", None)
        workspace["pool_signature"] = ""
        workspace["handoff_summary_token"] = ""
        st.session_state.pop(_travel_undo_stack_key(active_task_id), None)
        _clear_pending_actions(scope_name)
        _clear_last_applied_action(scope_name)
        _save_travel_workspace_snapshot(
            active_task_id,
            workspace=workspace,
            pool_list=pool_list,
            messages=messages,
            assignment=assignment,
            profiles=profiles,
            manual_overrides=manual_overrides,
            manual_slot_overrides=manual_slot_overrides,
            current_signature="",
            guide_payload=guide_payload,
        )
        st.success("识别缓存已清空，请重新点击“重新整理材料”。")
        st.rerun()

    recognition_job = dict(workspace.get("recognition_job") or {})
    job_signature = str(recognition_job.get("signature") or "")
    try:
        job_total = int(recognition_job.get("total") or 0)
    except (TypeError, ValueError):
        job_total = 0
    try:
        job_next_index = int(recognition_job.get("next_index") or 0)
    except (TypeError, ValueError):
        job_next_index = 0

    def _seed_incremental_recognition(
        files: list[Any],
        existing_profiles: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        existing_map: dict[str, dict[str, Any]] = {}
        for item in list(existing_profiles or []):
            if not isinstance(item, dict):
                continue
            key = _profile_file_key(item)
            if key and key not in existing_map:
                existing_map[key] = _clone_travel_profile(item)

        seeded_profiles: list[dict[str, Any]] = []
        seeded_logs: list[dict[str, Any]] = []
        next_index = 0
        for idx, uploaded_file in enumerate(list(files or [])):
            key = _uploaded_file_key(uploaded_file)
            profile = existing_map.get(key)
            if not profile:
                next_index = idx
                break
            profile["index"] = idx
            profile["file"] = uploaded_file
            profile["name"] = str(getattr(uploaded_file, "name", "") or profile.get("name") or "")
            seeded_profiles.append(profile)
            timing = dict(profile.get("timing") or {})
            seeded_logs.append(
                {
                    "index": idx + 1,
                    "name": str(profile.get("name") or ""),
                    "doc_type_label": _doc_type_label(str(profile.get("doc_type") or "unknown")),
                    "slot_label": _slot_label(str(profile.get("slot") or "unknown")),
                    "elapsed_sec": float(timing.get("total_sec") or 0.0),
                    "ocr_sec": float(timing.get("ocr_sec") or 0.0),
                    "classify_sec": float(timing.get("classify_sec") or 0.0),
                }
            )
            next_index = idx + 1
        else:
            next_index = len(files)

        return seeded_profiles, seeded_logs, next_index

    if not pool_list:
        recognition_job = {}
        workspace.pop("recognition_job", None)
    else:
        should_start_rebuild = bool(refresh_clicked) or (
            job_signature != current_signature
            and (str(workspace.get("pool_signature") or "") != current_signature or not assignment)
        )
        if should_start_rebuild:
            seeded_profiles: list[dict[str, Any]] = []
            seeded_logs: list[dict[str, Any]] = []
            next_index = 0
            if not refresh_clicked and profiles:
                seeded_profiles, seeded_logs, next_index = _seed_incremental_recognition(pool_list, profiles)
            if seeded_profiles:
                _apply_manual_overrides_to_profiles(seeded_profiles, manual_overrides)
                _apply_manual_slot_overrides_to_profiles(seeded_profiles, manual_slot_overrides)
                assignment = _build_assignment_from_profiles(seeded_profiles)
            else:
                assignment = {}
            profiles = list(seeded_profiles)
            recognition_job = {
                "signature": current_signature,
                "total": len(pool_list),
                "next_index": int(next_index),
                "profiles": list(seeded_profiles),
                "logs": list(seeded_logs[-120:]),
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            workspace["recognition_job"] = recognition_job
            workspace["assignment"] = dict(assignment)
            workspace["profiles"] = list(profiles)
            job_signature = current_signature
            job_total = len(pool_list)
            job_next_index = int(next_index)
        elif job_signature == current_signature:
            in_progress_profiles = list(recognition_job.get("profiles") or [])
            if in_progress_profiles:
                profiles = in_progress_profiles
                _apply_manual_overrides_to_profiles(profiles, manual_overrides)
                _apply_manual_slot_overrides_to_profiles(profiles, manual_slot_overrides)
                assignment = _build_assignment_from_profiles(profiles)
                workspace["assignment"] = dict(assignment)
                workspace["profiles"] = list(profiles)
        elif pool_list and not assignment:
            recognition_job = {
                "signature": current_signature,
                "total": len(pool_list),
                "next_index": 0,
                "profiles": [],
                "logs": [],
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            workspace["recognition_job"] = recognition_job
            assignment = {}
            profiles = []
            job_signature = current_signature
            job_total = len(pool_list)
            job_next_index = 0

    is_recognition_active = (
        bool(pool_list)
        and job_signature == current_signature
        and job_total == len(pool_list)
        and job_next_index < job_total
    )
    recognition_done_count = min(max(job_next_index, 0), max(job_total, 0))
    recognition_logs = list(recognition_job.get("logs") or [])

    def _render_recognition_progress_state(done_count: int, total_count: int, logs: list[dict[str, Any]]) -> None:
        if total_count <= 0:
            recognition_progress_placeholder.empty()
            recognition_log_placeholder.empty()
            return
        safe_total = max(total_count, 1)
        progress_ratio = min(max(done_count / safe_total, 0.0), 1.0)
        next_name = ""
        if done_count < len(pool_list):
            next_name = str(getattr(pool_list[done_count], "name", "") or "")
        def _safe_num(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        elapsed_values: list[float] = []
        for item in list(logs or []):
            value = _safe_num(item.get("elapsed_sec"))
            if value > 0:
                elapsed_values.append(value)
        eta_suffix = ""
        if elapsed_values and done_count < total_count:
            avg_elapsed = sum(elapsed_values) / len(elapsed_values)
            remaining = max(total_count - done_count, 0)
            eta_sec = avg_elapsed * remaining
            eta_suffix = f"，预计剩余 {eta_sec:.0f}s"
        if next_name:
            recognition_progress_placeholder.progress(
                progress_ratio,
                text=f"Agent 正在识别分配材料：{done_count}/{total_count}，下一份：{next_name}{eta_suffix}",
            )
        else:
            recognition_progress_placeholder.progress(
                progress_ratio,
                text=f"Agent 正在识别分配材料：{done_count}/{total_count}{eta_suffix}",
            )
        recent_logs = list(logs or [])[-8:]
        if recent_logs:
            recognition_log_placeholder.dataframe(
                [
                    {
                        "进度": f"{int(item.get('index') or 0)}/{total_count}",
                        "文件名": str(item.get("name") or ""),
                        "识别类型": str(item.get("doc_type_label") or ""),
                        "分配槽位": str(item.get("slot_label") or ""),
                        "总耗时(s)": f"{_safe_num(item.get('elapsed_sec')):.2f}",
                        "OCR(s)": f"{_safe_num(item.get('ocr_sec')):.2f}",
                        "分类(s)": f"{_safe_num(item.get('classify_sec')):.2f}",
                    }
                    for item in recent_logs
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            recognition_log_placeholder.empty()

    if is_recognition_active:
        _render_recognition_progress_state(recognition_done_count, job_total, recognition_logs)
    elif recognition_done_count <= 0:
        recognition_progress_placeholder.empty()
        recognition_log_placeholder.empty()

    if not messages:
        messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "我已经接管这个差旅任务。",
                    "你可以在中间继续对话、补材料，我会把结果同步到右侧结果面板。",
                    "先把所有票据交给我，或者直接问“现在还缺什么”。",
                ),
            }
        )

    status = (
        _build_travel_agent_status(assignment)
        if pool_list
        else {"missing": [], "issues": [], "issue_items": [], "tips": [], "complete": False}
    )
    pending_actions = [item for item in _get_pending_actions(scope_name) if str(item.get("status") or "pending") == "pending"]
    if guide_payload and pool_list:
        handoff_token = f"{str(guide_payload.get('recommended_flow') or '')}|{current_signature}|{len(pool_list)}"
        if str(workspace.get("handoff_summary_token") or "") != handoff_token:
            messages.append(
                {
                    "role": "assistant",
                    "content": _build_travel_handoff_status_reply(
                        profiles=profiles,
                        status=status,
                        guide_files=pool_list,
                    ),
                }
            )
            workspace["handoff_summary_token"] = handoff_token

    stage_label = _travel_stage_label(pool_list, status, pending_actions)
    summary_text = _travel_summary_text(profiles, status, pool_list)
    workbench.render_case_header(
        title=str(task_meta.get("title") or f"差旅任务 {active_task_id[-4:]}"),
        task_type_label="差旅",
        stage_label=stage_label,
        goal=str(task_meta.get("goal") or "整理本次差旅报销"),
        summary=summary_text,
        issue_text=_travel_issue_text(status),
        next_step=_travel_next_step_text(pool_list, status, pending_actions),
    )
    workbench.render_stat_strip(
        [
            ("已识别材料", len(profiles)),
            ("缺件数", len(status.get("missing") or [])),
            ("异常数", len(status.get("issues") or [])),
            ("待确认", len(pending_actions)),
            ("状态", stage_label),
        ]
    )
    task_hub.update_travel_task(
        active_task_id,
        goal=str(task_meta.get("goal") or ""),
        status=stage_label,
        summary=summary_text,
        file_count=len(pool_list),
    )
    with center_col:
        backend_warning = str(st.session_state.pop("travel_agent_backend_warning", "") or "").strip()
        if backend_warning:
            st.warning(f"差旅 Agent 后台暂不可用，已回退旧链路：{backend_warning}")
        if guide_payload:
            with st.expander("查看首页立案摘要", expanded=False):
                st.json(guide_payload)
        _render_agent_metric_caption("travel")
        _render_chat_messages(messages, stream_state_key=f"travel_chat_streamed_{active_task_id}")

        if user_input or attached_files:
            if attached_files:
                next_pool_list = _merge_uploaded_lists(pool_list, attached_files)
                upload_text = user_input or f"我上传了 {len(attached_files)} 份差旅材料，请先识别整理。"
                messages.append({"role": "user", "content": upload_text})
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            f"收到，这次新增 {len(attached_files)} 份材料。",
                            "我会先把新材料纳入当前差旅任务并逐份识别分类。",
                            "识别完成后你可以直接问“现在还缺什么”，或点右侧预览对比具体文件。",
                        ),
                    }
                )
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=next_pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()

            messages.append({"role": "user", "content": user_input})
            plan_ok, plan_payload, plan_summary, plan_commands = _run_conversation_agent_task(
                "plan_travel_turn",
                {
                    "user_text": user_input,
                    "intent_parser": classify_user_message_intent,
                    "intent_context": {
                        "domain": "travel",
                        "missing_count": len(status.get("missing") or []),
                        "issue_count": len(status.get("issues") or []),
                        "pending_count": len(pending_actions),
                    },
                    "pending_action_builder": _travel_pending_action_spec_from_text,
                    "reply_llm": _generate_travel_agent_reply_llm,
                    "reply_rule": _generate_travel_agent_reply_rule,
                    "assignment": assignment,
                    "status": status,
                    "profiles": profiles,
                    "messages": messages,
                    "summary_text": summary_text,
                    "execution_payload": _build_travel_execution_payload(
                        pool_list=pool_list,
                        assignment=assignment,
                        profiles=profiles,
                        manual_overrides=manual_overrides,
                        manual_slot_overrides=manual_slot_overrides,
                    ),
                },
            )
            planned_intent = dict(plan_payload.get("intent") or {})
            intent_type = str(planned_intent.get("intent_type") or "chat")
            needs_confirmation = bool(planned_intent.get("needs_confirmation"))

            if not plan_ok:
                # Fallback only when planner is unavailable; normal path is LLM-first.
                chat_context = travel_chat_usecase.ensure_travel_chat_context(workspace.get("travel_chat_context"))
                chat_query = travel_chat_usecase.parse_travel_chat_query(
                    user_input,
                    {
                        "assignment": assignment,
                        "status": status,
                        "profile_count": len(profiles),
                        "chat_context": chat_context.model_dump(),
                    },
                )
                chat_payload = travel_chat_usecase.execute_travel_chat_query(chat_query, assignment, status)
                next_chat_context = travel_chat_usecase.update_travel_chat_context(chat_context, chat_query, chat_payload)
                workspace["travel_chat_context"] = next_chat_context.model_dump()
                fallback_reply = travel_chat_usecase.render_travel_chat_answer(chat_payload).strip()
                messages.append(
                    {
                        "role": "assistant",
                        "content": fallback_reply
                        or _compose_three_stage_reply(
                            "我看到了你的输入。",
                            plan_summary or "这轮LLM规划不可用，已回退到规则问答。",
                            "你可以重试一次，或者直接告诉我具体文件名和目标类型。",
                        ),
                    }
                )
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()

            if intent_type == "strong_action" and needs_confirmation:
                action = _append_travel_pending_action_from_spec(
                    scope_name,
                    dict(plan_payload.get("pending_action_spec") or {}),
                )
                if action:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": str(plan_payload.get("reply") or "").strip()
                            or _compose_three_stage_reply(
                                "我理解你的意图，这是一个会影响整体结果的操作。",
                                f"我已把它放进右侧“待确认”：{action.get('summary') or '待确认动作'}。",
                                "你可以在右侧逐条确认，或者让我继续解释这条建议的影响。",
                            ),
                        }
                    )
                    _save_travel_workspace_snapshot(
                        active_task_id,
                        workspace=workspace,
                        pool_list=pool_list,
                        messages=messages,
                        assignment=assignment,
                        profiles=profiles,
                        manual_overrides=manual_overrides,
                        manual_slot_overrides=manual_slot_overrides,
                        current_signature=current_signature,
                        guide_payload=guide_payload,
                    )
                    st.rerun()

            if intent_type == "light_edit":
                _travel_push_undo_snapshot(
                    assignment,
                    profiles,
                    manual_overrides,
                    manual_slot_overrides,
                    task_id=active_task_id,
                )
                command = plan_commands[0] if plan_commands else AgentCommand(
                    command_type="travel_light_edit",
                    payload={
                        "user_text": user_input,
                        **_build_travel_execution_payload(
                            pool_list=pool_list,
                            assignment=assignment,
                            profiles=profiles,
                            manual_overrides=manual_overrides,
                            manual_slot_overrides=manual_slot_overrides,
                        ),
                    },
                    summary="执行差旅轻修正",
                    risk_level="low",
                    requires_confirmation=False,
                    created_by="travel_workbench_fallback",
                )
                with st.spinner("正在应用轻量修正..."):
                    edit_ok, edit_payload, edit_summary = _execute_agent_command(command)
                    assignment = dict(edit_payload.get("assignment") or assignment)
                    profiles = list(edit_payload.get("profiles") or profiles)
                result_type = str(edit_payload.get("result_type") or "")
                total_changed = int(edit_payload.get("total_changed") or 0)
                if edit_ok and total_changed > 0:
                    _record_last_applied_action(
                        scope_name,
                        {
                            "action_id": uuid4().hex,
                            "action_type": "travel_light_edit",
                            "summary": f"轻修正 {total_changed} 项",
                        },
                    )
                target_slot_value = str(edit_payload.get("target_slot") or "")
                reply_ok, reply_payload, _, _ = _run_conversation_agent_task(
                    "compose_travel_edit_reply",
                    {
                        "execution_ok": edit_ok,
                        "execution_summary": edit_summary,
                        "result_type": result_type,
                        "total_changed": total_changed,
                        "slot_changed_count": int(edit_payload.get("slot_changed_count") or 0),
                        "slot_changed_names": list(edit_payload.get("slot_changed_names") or []),
                        "target_slot_label": _slot_label(target_slot_value) if target_slot_value else "",
                        "changed_count": int(edit_payload.get("changed_count") or 0),
                        "changed_names": list(edit_payload.get("changed_names") or []),
                        "target_doc_type_label": _doc_type_label(str(edit_payload.get("target_doc_type") or ""))
                        if edit_payload.get("target_doc_type")
                        else "",
                        "amount_changed_count": int(edit_payload.get("amount_changed_count") or 0),
                        "amount_changed_names": list(edit_payload.get("amount_changed_names") or []),
                        "manual_amount_text": _format_amount(_safe_float(edit_payload.get("manual_amount"))),
                    },
                )
                reply_text = str(reply_payload.get("reply") or "").strip()
                if not reply_ok or not reply_text:
                    reply_text = _compose_three_stage_reply(
                        "我理解你的修改意图了。",
                        edit_summary or "这次没有命中可执行变更。",
                        "你可以继续说“改到返程机票明细/去程支付记录”，或者再点一次文件让我直接改。",
                    )
                messages.append({"role": "assistant", "content": reply_text})
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()

            if intent_type == "ambiguous":
                messages.append(
                    {
                        "role": "assistant",
                        "content": str(plan_payload.get("reply") or "").strip()
                        or _compose_three_stage_reply(
                            "我理解你觉得结果有点不对。",
                            summary_text,
                            "你可以告诉我具体文件名和目标类型，我会先给出调整再请你确认。",
                        ),
                    }
                )
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()

            reply = str(plan_payload.get("reply") or "").strip()
            if not reply:
                reply = _generate_travel_agent_reply_rule(user_input, assignment, status, profiles)
            messages.append({"role": "assistant", "content": reply})
            _save_travel_workspace_snapshot(
                active_task_id,
                workspace=workspace,
                pool_list=pool_list,
                messages=messages,
                assignment=assignment,
                profiles=profiles,
                manual_overrides=manual_overrides,
                manual_slot_overrides=manual_slot_overrides,
                current_signature=current_signature,
                guide_payload=guide_payload,
            )
            st.rerun()

    with right_col:
        st.markdown('<span class="travel-side-panel-anchor"></span>', unsafe_allow_html=True)
        overview_tab, result_tab = st.tabs(["概览", "结果"])
        with overview_tab:
            if status.get("missing"):
                st.warning("仍缺材料：" + "、".join(status.get("missing") or []))
            else:
                st.success("必需材料已齐全。")
            if status.get("issues"):
                st.error("发现核对问题：")
                for issue in status.get("issues") or []:
                    st.markdown(f"- {issue}")
            elif status.get("complete"):
                st.success("金额核对通过，可以导出材料。")
            for tip in status.get("tips") or []:
                st.info(tip)
            with st.expander("导出当前结果", expanded=False):
                _render_travel_package_export(task_id=active_task_id, assignment=assignment)

        with result_tab:
            workbench.render_trip_board(assignment)
            _render_travel_profile_preview_table(profiles, key_prefix=f"travel_result_{active_task_id}")

        pending_actions = [item for item in _get_pending_actions(scope_name) if str(item.get("status") or "pending") == "pending"]
        if pending_actions:
            pleft, pmid, pright = st.columns(3)
            apply_all = pleft.button("应用全部建议", key=f"travel_pending_apply_all_{active_task_id}", use_container_width=True, disabled=not pending_actions)
            undo_last = pmid.button("撤销上一步", key=f"travel_pending_undo_last_{active_task_id}", use_container_width=True)
            clear_pending = pright.button("清空待确认", key=f"travel_pending_clear_all_{active_task_id}", use_container_width=True, disabled=not pending_actions)
            if clear_pending:
                _clear_pending_actions(scope_name)
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()
            if undo_last:
                snapshot = _travel_pop_undo_snapshot(active_task_id)
                if snapshot:
                    assignment, profiles, manual_overrides, manual_slot_overrides = _travel_restore_undo_snapshot(snapshot)
                    _clear_last_applied_action(scope_name)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": _compose_three_stage_reply(
                                "好的，我收到你的撤销请求。",
                                "我已经恢复到上一步前的差旅分配状态。",
                                "你可以继续指出具体文件，我会按你的意思重新调整。",
                            ),
                        }
                    )
                    _save_travel_workspace_snapshot(
                        active_task_id,
                        workspace=workspace,
                        pool_list=pool_list,
                        messages=messages,
                        assignment=assignment,
                        profiles=profiles,
                        manual_overrides=manual_overrides,
                        manual_slot_overrides=manual_slot_overrides,
                        current_signature=current_signature,
                        guide_payload=guide_payload,
                    )
                    st.rerun()
            if pending_actions:
                for action in pending_actions:
                    action_id = str(action.get("action_id") or "")
                    if not action_id:
                        continue
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([7, 1, 1])
                        c1.markdown(
                            f"**{action.get('summary') or '待确认动作'}**  \n"
                            f"风险：`{action.get('risk_level') or 'medium'}`｜创建时间：`{action.get('created_at') or '-'}"
                        )
                        target_key = f"travel_pending_target_{active_task_id}_{action_id}"
                        current_target = str(action.get("target") or "")
                        edited_target = c1.text_input("目标值（可改）", value=current_target, key=target_key, label_visibility="collapsed")
                        if edited_target != current_target:
                            payload = dict(action.get("payload") or {})
                            if payload.get("command"):
                                payload["command"] = edited_target
                            _update_pending_action(scope_name, action_id, {"target": edited_target, "payload": payload})
                            action["target"] = edited_target
                        confirm_clicked = c2.button("确认", key=f"travel_pending_confirm_{active_task_id}_{action_id}", use_container_width=True)
                        cancel_clicked = c3.button("取消", key=f"travel_pending_cancel_{active_task_id}_{action_id}", use_container_width=True)
                        if cancel_clicked:
                            _remove_pending_action(scope_name, action_id)
                            st.rerun()
                        if confirm_clicked:
                            _travel_push_undo_snapshot(
                                assignment,
                                profiles,
                                manual_overrides,
                                manual_slot_overrides,
                                task_id=active_task_id,
                            )
                            ok, msg, assignment, profiles = _travel_execute_pending_action(
                                action,
                                pool_list,
                                assignment,
                                profiles,
                                manual_overrides,
                                manual_slot_overrides,
                            )
                            if ok:
                                _record_last_applied_action(scope_name, action)
                                _remove_pending_action(scope_name, action_id)
                                try:
                                    travel_usecase.learn_from_profiles(profiles, assignment, reason="pending_confirm")
                                except Exception:
                                    pass
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": _compose_three_stage_reply(
                                            "好的，我已按你的确认执行。",
                                            msg,
                                            "你可以继续让我检查缺件、金额异常，或继续确认剩余建议。",
                                        ),
                                    }
                                )
                                _save_travel_workspace_snapshot(
                                    active_task_id,
                                    workspace=workspace,
                                    pool_list=pool_list,
                                    messages=messages,
                                    assignment=assignment,
                                    profiles=profiles,
                                    manual_overrides=manual_overrides,
                                    manual_slot_overrides=manual_slot_overrides,
                                    current_signature=current_signature,
                                    guide_payload=guide_payload,
                                )
                                st.rerun()
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": _compose_three_stage_reply(
                                        "我收到了你的确认请求。",
                                        msg,
                                        "你可以补充更具体的文件名和目标，或先让我解释这条动作为什么没执行。",
                                    ),
                                }
                            )
                            _save_travel_workspace_snapshot(
                                active_task_id,
                                workspace=workspace,
                                pool_list=pool_list,
                                messages=messages,
                                assignment=assignment,
                                profiles=profiles,
                                manual_overrides=manual_overrides,
                                manual_slot_overrides=manual_slot_overrides,
                                current_signature=current_signature,
                                guide_payload=guide_payload,
                            )
                            st.rerun()
            else:
                st.info("当前没有待确认动作。高风险指令会先放在这里，确认后再执行。")
            if apply_all and pending_actions:
                _travel_push_undo_snapshot(
                    assignment,
                    profiles,
                    manual_overrides,
                    manual_slot_overrides,
                    task_id=active_task_id,
                )
                success_count = 0
                failed_count = 0
                for action in list(pending_actions):
                    ok, msg, assignment, profiles = _travel_execute_pending_action(
                        action,
                        pool_list,
                        assignment,
                        profiles,
                        manual_overrides,
                        manual_slot_overrides,
                    )
                    if ok:
                        success_count += 1
                        _record_last_applied_action(scope_name, action)
                        _remove_pending_action(scope_name, str(action.get("action_id") or ""))
                    else:
                        failed_count += 1
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "好的，我已经处理你的批量确认。",
                            f"已批量执行 {success_count} 条待确认动作，未执行 {failed_count} 条。",
                            "你可以继续微调分类，或直接导出当前结果。",
                        ),
                    }
                )
                _save_travel_workspace_snapshot(
                    active_task_id,
                    workspace=workspace,
                    pool_list=pool_list,
                    messages=messages,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                    current_signature=current_signature,
                    guide_payload=guide_payload,
                )
                st.rerun()

    if is_recognition_active and recognition_done_count < len(pool_list):
        process_index = recognition_done_count
        target_file = pool_list[process_index]
        profile = _build_travel_file_profile(target_file, index=process_index)

        next_profiles = list(recognition_job.get("profiles") or [])
        if process_index < len(next_profiles):
            next_profiles[process_index] = profile
        else:
            next_profiles.append(profile)

        _apply_manual_overrides_to_profiles(next_profiles, manual_overrides)
        _apply_manual_slot_overrides_to_profiles(next_profiles, manual_slot_overrides)
        assignment = _build_assignment_from_profiles(next_profiles)
        profiles = list(next_profiles)

        next_logs = list(recognition_logs)
        next_logs.append(
            {
                "index": process_index + 1,
                "name": str(profile.get("name") or getattr(target_file, "name", "") or ""),
                "doc_type_label": _doc_type_label(str(profile.get("doc_type") or "unknown")),
                "slot_label": _slot_label(str(profile.get("slot") or "unknown")),
                "elapsed_sec": float(((profile.get("timing") or {}).get("total_sec") or 0.0)),
                "ocr_sec": float(((profile.get("timing") or {}).get("ocr_sec") or 0.0)),
                "classify_sec": float(((profile.get("timing") or {}).get("classify_sec") or 0.0)),
            }
        )
        recognition_done_count = process_index + 1
        recognition_logs = next_logs[-120:]

        if recognition_done_count >= len(pool_list):
            workspace.pop("recognition_job", None)
            workspace["pool_signature"] = current_signature
        else:
            recognition_job["signature"] = current_signature
            recognition_job["total"] = len(pool_list)
            recognition_job["next_index"] = recognition_done_count
            recognition_job["profiles"] = list(next_profiles)
            recognition_job["logs"] = list(recognition_logs)
            workspace["recognition_job"] = recognition_job

        _save_travel_workspace_snapshot(
            active_task_id,
            workspace=workspace,
            pool_list=pool_list,
            messages=messages,
            assignment=assignment,
            profiles=profiles,
            manual_overrides=manual_overrides,
            manual_slot_overrides=manual_slot_overrides,
            current_signature=current_signature,
            guide_payload=guide_payload,
        )
        st.rerun()

    _save_travel_workspace_snapshot(
        active_task_id,
        workspace=workspace,
        pool_list=pool_list,
        messages=messages,
        assignment=assignment,
        profiles=profiles,
        manual_overrides=manual_overrides,
        manual_slot_overrides=manual_slot_overrides,
        current_signature=current_signature,
        guide_payload=guide_payload,
    )
    return status


def _sanitize_export_name(name: str) -> str:
    return travel_usecase.sanitize_export_name(name)


def _safe_uploaded_filename(name: str, default_stem: str) -> str:
    return travel_usecase.safe_uploaded_filename(name, default_stem)


def _amount_suffix(amount: float | None) -> str:
    return travel_usecase.amount_suffix(amount)


def _zip_ensure_dir(zip_file, dir_path: str) -> None:
    travel_usecase.zip_ensure_dir(zip_file, dir_path)


def _zip_write_uploaded_files(zip_file, target_dir: str, files: list[Any]) -> None:
    travel_usecase.zip_write_uploaded_files(zip_file, target_dir, files)


def _build_travel_package_zip(
    package_name: str,
    go_ticket_files: list[Any],
    go_payment_files: list[Any],
    go_detail_files: list[Any],
    return_ticket_files: list[Any],
    return_payment_files: list[Any],
    return_detail_files: list[Any],
    hotel_invoice_files: list[Any],
    hotel_payment_files: list[Any],
    hotel_order_files: list[Any],
    go_ticket_amount: float | None,
    go_payment_amount: float | None,
    return_ticket_amount: float | None,
    return_payment_amount: float | None,
    hotel_invoice_amount: float | None,
    hotel_payment_amount: float | None,
) -> bytes:
    return travel_usecase.build_travel_package_zip(
        package_name=package_name,
        go_ticket_files=go_ticket_files,
        go_payment_files=go_payment_files,
        go_detail_files=go_detail_files,
        return_ticket_files=return_ticket_files,
        return_payment_files=return_payment_files,
        return_detail_files=return_detail_files,
        hotel_invoice_files=hotel_invoice_files,
        hotel_payment_files=hotel_payment_files,
        hotel_order_files=hotel_order_files,
        go_ticket_amount=go_ticket_amount,
        go_payment_amount=go_payment_amount,
        return_ticket_amount=return_ticket_amount,
        return_payment_amount=return_payment_amount,
        hotel_invoice_amount=hotel_invoice_amount,
        hotel_payment_amount=hotel_payment_amount,
    )


def _render_travel_package_export(*, task_id: str | None = None, assignment: dict[str, Any] | None = None) -> None:
    st.subheader("差旅材料打包导出")
    default_name = f"差旅报销材料_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_name = st.text_input(
        "压缩包名称（无需 .zip）",
        value=default_name,
        key=f"travel_export_package_name_{str(task_id or 'default')}",
    )

    active_assignment = dict(assignment or {})
    if not active_assignment:
        workspace = task_hub.get_or_create_travel_workspace(str(task_id or task_hub.get_active_travel_task_id() or "default"))
        active_assignment = dict(workspace.get("assignment") or {})

    go_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("go_ticket")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_file")),
    )
    go_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("go_payment")),
        _as_uploaded_list(st.session_state.get("travel_go_payment_file")),
    )
    go_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("go_detail")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_detail_file")),
    )

    return_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("return_ticket")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_file")),
    )
    return_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("return_payment")),
        _as_uploaded_list(st.session_state.get("travel_return_payment_file")),
    )
    return_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("return_detail")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_detail_file")),
    )

    hotel_invoice_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("hotel_invoice")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_invoice")),
    )
    hotel_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("hotel_payment")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_payment")),
    )
    hotel_order_files = _merge_uploaded_lists(
        _as_uploaded_list(active_assignment.get("hotel_order")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_order")),
    )

    go_ticket_amount = _safe_float(st.session_state.get("travel_go_ticket_amount"))
    go_payment_amount = _safe_float(st.session_state.get("travel_go_payment_amount"))
    return_ticket_amount = _safe_float(st.session_state.get("travel_return_ticket_amount"))
    return_payment_amount = _safe_float(st.session_state.get("travel_return_payment_amount"))
    hotel_invoice_amount = _safe_float(st.session_state.get("travel_hotel_ticket_amount"))
    hotel_payment_amount = _safe_float(st.session_state.get("travel_hotel_payment_amount"))

    if go_ticket_amount is None:
        go_ticket_amount = _safe_float(active_assignment.get("go_ticket_amount"))
    if go_payment_amount is None:
        go_payment_amount = _safe_float(active_assignment.get("go_payment_amount"))
    if return_ticket_amount is None:
        return_ticket_amount = _safe_float(active_assignment.get("return_ticket_amount"))
    if return_payment_amount is None:
        return_payment_amount = _safe_float(active_assignment.get("return_payment_amount"))
    if hotel_invoice_amount is None:
        hotel_invoice_amount = _safe_float(active_assignment.get("hotel_invoice_amount"))
    if hotel_payment_amount is None:
        hotel_payment_amount = _safe_float(active_assignment.get("hotel_payment_amount"))

    total_uploaded = (
        len(go_ticket_files)
        + len(go_payment_files)
        + len(go_detail_files)
        + len(return_ticket_files)
        + len(return_payment_files)
        + len(return_detail_files)
        + len(hotel_invoice_files)
        + len(hotel_payment_files)
        + len(hotel_order_files)
    )
    if total_uploaded == 0:
        st.info("请先上传差旅材料，再导出压缩包。")
        return

    missing_for_recommended = []
    if not go_ticket_files:
        missing_for_recommended.append("去程机票发票")
    if not go_payment_files:
        missing_for_recommended.append("去程支付记录")
    if not go_detail_files:
        missing_for_recommended.append("去程机票明细")
    if not return_ticket_files:
        missing_for_recommended.append("返程机票发票")
    if not return_payment_files:
        missing_for_recommended.append("返程支付记录")
    if not return_detail_files:
        missing_for_recommended.append("返程机票明细")
    if not hotel_invoice_files:
        missing_for_recommended.append("酒店发票")
    if not hotel_payment_files:
        missing_for_recommended.append("酒店支付记录")
    if not hotel_order_files:
        missing_for_recommended.append("酒店订单截图")

    if missing_for_recommended:
        st.warning(f"当前仍缺少：{'、'.join(missing_for_recommended)}。你仍可先导出已有材料。")

    zip_bytes = _build_travel_package_zip(
        package_name=package_name,
        go_ticket_files=go_ticket_files,
        go_payment_files=go_payment_files,
        go_detail_files=go_detail_files,
        return_ticket_files=return_ticket_files,
        return_payment_files=return_payment_files,
        return_detail_files=return_detail_files,
        hotel_invoice_files=hotel_invoice_files,
        hotel_payment_files=hotel_payment_files,
        hotel_order_files=hotel_order_files,
        go_ticket_amount=go_ticket_amount,
        go_payment_amount=go_payment_amount,
        return_ticket_amount=return_ticket_amount,
        return_payment_amount=return_payment_amount,
        hotel_invoice_amount=hotel_invoice_amount,
        hotel_payment_amount=hotel_payment_amount,
    )
    zip_file_name = f"{_sanitize_export_name(package_name)}.zip"
    st.download_button(
        label="导出差旅材料压缩包",
        data=zip_bytes,
        file_name=zip_file_name,
        mime="application/zip",
        use_container_width=True,
    )




def render_travel_workbench() -> dict[str, Any]:
    return _render_travel_workbench()


def render_travel_conversation_agent() -> dict[str, Any]:
    return _render_travel_conversation_agent()
