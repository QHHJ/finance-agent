from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from pypdf import PdfReader

from app.usecases import home_guide_agent as guide_usecase
from app.usecases import travel_agent as travel_usecase

PAGE_HOME_GUIDE = "home_guide"
PAGE_TRAVEL_FLOW = "travel_flow"
PAGE_MATERIAL_FLOW = "material_flow"
VALID_ROUTER_PAGES = {PAGE_HOME_GUIDE, PAGE_TRAVEL_FLOW, PAGE_MATERIAL_FLOW}


def _as_uploaded_list(uploaded_value) -> list[Any]:
    return travel_usecase.as_uploaded_list(uploaded_value)


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    return travel_usecase.merge_uploaded_lists(first, second)


def _extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = str(text).strip()
        if text:
            pages.append(text)
    return "\n".join(chunk for chunk in pages if chunk)


@st.cache_data(show_spinner=False)
def _extract_pdf_preview_text(file_bytes: bytes, max_chars: int = 1800) -> str:
    try:
        text = _extract_pdf_text_from_bytes(file_bytes)
    except Exception:
        return ""
    value = str(text or "").strip()
    if len(value) > max_chars:
        value = value[:max_chars]
    return value


def _home_guide_file_signature(files: list[Any]) -> str:
    parts: list[str] = []
    for file in files:
        name = str(getattr(file, "name", ""))
        size = str(getattr(file, "size", ""))
        parts.append(f"{name}:{size}")
    return "|".join(parts)


def _home_guide_build_file_infos(files: list[Any]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for item in files:
        if item is None:
            continue
        name = str(getattr(item, "name", "") or "")
        suffix = Path(name).suffix.lower()
        size = int(getattr(item, "size", 0) or 0)
        preview = ""
        if suffix == ".pdf":
            try:
                preview = _extract_pdf_preview_text(item.getvalue())
            except Exception:
                preview = ""
        infos.append(
            {
                "name": name,
                "size": size,
                "suffix": suffix,
                "text_preview": preview,
            }
        )
    return infos


def _guide_flow_label(flow: str) -> str:
    mapping = {
        "travel": "差旅",
        "material": "材料费",
        "policy": "制度咨询",
        "unknown": "未知",
    }
    return mapping.get(str(flow or "unknown"), "未知")


def _guide_status_badge_text(can_enter: bool) -> str:
    return "可进入正式流程" if can_enter else "等待分流判断"


def _flow_name_to_page(flow: str) -> str:
    if flow == "travel":
        return PAGE_TRAVEL_FLOW
    if flow == "material":
        return PAGE_MATERIAL_FLOW
    return PAGE_HOME_GUIDE


def _sync_legacy_flow_selector_from_page() -> None:
    page = str(st.session_state.get("current_page") or PAGE_HOME_GUIDE)
    if page == PAGE_TRAVEL_FLOW:
        st.session_state["flow_mode_selector"] = "差旅费流程"
    elif page == PAGE_MATERIAL_FLOW:
        st.session_state["flow_mode_selector"] = "材料费流程"


def ensure_router_state() -> None:
    page = str(st.session_state.get("current_page") or "")
    if page not in VALID_ROUTER_PAGES:
        st.session_state["current_page"] = PAGE_HOME_GUIDE
    previous_page = st.session_state.get("previous_page")
    if not isinstance(previous_page, str):
        st.session_state["previous_page"] = ""
    if not isinstance(st.session_state.get("active_flow_context"), dict):
        st.session_state["active_flow_context"] = {}
    if not isinstance(st.session_state.get("guide_auto_route_paused"), bool):
        st.session_state["guide_auto_route_paused"] = False
    if not isinstance(st.session_state.get("router_flash_message"), str):
        st.session_state["router_flash_message"] = ""
    _sync_legacy_flow_selector_from_page()


def set_current_page(page: str, *, pause_auto_route: bool = False, flash_message: str = "") -> None:
    target = str(page or PAGE_HOME_GUIDE)
    if target not in VALID_ROUTER_PAGES:
        target = PAGE_HOME_GUIDE
    current = str(st.session_state.get("current_page") or PAGE_HOME_GUIDE)
    if current != target:
        st.session_state["previous_page"] = current
    st.session_state["current_page"] = target
    if pause_auto_route:
        st.session_state["guide_auto_route_paused"] = True
    elif target != PAGE_HOME_GUIDE:
        st.session_state["guide_auto_route_paused"] = False
    if flash_message:
        st.session_state["router_flash_message"] = flash_message
    _sync_legacy_flow_selector_from_page()


def pop_router_flash_message() -> str:
    message = str(st.session_state.get("router_flash_message") or "").strip()
    if message:
        st.session_state["router_flash_message"] = ""
    return message


def _build_active_flow_context(flow: str, payload: dict[str, Any], files: list[Any]) -> dict[str, Any]:
    precheck = dict(payload.get("precheck_result") or {})
    identified = dict(payload.get("identified_doc_types") or {})
    missing_items = list(payload.get("missing_items") or [])
    guide_summary = dict(payload.get("guide_summary") or {})
    return {
        "flow_type": flow,
        "source": "home_guide",
        "files": list(files),
        "file_count": len(files),
        "guide_summary": guide_summary,
        "precheck_result": precheck,
        "missing_items": missing_items,
        "identified_doc_types": identified,
        "recommended_flow": str(payload.get("recommended_flow") or flow),
        "route_reason": str(payload.get("route_reason") or ""),
        "entered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _enter_recommended_flow(*, flow: str, payload: dict[str, Any], files: list[Any], auto: bool) -> None:
    flow_name = str(flow or "")
    if flow_name not in {"travel", "material"}:
        return
    st.session_state["guide_handoff_payload"] = dict(payload)
    st.session_state["guide_handoff_uploaded_files"] = list(files)
    st.session_state["guide_handoff_target_flow"] = flow_name
    st.session_state["guide_handoff_entered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["active_flow_context"] = _build_active_flow_context(flow_name, payload, files)
    flow_label = "差旅流程" if flow_name == "travel" else "材料费流程"
    jump_reason = "已自动进入" if auto else "已进入"
    set_current_page(
        _flow_name_to_page(flow_name),
        flash_message=f"{jump_reason}{flow_label}，并带入首页引导的材料与预检查结果。",
    )


def render_flow_back_to_home(flow: str) -> None:
    context = st.session_state.get("active_flow_context")
    context_dict = dict(context) if isinstance(context, dict) else {}
    is_from_home = str(context_dict.get("source") or "") == "home_guide" and str(context_dict.get("flow_type") or "") == flow
    top_left, top_right = st.columns([2, 5])
    if top_left.button("← 返回首页引导页", use_container_width=True, key=f"{flow}_back_home"):
        set_current_page(PAGE_HOME_GUIDE, pause_auto_route=True, flash_message="已返回首页引导页，自动跳转已临时暂停。")
        st.rerun()
    if is_from_home:
        file_count = int(context_dict.get("file_count") or 0)
        route_reason = str(context_dict.get("route_reason") or "")
        top_right.info(f"已从首页引导页带入 {file_count} 份材料。{route_reason}")


def get_guide_handoff_for_flow(flow: str) -> tuple[dict[str, Any], list[Any]]:
    target_flow = str(st.session_state.get("guide_handoff_target_flow") or "")
    if target_flow != str(flow or ""):
        return {}, []
    payload = st.session_state.get("guide_handoff_payload")
    files = st.session_state.get("guide_handoff_uploaded_files")
    return (dict(payload) if isinstance(payload, dict) else {}), (list(files) if isinstance(files, list) else [])


def render_home_guide_agent(upload_types: list[str]) -> None:
    st.subheader("报销助手（首页引导 Agent）")
    st.caption(
        "你可以先告诉我要报销什么、手头有哪些材料。"
        "我会先判断流程方向（差旅/材料费），并给你对应材料要求；准备好后再进入正式流程深度处理。"
    )

    auto_route_paused = bool(st.session_state.get("guide_auto_route_paused"))
    if auto_route_paused:
        c_resume_1, c_resume_2 = st.columns([5, 1])
        c_resume_1.info("当前已暂停自动跳转。你可以继续聊天或补材料，准备好后再恢复自动跳转。")
        if c_resume_2.button("恢复自动跳转", use_container_width=True, key="home_guide_resume_auto"):
            st.session_state["guide_auto_route_paused"] = False
            st.rerun()

    state_raw = st.session_state.get("home_guide_state")
    state = guide_usecase.normalize_guide_session(state_raw)
    st.session_state["home_guide_state"] = state

    runtime_files = st.session_state.setdefault("home_guide_uploaded_files", [])
    if not isinstance(runtime_files, list):
        runtime_files = []
        st.session_state["home_guide_uploaded_files"] = runtime_files

    uploaded = st.file_uploader(
        "上传材料用于首页预检查（PDF/图片，可多选）",
        type=upload_types,
        accept_multiple_files=True,
        key="home_guide_upload_files",
    )
    current_files = _merge_uploaded_lists(runtime_files, _as_uploaded_list(uploaded))
    st.session_state["home_guide_uploaded_files"] = current_files

    current_sig = _home_guide_file_signature(current_files)
    prev_sig = str(st.session_state.get("home_guide_upload_signature") or "")
    if current_sig != prev_sig:
        st.session_state["home_guide_upload_signature"] = current_sig
        file_infos = _home_guide_build_file_infos(current_files)
        state, _ = guide_usecase.process_guide_turn(
            state,
            user_message="",
            uploaded_files=file_infos,
            record_history=False,
        )
        st.session_state["home_guide_state"] = state

    with st.container(border=True):
        for msg in list(state.get("conversation_history") or [])[-12:]:
            role = str(msg.get("role") or "assistant")
            content = str(msg.get("content") or "")
            if not content:
                continue
            with st.chat_message(role):
                st.markdown(content)

    c1, c2 = st.columns([6, 1])
    c1.caption("直接聊天即可，例如：`我要报销差旅，需要准备什么材料？`")
    clear_clicked = c2.button("清空引导会话", use_container_width=True, key="home_guide_clear_session")

    if clear_clicked:
        st.session_state["home_guide_state"] = guide_usecase.new_guide_session()
        st.session_state["home_guide_uploaded_files"] = []
        st.session_state["home_guide_upload_signature"] = ""
        st.session_state["guide_auto_route_paused"] = False
        st.session_state["guide_last_auto_route_key"] = ""
        st.session_state["active_flow_context"] = {}
        st.session_state.pop("guide_handoff_payload", None)
        st.session_state.pop("guide_handoff_uploaded_files", None)
        st.session_state.pop("guide_handoff_target_flow", None)
        st.success("已清空首页引导会话。")
        st.rerun()

    user_message = st.chat_input(
        "告诉我你的报销目标或问题（例如：我要报销差旅，需要准备什么材料）",
        key="home_guide_chat_input",
    )
    if user_message is not None:
        user_message = str(user_message or "").strip()
        if not user_message and not current_files:
            st.info("你可以先说一句目标，或者先上传 1-2 份材料。")
        else:
            if not user_message:
                user_message = "我已上传这些材料，请先帮我做首页分流。"
            file_infos = _home_guide_build_file_infos(current_files)
            state, _ = guide_usecase.process_guide_turn(
                st.session_state.get("home_guide_state"),
                user_message=user_message,
                uploaded_files=file_infos,
            )
            st.session_state["home_guide_state"] = state
            st.rerun()

    state = guide_usecase.normalize_guide_session(st.session_state.get("home_guide_state"))
    st.session_state["home_guide_state"] = state

    st.markdown("### 引导工作台")
    g1, g2, g3, g4 = st.columns(4)
    recommended_flow = str(state.get("recommended_flow") or "unknown")
    can_enter_now = recommended_flow in {"travel", "material"}
    g1.metric("推荐流程", _guide_flow_label(recommended_flow))
    g2.metric("已上传", len(current_files))
    identified = dict(state.get("identified_doc_types") or {})
    g3.metric("识别类型数", sum(int(v or 0) for v in identified.values()))
    g4.metric("进入状态", _guide_status_badge_text(can_enter_now))

    st.caption(f"判断依据：{str(state.get('route_reason') or '待补充信息')}")

    with st.expander("查看已上传材料与预检查结果", expanded=False):
        precheck = dict(state.get("precheck_result") or {})
        classified_files = list(precheck.get("classified_files") or [])
        if classified_files:
            rows = []
            for item in classified_files:
                rows.append(
                    {
                        "文件名": item.get("name"),
                        "类型": item.get("doc_type"),
                        "依据": item.get("reason"),
                    }
                )
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("暂无可展示的预检查结果。")

        if identified:
            stats_rows = [{"材料类型": k, "数量": v} for k, v in sorted(identified.items(), key=lambda x: (-x[1], x[0]))]
            st.dataframe(stats_rows, hide_index=True, use_container_width=True)

    enter_disabled = recommended_flow not in {"travel", "material", "policy"}
    payload = dict(state.get("target_flow_payload") or {})
    auto_route_key = "|".join(
        [
            str(state.get("session_id") or ""),
            recommended_flow,
            current_sig,
            str(payload.get("guide_summary", {}).get("updated_at") or ""),
        ]
    )
    can_auto_route = (
        recommended_flow in {"travel", "material"}
        and bool(payload)
        and not bool(st.session_state.get("guide_auto_route_paused"))
    )
    if can_auto_route and st.session_state.get("guide_last_auto_route_key") != auto_route_key:
        st.session_state["guide_last_auto_route_key"] = auto_route_key
        _enter_recommended_flow(flow=recommended_flow, payload=payload, files=list(current_files), auto=True)
        st.rerun()

    enter_label = "进入推荐流程"
    enter_clicked = st.button(enter_label, use_container_width=True, disabled=enter_disabled, key="home_guide_enter_flow")

    if enter_clicked:
        if recommended_flow == "travel":
            _enter_recommended_flow(flow="travel", payload=payload, files=list(current_files), auto=False)
            st.rerun()
        if recommended_flow == "material":
            _enter_recommended_flow(flow="material", payload=payload, files=list(current_files), auto=False)
            st.rerun()
        if recommended_flow == "policy":
            st.info("当前推荐为制度咨询。你可以继续在首页对话区提问，或手动进入其他流程。")

    st.divider()
