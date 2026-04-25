from __future__ import annotations

import hashlib
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any

import requests
import streamlit as st
import json

from app.agents import AgentTask, ReimbursementAgentOrchestrator
from app.services import parser as ocr_parser
from app.ui import task_hub, workbench
from app.usecases import home_guide_agent as guide_usecase
from app.usecases import travel_agent as travel_usecase

PAGE_HOME_GUIDE = "home_guide"
PAGE_TRAVEL_FLOW = "travel_flow"
PAGE_MATERIAL_FLOW = "material_flow"
VALID_ROUTER_PAGES = {PAGE_HOME_GUIDE, PAGE_TRAVEL_FLOW, PAGE_MATERIAL_FLOW}
_HOME_AGENT_ORCHESTRATOR: ReimbursementAgentOrchestrator | None = None


def _get_agent_orchestrator() -> ReimbursementAgentOrchestrator:
    global _HOME_AGENT_ORCHESTRATOR
    if _HOME_AGENT_ORCHESTRATOR is None:
        _HOME_AGENT_ORCHESTRATOR = ReimbursementAgentOrchestrator()
    return _HOME_AGENT_ORCHESTRATOR


def _as_uploaded_list(uploaded_value) -> list[Any]:
    return travel_usecase.as_uploaded_list(uploaded_value)


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    return travel_usecase.merge_uploaded_lists(first, second)


def _extract_file_text_from_bytes(file_bytes: bytes, suffix: str) -> str:
    if not file_bytes:
        return ""
    return str(ocr_parser.parse_file_bytes(file_bytes, suffix) or "")


@st.cache_data(show_spinner=False)
def _extract_file_preview_text(file_bytes: bytes, suffix: str, max_chars: int = 1800) -> str:
    try:
        text = _extract_file_text_from_bytes(file_bytes, suffix)
    except Exception:
        return ""
    value = str(text or "").strip()
    if len(value) > max_chars:
        value = value[:max_chars]
    return value


def _home_guide_build_file_infos(files: list[Any]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for item in files:
        if item is None:
            continue
        name = str(getattr(item, "name", "") or "")
        suffix = Path(name).suffix.lower()
        size = int(getattr(item, "size", 0) or 0)
        preview = ""
        if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            try:
                preview = _extract_file_preview_text(item.getvalue(), suffix)
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
    st.session_state["guide_handoff_entered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["active_flow_context"] = _build_active_flow_context(flow_name, payload, files)
    flow_label = "差旅流程" if flow_name == "travel" else "材料费流程"
    jump_reason = "已自动进入" if auto else "已进入"
    if flow_name == "travel":
        title = f"差旅任务 {datetime.now().strftime('%m-%d %H:%M')}"
        task_hub.create_travel_task(
            title=title,
            goal=str(payload.get("route_reason") or ""),
            seed_files=list(files),
            guide_payload=dict(payload),
            source="home_guide",
        )
        st.session_state.pop("guide_handoff_payload", None)
        st.session_state.pop("guide_handoff_uploaded_files", None)
        st.session_state.pop("guide_handoff_target_flow", None)
    else:
        st.session_state["guide_handoff_payload"] = dict(payload)
        st.session_state["guide_handoff_uploaded_files"] = list(files)
        st.session_state["guide_handoff_target_flow"] = flow_name
        task_hub.set_selected_material_task("")
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


def _extract_home_composer_submission(raw_value: Any) -> tuple[str, list[Any]]:
    if raw_value is None:
        return "", []
    text_value = getattr(raw_value, "text", raw_value)
    file_value = getattr(raw_value, "files", [])
    return str(text_value or "").strip(), _as_uploaded_list(file_value)


def _chat_typewriter_enabled() -> bool:
    return str(os.getenv("CHAT_TYPEWRITER", "1")).strip().lower() in {"1", "true", "yes", "on"}


def _iter_typewriter_chunks(text: str):
    source = str(text or "")
    if not source:
        return
    length = len(source)
    if length >= 1000:
        chunk_size, pause = 18, 0.002
    elif length >= 600:
        chunk_size, pause = 12, 0.004
    elif length >= 260:
        chunk_size, pause = 6, 0.008
    else:
        chunk_size, pause = 2, 0.012
    for idx in range(0, length, chunk_size):
        yield source[idx : idx + chunk_size]
        if pause > 0:
            time.sleep(pause)


def _render_typewriter_markdown(content: str) -> None:
    holder = st.empty()
    rendered = ""
    for chunk in _iter_typewriter_chunks(content):
        rendered += chunk
        holder.markdown(rendered + "▌")
    holder.markdown(rendered)


@st.cache_data(show_spinner=False, ttl=60)
def _list_ollama_model_names(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{str(base_url or '').rstrip('/')}/api/tags", timeout=(4, 10))
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _home_chat_model() -> str:
    base_url = str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    preferred = (
        os.getenv("OLLAMA_CHAT_MODEL")
        or os.getenv("OLLAMA_TEXT_MODEL")
        or os.getenv("OLLAMA_VL_MODEL")
        or os.getenv("OLLAMA_MODEL")
        or "qwen2.5:7b-instruct"
    )
    available = _list_ollama_model_names(base_url)
    if not available:
        return str(preferred)
    if preferred in available:
        return str(preferred)
    for candidate in ["qwen2.5:7b-instruct", "qwen2.5vl:7b", "qwen2.5vl:3b"]:
        if candidate in available:
            return candidate
    return available[0]


def _generate_home_reply_llm(user_message: str, state: dict[str, Any], uploaded_files: list[Any]) -> str | None:
    text = str(user_message or "").strip()
    if not text:
        return None

    base_url = str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = _home_chat_model()
    flow = str(state.get("recommended_flow") or "unknown")
    reason = str(state.get("route_reason") or "")
    identified = dict(state.get("identified_doc_types") or {})
    history = list(state.get("conversation_history") or [])
    file_count = len(list(uploaded_files or []))

    context = {
        "recommended_flow": flow,
        "route_reason": reason,
        "file_count": file_count,
        "identified_doc_types": identified,
    }
    system_prompt = (
        "你是报销首页引导助手。"
        "回答要自然口语化，不要模板腔，不要官话。"
        "优先用2-4句短句回答，除非用户明确要清单。"
        "禁止编造材料状态，只能基于给定上下文。"
    )
    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in history[-4:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            llm_messages.append({"role": role, "content": content[:800]})
    llm_messages.append(
        {
            "role": "user",
            "content": f"当前首页状态(JSON): {json.dumps(context, ensure_ascii=False)}\n用户输入: {text}",
        }
    )

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": llm_messages,
            "options": {"temperature": 0.35},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 60))
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "").strip()
        if content:
            return content
    except Exception:
        pass

    prompt = "\n\n".join(f"[{msg.get('role')}] {msg.get('content')}" for msg in llm_messages[-6:])
    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0.35},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 60))
        resp.raise_for_status()
        content = str(resp.json().get("response") or "").strip()
        if content:
            return content
    except Exception:
        return None
    return None


def _render_home_chat_messages(history: list[dict[str, Any]], *, stream_state_key: str) -> None:
    latest_assistant_idx: int | None = None
    latest_assistant_text = ""
    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") == "assistant":
            latest_assistant_idx = idx
            latest_assistant_text = str(item.get("content") or "")
            break

    signature = ""
    if latest_assistant_idx is not None:
        digest = hashlib.sha1(latest_assistant_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        signature = f"{len(history)}:{latest_assistant_idx}:{digest}"

    last_streamed = str(st.session_state.get(stream_state_key) or "")
    should_stream = (
        _chat_typewriter_enabled()
        and latest_assistant_idx is not None
        and signature
        and signature != last_streamed
        and any(str((item or {}).get("role") or "") == "user" for item in history)
    )

    for idx, msg in enumerate(history):
        role = str(msg.get("role") or "assistant")
        content = str(msg.get("content") or "")
        if not content:
            continue
        with st.chat_message(role):
            if should_stream and idx == latest_assistant_idx:
                _render_typewriter_markdown(content)
                st.session_state[stream_state_key] = signature
            else:
                st.markdown(content)


def _fallback_home_payload(state: dict[str, Any], files: list[Any]) -> dict[str, Any]:
    payload = dict(state.get("target_flow_payload") or {})
    if payload:
        return payload
    return {
        "session_id": state.get("session_id"),
        "recommended_flow": state.get("recommended_flow"),
        "route_reason": state.get("route_reason"),
        "user_goal": state.get("user_goal"),
        "missing_items": list(state.get("missing_items") or []),
        "identified_doc_types": dict(state.get("identified_doc_types") or {}),
        "precheck_result": dict(state.get("precheck_result") or {}),
        "guide_summary": {
            "uploaded_count": len(list(files or [])),
            "ready": bool(state.get("is_ready_to_enter_flow")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def render_home_guide_agent(upload_types: list[str]) -> None:
    st.subheader("报销任务立案")
    st.caption("直接在底部输入框描述任务或拖入材料。我会先做首页预检查，再建议进入差旅或材料工作台。")

    state_raw = st.session_state.get("home_guide_state")
    state = guide_usecase.normalize_guide_session(state_raw)
    st.session_state["home_guide_state"] = state

    runtime_files = st.session_state.setdefault("home_guide_uploaded_files", [])
    if not isinstance(runtime_files, list):
        runtime_files = []
        st.session_state["home_guide_uploaded_files"] = runtime_files

    current_files = list(runtime_files)
    identified = dict(state.get("identified_doc_types") or {})
    identified_summary = "、".join(
        f"{doc_type} {count}份"
        for doc_type, count in sorted(identified.items(), key=lambda item: (-int(item[1] or 0), item[0]))
        if int(count or 0) > 0
    )
    recommended_flow = str(state.get("recommended_flow") or "unknown")
    file_names = [str(getattr(file, "name", "") or "") for file in current_files if getattr(file, "name", None)]
    payload = _fallback_home_payload(state, current_files)

    intake_left, intake_right = st.columns([1.6, 1], gap="large")
    with intake_left:
        with st.container(border=True):
            st.markdown("<div class='wb-card-title'>Agent 对话</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='wb-card-muted'>把问题和材料都交给底部输入框；支持一次附多个 PDF/图片。</div>",
                unsafe_allow_html=True,
            )
            _render_home_chat_messages(
                list(state.get("conversation_history") or [])[-12:],
                stream_state_key=f"home_guide_chat_streamed_{str(state.get('session_id') or '')}",
            )

    with intake_right:
        travel_clicked, material_clicked = workbench.render_recommendation_card(
            recommended_flow_label=_guide_flow_label(recommended_flow),
            route_reason=str(state.get("route_reason") or ""),
            file_count=len(current_files),
            identified_summary=identified_summary or "暂无明显材料类型",
            can_enter=recommended_flow in {"travel", "material"},
            file_names=file_names,
            show_entry_buttons=True,
            travel_button_label="进入差旅",
            material_button_label="进入材料",
            travel_button_key="home_guide_enter_travel",
            material_button_key="home_guide_enter_material",
        )
        with st.expander("查看预检查详情", expanded=False):
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
                st.markdown("<div class='wb-card-muted'>暂无可展示的预检查结果。</div>", unsafe_allow_html=True)

            if identified:
                stats_rows = [{"材料类型": k, "数量": v} for k, v in sorted(identified.items(), key=lambda x: (-x[1], x[0]))]
                st.dataframe(stats_rows, hide_index=True, use_container_width=True)

        clear_clicked = st.button("重置首页会话", use_container_width=True, key="home_guide_clear_session")

    if clear_clicked:
        st.session_state["home_guide_state"] = guide_usecase.new_guide_session()
        st.session_state["home_guide_uploaded_files"] = []
        st.session_state["guide_auto_route_paused"] = False
        st.session_state["guide_last_auto_route_key"] = ""
        st.session_state["active_flow_context"] = {}
        st.session_state.pop("guide_handoff_payload", None)
        st.session_state.pop("guide_handoff_uploaded_files", None)
        st.session_state.pop("guide_handoff_target_flow", None)
        st.success("已重置首页引导会话。")
        st.rerun()

    if travel_clicked:
        _enter_recommended_flow(flow="travel", payload=payload, files=list(current_files), auto=False)
        st.rerun()
    if material_clicked:
        _enter_recommended_flow(flow="material", payload=payload, files=list(current_files), auto=False)
        st.rerun()

    composer_value = st.chat_input(
        "直接描述报销目标，或把文件拖到这里（支持多文件）",
        key="home_guide_chat_input",
        accept_file="multiple",
        file_type=upload_types,
        max_upload_size=200,
    )
    if composer_value is None:
        return

    user_message, attached_files = _extract_home_composer_submission(composer_value)
    merged_files = _merge_uploaded_lists(current_files, attached_files)
    st.session_state["home_guide_uploaded_files"] = merged_files

    if not user_message and not merged_files:
        st.info("你可以直接提问，或者把 1-2 份材料拖进底部输入框。")
        return

    if not user_message:
        if attached_files:
            user_message = f"我上传了 {len(attached_files)} 份材料，请先帮我做首页分流。"
        else:
            user_message = "我已上传这些材料，请先帮我做首页分流。"

    file_infos = _home_guide_build_file_infos(merged_files)
    result = _get_agent_orchestrator().run_task(
        AgentTask(
            agent="conversation_agent",
            objective="run_home_turn",
            payload={
                "turn_processor": guide_usecase.process_guide_turn,
                "state": st.session_state.get("home_guide_state"),
                "user_message": user_message,
                "uploaded_files": file_infos,
                "reply_llm": _generate_home_reply_llm,
            },
        )
    )
    if result.ok:
        state = dict(result.payload.get("state") or {})
        enter_flow = str(result.payload.get("enter_flow") or "").strip()
        if enter_flow in {"travel", "material"}:
            payload = _fallback_home_payload(state, merged_files)
            _enter_recommended_flow(flow=enter_flow, payload=payload, files=list(merged_files), auto=False)
            st.session_state["home_guide_state"] = state
            st.rerun()
    else:
        state, _ = guide_usecase.process_guide_turn(
            st.session_state.get("home_guide_state"),
            user_message=user_message,
            uploaded_files=file_infos,
        )
    st.session_state["home_guide_state"] = state
    st.rerun()
