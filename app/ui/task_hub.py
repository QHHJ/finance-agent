from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st


TRAVEL_TASK_REGISTRY_KEY = "workbench_travel_task_registry"
TRAVEL_TASK_ORDER_KEY = "workbench_travel_task_order"
TRAVEL_WORKSPACES_KEY = "workbench_travel_workspaces"
ACTIVE_TRAVEL_TASK_KEY = "workbench_active_travel_task_id"
SELECTED_MATERIAL_TASK_KEY = "workbench_selected_material_task_id"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def ensure_task_hub_state() -> None:
    if not isinstance(st.session_state.get(TRAVEL_TASK_REGISTRY_KEY), dict):
        st.session_state[TRAVEL_TASK_REGISTRY_KEY] = {}
    if not isinstance(st.session_state.get(TRAVEL_TASK_ORDER_KEY), list):
        st.session_state[TRAVEL_TASK_ORDER_KEY] = []
    if not isinstance(st.session_state.get(TRAVEL_WORKSPACES_KEY), dict):
        st.session_state[TRAVEL_WORKSPACES_KEY] = {}
    if not isinstance(st.session_state.get(ACTIVE_TRAVEL_TASK_KEY), str):
        st.session_state[ACTIVE_TRAVEL_TASK_KEY] = ""
    if not isinstance(st.session_state.get(SELECTED_MATERIAL_TASK_KEY), str):
        st.session_state[SELECTED_MATERIAL_TASK_KEY] = ""


def create_travel_task(
    *,
    title: str | None = None,
    goal: str = "",
    seed_files: list[Any] | None = None,
    guide_payload: dict[str, Any] | None = None,
    source: str = "manual",
) -> str:
    ensure_task_hub_state()
    task_id = f"travel_{uuid4().hex[:8]}"
    created_at = _now_text()
    clean_goal = str(goal or "").strip()
    clean_title = str(title or "").strip() or f"差旅任务 {created_at}"
    file_count = len(list(seed_files or []))
    registry = dict(st.session_state.get(TRAVEL_TASK_REGISTRY_KEY) or {})
    registry[task_id] = {
        "task_id": task_id,
        "task_type": "travel",
        "title": clean_title,
        "goal": clean_goal,
        "status": "草稿",
        "summary": f"已带入 {file_count} 份材料" if file_count else "等待上传材料",
        "file_count": file_count,
        "created_at": created_at,
        "updated_at": created_at,
        "source": str(source or "manual"),
    }
    st.session_state[TRAVEL_TASK_REGISTRY_KEY] = registry

    order = [task_id]
    for existing in list(st.session_state.get(TRAVEL_TASK_ORDER_KEY) or []):
        if existing != task_id:
            order.append(existing)
    st.session_state[TRAVEL_TASK_ORDER_KEY] = order

    workspaces = dict(st.session_state.get(TRAVEL_WORKSPACES_KEY) or {})
    workspaces[task_id] = {
        "files": list(seed_files or []),
        "messages": [],
        "assignment": {},
        "profiles": [],
        "manual_overrides": {},
        "manual_slot_overrides": {},
        "pool_signature": "",
        "guide_payload": dict(guide_payload or {}),
        "handoff_summary_token": "",
    }
    st.session_state[TRAVEL_WORKSPACES_KEY] = workspaces
    set_active_travel_task(task_id)
    return task_id


def set_active_travel_task(task_id: str) -> None:
    ensure_task_hub_state()
    value = str(task_id or "").strip()
    st.session_state[ACTIVE_TRAVEL_TASK_KEY] = value
    if value:
        order = [value]
        for existing in list(st.session_state.get(TRAVEL_TASK_ORDER_KEY) or []):
            if existing != value:
                order.append(existing)
        st.session_state[TRAVEL_TASK_ORDER_KEY] = order


def get_active_travel_task_id() -> str:
    ensure_task_hub_state()
    return str(st.session_state.get(ACTIVE_TRAVEL_TASK_KEY) or "")


def get_or_create_travel_workspace(task_id: str) -> dict[str, Any]:
    ensure_task_hub_state()
    key = str(task_id or "").strip()
    workspaces = dict(st.session_state.get(TRAVEL_WORKSPACES_KEY) or {})
    workspace = dict(workspaces.get(key) or {})
    changed = False
    defaults = {
        "files": [],
        "messages": [],
        "assignment": {},
        "profiles": [],
        "manual_overrides": {},
        "manual_slot_overrides": {},
        "pool_signature": "",
        "guide_payload": {},
        "handoff_summary_token": "",
    }
    for field, default_value in defaults.items():
        if field not in workspace:
            workspace[field] = default_value
            changed = True
    if changed or key not in workspaces:
        workspaces[key] = workspace
        st.session_state[TRAVEL_WORKSPACES_KEY] = workspaces
    return workspace


def save_travel_workspace(task_id: str, workspace: dict[str, Any]) -> None:
    ensure_task_hub_state()
    key = str(task_id or "").strip()
    workspaces = dict(st.session_state.get(TRAVEL_WORKSPACES_KEY) or {})
    workspaces[key] = dict(workspace or {})
    st.session_state[TRAVEL_WORKSPACES_KEY] = workspaces


def list_travel_tasks() -> list[dict[str, Any]]:
    ensure_task_hub_state()
    registry = dict(st.session_state.get(TRAVEL_TASK_REGISTRY_KEY) or {})
    order = list(st.session_state.get(TRAVEL_TASK_ORDER_KEY) or [])
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task_id in order:
        key = str(task_id or "").strip()
        if not key or key in seen:
            continue
        meta = registry.get(key)
        if isinstance(meta, dict):
            output.append(dict(meta))
            seen.add(key)
    for task_id, meta in registry.items():
        key = str(task_id or "").strip()
        if not key or key in seen or not isinstance(meta, dict):
            continue
        output.append(dict(meta))
    return output


def update_travel_task(
    task_id: str,
    *,
    title: str | None = None,
    goal: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    file_count: int | None = None,
) -> None:
    ensure_task_hub_state()
    key = str(task_id or "").strip()
    if not key:
        return
    registry = dict(st.session_state.get(TRAVEL_TASK_REGISTRY_KEY) or {})
    meta = dict(registry.get(key) or {})
    if not meta:
        meta = {
            "task_id": key,
            "task_type": "travel",
            "title": title or f"差旅任务 {_now_text()}",
            "goal": goal or "",
            "status": status or "草稿",
            "summary": summary or "等待上传材料",
            "file_count": int(file_count or 0),
            "created_at": _now_text(),
            "updated_at": _now_text(),
            "source": "manual",
        }
    if title is not None:
        meta["title"] = str(title or "").strip() or meta.get("title") or f"差旅任务 {_now_text()}"
    if goal is not None:
        meta["goal"] = str(goal or "").strip()
    if status is not None:
        meta["status"] = str(status or "").strip() or meta.get("status") or "草稿"
    if summary is not None:
        meta["summary"] = str(summary or "").strip()
    if file_count is not None:
        meta["file_count"] = int(file_count or 0)
    meta["updated_at"] = _now_text()
    registry[key] = meta
    st.session_state[TRAVEL_TASK_REGISTRY_KEY] = registry


def set_selected_material_task(task_id: str) -> None:
    ensure_task_hub_state()
    st.session_state[SELECTED_MATERIAL_TASK_KEY] = str(task_id or "").strip()


def get_selected_material_task_id() -> str:
    ensure_task_hub_state()
    return str(st.session_state.get(SELECTED_MATERIAL_TASK_KEY) or "")


def _material_task_summary(task: Any) -> tuple[str, str]:
    final_data = dict(getattr(task, "final_data", {}) or {})
    extracted_data = dict(getattr(task, "extracted_data", {}) or {})
    category = str(final_data.get("expense_category") or "").strip()
    line_items = list(extracted_data.get("line_items") or [])
    line_count = len([row for row in line_items if isinstance(row, dict)])
    status = str(getattr(task, "status", "") or "uploaded")
    summary = f"{line_count} 行明细"
    if category:
        summary = f"{category} · {summary}"
    return status, summary


def render_task_sidebar(
    *,
    current_page: str,
    material_tasks: list[Any],
) -> dict[str, Any] | None:
    ensure_task_hub_state()
    active_travel_task_id = get_active_travel_task_id()
    selected_material_task_id = get_selected_material_task_id()
    with st.sidebar:
        st.markdown("## 报销任务")
        quick_1, quick_2 = st.columns(2)
        new_travel = quick_1.button("+ 差旅", use_container_width=True, key="sidebar_new_travel_task")
        new_material = quick_2.button("+ 材料", use_container_width=True, key="sidebar_new_material_task")
        go_home = st.button("首页立案", use_container_width=True, key="sidebar_open_home")
        query = str(st.text_input("搜索任务", key="sidebar_task_search") or "").strip().lower()

        if new_travel:
            return {"action": "new_travel"}
        if new_material:
            return {"action": "new_material"}
        if go_home:
            return {"action": "open_home"}

        st.caption("差旅任务")
        travel_tasks = list_travel_tasks()
        if not travel_tasks:
            st.info("还没有差旅任务。")
        for meta in travel_tasks:
            task_id = str(meta.get("task_id") or "")
            title = str(meta.get("title") or "未命名差旅任务")
            summary = str(meta.get("summary") or "等待上传材料")
            status = str(meta.get("status") or "草稿")
            updated_at = str(meta.get("updated_at") or "")
            if query and query not in title.lower() and query not in summary.lower():
                continue
            with st.container(border=True):
                current_hint = "当前任务" if current_page == "travel_flow" and active_travel_task_id == task_id else "打开"
                st.markdown(f"**{title}**")
                st.caption(f"{status} · {updated_at}")
                st.write(summary)
                if st.button(current_hint, use_container_width=True, key=f"sidebar_open_travel_{task_id}"):
                    return {"action": "open_travel", "task_id": task_id}

        st.caption("材料任务")
        if not material_tasks:
            st.info("还没有材料任务。")
        for task in material_tasks:
            task_id = str(getattr(task, "id", "") or "")
            title = Path(str(getattr(task, "original_filename", "") or "材料任务")).name
            status, summary = _material_task_summary(task)
            updated_at = getattr(task, "updated_at", None)
            updated_text = updated_at.strftime("%Y-%m-%d %H:%M") if hasattr(updated_at, "strftime") else ""
            search_space = f"{title} {summary} {status}".lower()
            if query and query not in search_space:
                continue
            with st.container(border=True):
                current_hint = "当前任务" if current_page == "material_flow" and selected_material_task_id == task_id else "打开"
                st.markdown(f"**{title}**")
                st.caption(f"{status} · {updated_text}")
                st.write(summary)
                if st.button(current_hint, use_container_width=True, key=f"sidebar_open_material_{task_id}"):
                    return {"action": "open_material", "task_id": task_id}
    return None
