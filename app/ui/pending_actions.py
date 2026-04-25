from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st

from app.usecases import dto as usecase_dto


def pending_actions_key(scope: str) -> str:
    return f"{scope}_pending_actions"


def last_action_key(scope: str) -> str:
    return f"{scope}_last_applied_action"


def get_pending_actions(scope: str) -> list[dict[str, Any]]:
    key = pending_actions_key(scope)
    actions = st.session_state.setdefault(key, [])
    if not isinstance(actions, list):
        actions = []
        st.session_state[key] = actions
    normalized: list[dict[str, Any]] = []
    changed = False
    for item in actions:
        if not isinstance(item, dict):
            changed = True
            continue
        normalized.append(
            {
                "action_id": str(item.get("action_id") or uuid4().hex),
                "action_type": str(item.get("action_type") or ""),
                "summary": str(item.get("summary") or ""),
                "target": str(item.get("target") or ""),
                "risk_level": str(item.get("risk_level") or "medium"),
                "status": str(item.get("status") or "pending"),
                "payload": dict(item.get("payload") or {}),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    if changed or normalized != actions:
        st.session_state[key] = normalized
    return normalized


def set_pending_actions(scope: str, actions: list[dict[str, Any]]) -> None:
    st.session_state[pending_actions_key(scope)] = [dict(item) for item in actions if isinstance(item, dict)]


def append_pending_action(
    scope: str,
    *,
    action_type: str,
    summary: str,
    target: str = "",
    risk_level: str = "medium",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions = get_pending_actions(scope)
    action = usecase_dto.PendingAction(
        action_id=uuid4().hex,
        action_type=str(action_type or "").strip(),
        summary=str(summary or "").strip(),
        target=str(target or "").strip(),
        risk_level=str(risk_level or "medium"),
        status="pending",
        payload=dict(payload or {}),
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ).to_dict()
    actions.append(action)
    set_pending_actions(scope, actions)
    return action


def update_pending_action(scope: str, action_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    target = str(action_id or "").strip()
    if not target:
        return None
    actions = get_pending_actions(scope)
    updated = None
    for item in actions:
        if str(item.get("action_id") or "") != target:
            continue
        item.update(dict(patch or {}))
        updated = item
        break
    set_pending_actions(scope, actions)
    return updated


def remove_pending_action(scope: str, action_id: str) -> bool:
    target = str(action_id or "").strip()
    if not target:
        return False
    actions = get_pending_actions(scope)
    filtered = [item for item in actions if str(item.get("action_id") or "") != target]
    changed = len(filtered) != len(actions)
    if changed:
        set_pending_actions(scope, filtered)
    return changed


def clear_pending_actions(scope: str) -> None:
    set_pending_actions(scope, [])


def record_last_applied_action(scope: str, action: dict[str, Any]) -> None:
    entry = usecase_dto.LastAppliedAction(
        action_id=str(action.get("action_id") or uuid4().hex),
        action_type=str(action.get("action_type") or ""),
        summary=str(action.get("summary") or ""),
        scope=scope,
        applied_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ).to_dict()
    st.session_state[last_action_key(scope)] = entry


def get_last_applied_action(scope: str) -> dict[str, Any] | None:
    value = st.session_state.get(last_action_key(scope))
    return dict(value) if isinstance(value, dict) else None


def clear_last_applied_action(scope: str) -> None:
    st.session_state.pop(last_action_key(scope), None)
