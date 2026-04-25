from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import requests
import streamlit as st

from app.agents import AgentCommand
from app.services.ollama_config import chat_model as _chat_model
from app.ui import task_hub, workbench
from app.ui.agent_metrics import (
    record_llm_outcome as _record_llm_outcome,
    render_agent_metric_caption as _render_agent_metric_caption,
)
from app.ui.chat_widgets import (
    compose_three_stage_reply as _compose_three_stage_reply,
    render_chat_messages as _render_chat_messages,
)
from app.ui.pending_actions import (
    append_pending_action as _append_pending_action,
    clear_last_applied_action as _clear_last_applied_action,
    clear_pending_actions as _clear_pending_actions,
    get_pending_actions as _get_pending_actions,
    pending_actions_key as _pending_actions_key,
    record_last_applied_action as _record_last_applied_action,
    remove_pending_action as _remove_pending_action,
    update_pending_action as _update_pending_action,
)
from app.usecases import dto as usecase_dto
from app.usecases import material_agent as material_usecase


CHINESE_NUM_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_DEPENDENCIES: dict[str, Callable[..., Any]] = {}


def configure_material_workbench(**dependencies: Callable[..., Any]) -> None:
    _DEPENDENCIES.update({key: value for key, value in dependencies.items() if callable(value)})


def _require_dependency(name: str) -> Callable[..., Any]:
    dependency = _DEPENDENCIES.get(name)
    if not callable(dependency):
        raise RuntimeError(f"material_workbench dependency is not configured: {name}")
    return dependency


def _run_conversation_agent_task(objective: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str, list[Any]]:
    return _require_dependency("run_conversation_agent_task")(objective, payload)


def _execute_agent_command(command: AgentCommand, *, scope: str | None = None) -> tuple[bool, dict[str, Any], str]:
    return _require_dependency("execute_agent_command")(command, scope=scope)


def classify_user_message_intent(message: str, context: dict[str, Any] | None = None) -> Any:
    return _require_dependency("classify_user_message_intent")(message, context)


def _get_guide_handoff_for_flow(flow: str) -> tuple[dict[str, Any], list[Any]]:
    return _require_dependency("get_guide_handoff_for_flow")(flow)


def _render_export_download(task, key_scope: str = "default") -> None:
    _require_dependency("render_export_download")(task, key_scope)


def _render_included_file_list(label: str, files: list[Any], *, key_prefix: str = "file") -> None:
    _require_dependency("render_included_file_list")(label, files, key_prefix=key_prefix)


def _material_mark_export_confirmed(flag_key: str) -> None:
    st.session_state[str(flag_key or "")] = True


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("¥", "").replace("￥", "").strip())
    except (TypeError, ValueError):
        return None


def _format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _to_editor_rows(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows = value.get("rows") or value.get("items") or []
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _normalize_quantity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _normalize_line_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "item_name": str(row.get("item_name") or "").strip(),
                "spec": str(row.get("spec") or "").strip(),
                "quantity": _normalize_quantity(row.get("quantity")),
                "unit": str(row.get("unit") or "").strip(),
                "line_total_with_tax": _format_amount(_safe_float(row.get("line_total_with_tax"))),
            }
        )
    return normalized


def _line_items_total(rows: list[dict[str, str]]) -> float | None:
    amounts = [_safe_float(row.get("line_total_with_tax")) for row in rows or []]
    amounts = [value for value in amounts if value is not None]
    if not amounts:
        return None
    return round(sum(amounts), 2)


def _as_uploaded_list(uploaded_value: Any) -> list[Any]:
    if not uploaded_value:
        return []
    if isinstance(uploaded_value, (list, tuple)):
        return list(uploaded_value)
    return [uploaded_value]


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    return list(first or []) + [item for item in list(second or []) if item not in list(first or [])]


def _short_join_items(items: list[str], limit: int = 3, empty_text: str = "无") -> str:
    clean = [str(item or "").strip() for item in list(items or []) if str(item or "").strip()]
    if not clean:
        return empty_text
    shown = clean[:limit]
    suffix = f"等{len(clean)}项" if len(clean) > limit else ""
    return "、".join(shown) + suffix


MATERIAL_AGENT_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]


def _material_agent_extract_fields(task) -> dict[str, Any]:
    return material_usecase.extract_fields(task)


def _material_agent_build_fields_payload(fields: dict[str, Any]) -> dict[str, Any]:
    return material_usecase.build_fields_payload(fields)


def _material_agent_apply_updates(task_id: str, fields: dict[str, Any]) -> tuple[bool, str]:
    result: usecase_dto.OperationResult = material_usecase.apply_updates(task_id, fields)
    return result.ok, result.message


def _material_agent_quality_hints(fields: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    if not rows:
        hints.append("未识别到明细行，建议用“重新识别”或手工新增行。")
        return hints

    def _compact(text: str) -> str:
        return re.sub(r"[\s\-_/,，;；:：()（）\[\]【】]+", "", text or "")

    def _suspicious_overlap(name: str, spec: str) -> bool:
        if not name or not spec:
            return False
        name_c = _compact(name)
        spec_c = _compact(spec)
        if len(spec_c) < 3:
            return False
        # Avoid noisy warning for pure Chinese short descriptors.
        if re.fullmatch(r"[\u4e00-\u9fff]+", spec_c) and len(spec_c) <= 4:
            return False
        if spec_c not in name_c:
            return False
        # Only warn on clear duplication: appears as suffix or repeated in name.
        return name_c.endswith(spec_c) or name_c.count(spec_c) >= 2

    amount_value = _safe_float(fields.get("amount"))
    row_total = _line_items_total(rows)
    if (
        amount_value is not None
        and row_total is not None
        and abs(amount_value - row_total) > 10
        and abs(amount_value - row_total) > max(1.0, amount_value * 0.005)
    ):
        hints.append(f"发票总金额与明细合计不一致：{_format_amount(amount_value)} vs {_format_amount(row_total)}")

    review_items = fields.get("low_confidence_review")
    if isinstance(review_items, list) and review_items:
        hints.append(f"存在 {len(review_items)} 条低置信度修复建议，建议人工复核。")

    llm_stats = fields.get("llm_agent_stats")
    llm_ran = isinstance(llm_stats, dict) and int(llm_stats.get("chunks_total") or 0) > 0
    if llm_ran:
        failed = int(llm_stats.get("chunks_failed") or 0)
        if failed > 0:
            hints.append(f"LLM分块执行有 {failed} 个分块失败，建议重试“智能修复”。")
        suspicious = int(llm_stats.get("suspicious_rows") or 0)
        auto_fixed = int(llm_stats.get("auto_fixed_rows") or 0)
        review_rows = int(llm_stats.get("review_rows") or 0)
        if suspicious > 0 and review_rows <= 0 and auto_fixed > 0:
            hints.append(f"LLM已自动修复 {auto_fixed} 行可疑数据。")
        return hints

    long_stats = fields.get("long_mode_stats")
    if isinstance(long_stats, dict):
        candidate_rows = int(long_stats.get("candidate_rows") or 0)
        final_rows = int(long_stats.get("final_rows") or 0)
        if candidate_rows > 0 and final_rows > 0 and (candidate_rows - final_rows) >= 3:
            hints.append(f"长票候选行 {candidate_rows}，最终行 {final_rows}，可能仍有漏项。")

    for idx, row in enumerate(rows, start=1):
        name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        if _suspicious_overlap(name, spec):
            hints.append(f"第{idx}行项目名称可能混入规格：{name}")
            if len(hints) >= 4:
                break
    return hints


def _build_material_handoff_status_reply(
    *,
    task: Any,
    rows: list[dict[str, Any]],
    amount_value: float | None,
    row_total: float | None,
    quality_hints: list[str],
    guide_files: list[Any],
    guide_payload: dict[str, Any],
) -> str:
    missing_text = _short_join_items(
        list((guide_payload or {}).get("missing_items") or []),
        limit=3,
        empty_text="暂无明显缺件",
    )
    risk_text = _short_join_items(quality_hints, limit=2, empty_text="暂无明显质量风险")
    return _compose_three_stage_reply(
        f"已接收首页带入的 {len(_as_uploaded_list(guide_files))} 份材料，当前任务：{getattr(task, 'original_filename', '')}。",
        f"已抽取明细 {len(rows)} 行；发票总额 {_format_amount(amount_value)}，明细合计 {_format_amount(row_total)}。缺件提示：{missing_text}。风险提示：{risk_text}。",
        "你可以直接问“哪些行有问题”或给我修改指令（例如“第2行规格改为...”），我会马上处理。",
    )


def _material_agent_split_name_spec(name: str, spec: str) -> tuple[str, str]:
    return material_usecase.split_name_spec(name, spec)


def _material_agent_auto_split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    return material_usecase.auto_split_rows(rows)


def _material_agent_run_llm_fix(
    task,
    fields: dict[str, Any],
) -> tuple[bool, str, Any, dict[str, Any]]:
    return material_usecase.run_llm_fix(task, fields)


def _material_review_dialog_state_key(task_id: str) -> str:
    return f"material_review_dialog_open_{task_id}"


def _material_review_editor_key(task_id: str) -> str:
    return f"material_review_editor_{task_id}"


def _material_agent_build_review_compare_rows(fields: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return material_usecase.build_review_compare_rows(fields)


def _material_agent_apply_review_compare_edits(
    task_id: str,
    fields: dict[str, Any],
    edited_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    result: usecase_dto.OperationResult = material_usecase.apply_review_compare_edits(task_id, fields, edited_rows)
    return result.ok, result.message


@st.dialog("质量风险复核（左原始 / 右LLM建议）", width="large")
def _render_material_review_dialog(task_id: str) -> None:
    task = material_usecase.get_task(task_id)
    if task is None:
        st.error("任务不存在，无法复核。")
        return

    fields = _material_agent_extract_fields(task)
    left_rows, right_rows = _material_agent_build_review_compare_rows(fields)
    if not right_rows:
        st.info("当前没有可复核的低置信度行。")
        if st.button("关闭", key=f"material_review_close_empty_{task_id}", use_container_width=True):
            st.session_state[_material_review_dialog_state_key(task_id)] = False
            st.rerun()
        return

    st.caption("左侧是当前主表中的风险行，右侧是 LLM+案例RAG 建议。可直接改右侧后应用。")
    st.caption("也可关闭弹窗后在聊天区说：`第N行项目名称改为... 规格改为...`。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**风险原始表**")
        st.dataframe(left_rows, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**LLM建议表（可编辑）**")
        edited = st.data_editor(
            right_rows,
            use_container_width=True,
            hide_index=True,
            key=_material_review_editor_key(task_id),
            column_config={
                "row_no": st.column_config.NumberColumn("行号", disabled=True, width="small"),
                "item_name": st.column_config.TextColumn("项目名称(建议)"),
                "spec": st.column_config.TextColumn("规格型号(建议)"),
                "quantity": st.column_config.TextColumn("数量"),
                "unit": st.column_config.TextColumn("单位"),
                "line_total_with_tax": st.column_config.TextColumn("每项含税总价"),
                "confidence_text": st.column_config.TextColumn("置信度", disabled=True),
                "risk_types": st.column_config.TextColumn("风险类型", disabled=True),
                "reason": st.column_config.TextColumn("原因", disabled=True),
            },
        )
        edited_rows = _to_editor_rows(edited)

    b1, b2, b3 = st.columns(3)
    if b1.button("应用右侧到主表并标记已复核", key=f"material_review_apply_{task_id}", use_container_width=True):
        ok, msg = _material_agent_apply_review_compare_edits(task_id, fields, edited_rows)
        if ok:
            st.session_state[_material_review_dialog_state_key(task_id)] = False
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    if b2.button("仅标记已复核（不改动）", key=f"material_review_mark_only_{task_id}", use_container_width=True):
        new_fields = dict(fields)
        new_fields["low_confidence_review"] = []
        ok, err = _material_agent_apply_updates(task_id, new_fields)
        if ok:
            st.session_state[_material_review_dialog_state_key(task_id)] = False
            st.success("已清空复核队列。")
            st.rerun()
        else:
            st.error(f"保存失败：{err}")

    if b3.button("关闭弹窗", key=f"material_review_close_{task_id}", use_container_width=True):
        st.session_state[_material_review_dialog_state_key(task_id)] = False
        st.rerun()


def _material_rule_llm_compare_dialog_state_key(task_id: str) -> str:
    return f"material_rule_llm_compare_open_{task_id}"


def _material_rule_llm_compare_editor_key(task_id: str) -> str:
    return f"material_rule_llm_compare_editor_{task_id}"


def _material_agent_merge_editor_delta(
    task_id: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Streamlit data_editor may not always reflect an in-focus cell change in the returned table
    at the click moment. Merge session-state delta as a safety net (especially for clear-to-empty).
    """
    key = _material_rule_llm_compare_editor_key(task_id)
    state = st.session_state.get(key)
    if isinstance(state, list):
        full_rows = _to_editor_rows(state)
        if full_rows:
            rows = full_rows
    if not isinstance(state, dict):
        return rows

    full_rows = _to_editor_rows(state.get("data"))
    if full_rows:
        rows = full_rows

    edited_rows = state.get("edited_rows")
    if not isinstance(edited_rows, dict):
        return rows

    output = [dict(row) for row in rows]
    allowed_fields = {"item_name", "spec", "quantity", "unit", "line_total_with_tax"}
    ordered_fields = ["row_no", "item_name", "spec", "quantity", "unit", "line_total_with_tax"]
    visible_fields = list(output[0].keys()) if output else ordered_fields
    for row_idx_raw, change in edited_rows.items():
        try:
            row_idx = int(row_idx_raw)
        except (TypeError, ValueError):
            continue
        if row_idx < 0:
            continue
        while row_idx >= len(output):
            output.append(
                {
                    "row_no": len(output) + 1,
                    "item_name": "",
                    "spec": "",
                    "quantity": "",
                    "unit": "",
                    "line_total_with_tax": "",
                }
            )
        if not isinstance(change, dict):
            continue
        for field, value in change.items():
            normalized_field = field
            if normalized_field not in allowed_fields:
                candidates: list[str] = []
                try:
                    col_idx = int(str(field))
                except (TypeError, ValueError):
                    col_idx = -1
                if 0 <= col_idx < len(visible_fields):
                    candidates.append(str(visible_fields[col_idx]))
                if 0 <= (col_idx - 1) < len(visible_fields):
                    candidates.append(str(visible_fields[col_idx - 1]))
                if 0 <= col_idx < len(ordered_fields):
                    candidates.append(ordered_fields[col_idx])
                if 0 <= (col_idx - 1) < len(ordered_fields):
                    candidates.append(ordered_fields[col_idx - 1])
                for candidate in candidates:
                    if candidate in allowed_fields:
                        normalized_field = candidate
                        break
            if normalized_field not in allowed_fields:
                continue
            output[row_idx][normalized_field] = "" if value in (None, "") else str(value).strip()
    return output


def _material_agent_get_rule_llm_compare_rows(
    task_id: str,
    edited_value: Any,
    fallback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key = _material_rule_llm_compare_editor_key(task_id)
    state = st.session_state.get(key)

    rows = _to_editor_rows(edited_value)
    if not rows:
        rows = _to_editor_rows(state)
    if not rows and isinstance(state, dict):
        rows = _to_editor_rows(state.get("data"))
    if not rows:
        rows = _to_editor_rows(fallback_rows)
    return _material_agent_merge_editor_delta(task_id, rows)


def _material_agent_build_rule_llm_compare_rows(fields: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return material_usecase.build_rule_llm_compare_rows(fields)


def _material_agent_rule_llm_diff_count(fields: dict[str, Any]) -> int:
    return material_usecase.rule_llm_diff_count(fields)


def _material_agent_apply_rule_llm_compare_edits(
    task_id: str,
    fields: dict[str, Any],
    edited_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    result: usecase_dto.OperationResult = material_usecase.apply_rule_llm_compare_edits(task_id, fields, edited_rows)
    return result.ok, result.message


@st.dialog("智能修复对比（规则识别 vs LLM修复）", width="large")
def _render_material_rule_llm_compare_dialog(task_id: str) -> None:
    task = material_usecase.get_task(task_id)
    if task is None:
        st.error("任务不存在。")
        return

    fields = _material_agent_extract_fields(task)
    left_rows, right_rows = _material_agent_build_rule_llm_compare_rows(fields)
    initial_diff_count = _material_agent_rule_llm_diff_count(fields)
    if not left_rows and not right_rows:
        st.info("暂无可对比数据。请先执行一次 LLM 修复。")
        if st.button("关闭", key=f"material_rule_llm_close_empty_{task_id}", use_container_width=True):
            st.session_state[_material_rule_llm_compare_dialog_state_key(task_id)] = False
            st.rerun()
        return

    st.caption("左侧是规则识别结果（基线），右侧是 LLM+案例修复结果（可编辑）。")
    st.caption("提示：改完右侧后直接点“应用右侧结果到主表”（form提交可确保当前单元格修改被捕获）。")
    if initial_diff_count <= 0:
        st.info("当前规则表与LLM修复表一致，无需处理；可直接关闭弹窗。")
    with st.form(key=f"material_rule_llm_form_{task_id}", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**规则识别表（基线）**")
            st.dataframe(left_rows, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**LLM修复表（可编辑）**")
            edited = st.data_editor(
                right_rows,
                use_container_width=True,
                hide_index=True,
                key=_material_rule_llm_compare_editor_key(task_id),
                column_config={
                    "row_no": st.column_config.NumberColumn("行号", disabled=True, width="small"),
                    "item_name": st.column_config.TextColumn("项目名称"),
                    "spec": st.column_config.TextColumn("规格型号"),
                    "quantity": st.column_config.TextColumn("数量"),
                    "unit": st.column_config.TextColumn("单位"),
                    "line_total_with_tax": st.column_config.TextColumn("每项含税总价"),
                },
            )
        apply_submit = st.form_submit_button(
            "应用右侧结果到主表",
            use_container_width=True,
            disabled=initial_diff_count <= 0,
        )

    if apply_submit:
        edited_rows = _material_agent_get_rule_llm_compare_rows(task_id, edited, right_rows)
        ok, msg = _material_agent_apply_rule_llm_compare_edits(task_id, fields, edited_rows)
        if ok:
            if "无需应用" in msg:
                st.info(msg)
            else:
                st.session_state[_material_rule_llm_compare_dialog_state_key(task_id)] = False
                st.success(msg)
                st.rerun()
        else:
            st.error(msg)

    if st.button("关闭", key=f"material_rule_llm_close_{task_id}", use_container_width=True):
        st.session_state[_material_rule_llm_compare_dialog_state_key(task_id)] = False
        st.rerun()


def _material_agent_parse_chinese_number(token: str) -> int | None:
    text = str(token or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    if text == "十":
        return 10
    if text.startswith("十") and len(text) >= 2:
        tail = CHINESE_NUM_MAP.get(text[1:])
        if tail is not None:
            return 10 + tail
    if text.endswith("十") and len(text) >= 2:
        head = CHINESE_NUM_MAP.get(text[:-1])
        if head is not None:
            return head * 10
    if "十" in text:
        parts = text.split("十", 1)
        head = CHINESE_NUM_MAP.get(parts[0], 1 if parts[0] == "" else None)
        tail = CHINESE_NUM_MAP.get(parts[1], 0 if parts[1] == "" else None)
        if head is not None and tail is not None:
            return head * 10 + tail

    return CHINESE_NUM_MAP.get(text)


def _material_agent_resolve_row_index(text: str, row_count: int) -> int | None:
    if row_count <= 0:
        return None
    source = str(text or "")

    if any(token in source for token in ["最后一行", "末行", "最后1行"]):
        return row_count - 1
    if any(token in source for token in ["第一行", "首行"]):
        return 0

    match = re.search(r"倒数第\s*([0-9一二两三四五六七八九十]+)\s*行", source)
    if match:
        n = _material_agent_parse_chinese_number(match.group(1))
        if n is None:
            return None
        idx = row_count - n
        return idx if 0 <= idx < row_count else None

    match = re.search(r"第\s*([0-9一二两三四五六七八九十]+)\s*行", source)
    if match:
        n = _material_agent_parse_chinese_number(match.group(1))
        if n is None:
            return None
        idx = n - 1
        return idx if 0 <= idx < row_count else None

    return None


def _material_agent_extract_row_updates(text: str) -> dict[str, str]:
    field_map = {
        "项目名称": "item_name",
        "项目名": "item_name",
        "规格型号": "spec",
        "规格": "spec",
        "数量": "quantity",
        "单位": "unit",
        "每项含税总价": "line_total_with_tax",
        "含税总价": "line_total_with_tax",
        "金额": "line_total_with_tax",
    }
    updates: dict[str, str] = {}
    labels = "|".join(sorted((re.escape(k) for k in field_map.keys()), key=len, reverse=True))
    op = r"(?:应为|改为|设置为|设为|为|=|:|：)"
    pattern = re.compile(rf"({labels})\s*{op}\s*", re.IGNORECASE)
    matches = list(pattern.finditer(str(text or "")))
    if not matches:
        return updates

    for idx, match in enumerate(matches):
        alias = str(match.group(1) or "")
        target = field_map.get(alias)
        if not target:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = str(text[start:end] or "").strip().strip("。；;，,")
        if not value:
            continue
        if target == "line_total_with_tax":
            amount_value = _format_amount(_safe_float(value))
            if amount_value:
                updates[target] = amount_value
        else:
            updates[target] = value

    return updates


def _material_agent_looks_like_edit_intent(text: str) -> bool:
    source = str(text or "")
    action_hit = any(
        token in source
        for token in ["改为", "应为", "设置为", "设为", "删除", "新增一行", "添加一行", "重新识别"]
    )
    target_hit = any(
        token in source
        for token in [
            "第",
            "最后一行",
            "倒数第",
            "行",
            "项目名称",
            "项目名",
            "规格",
            "数量",
            "单位",
            "金额",
            "含税总价",
        ]
    )
    return action_hit and target_hit


def _material_agent_has_action_intent(text: str) -> bool:
    source = str(text or "").strip()
    if not source:
        return False

    strong_tokens = [
        "第", "最后一行", "倒数第", "删除", "新增一行", "添加一行",
        "改为", "应为", "设置为", "设为", "重新识别", "智能修复",
        "打开对比", "应用llm修复表", "应用llm结果", "应用对比结果",
        "撤销上一步", "查看最近变更", "变更记录",
    ]
    return any(token in source for token in strong_tokens)


def _material_agent_is_smalltalk(text: str) -> bool:
    source = str(text or "").strip().lower()
    if not source:
        return False
    smalltalk_tokens = {
        "你好", "您好", "在吗", "在不在", "hi", "hello", "hey",
        "好的", "ok", "谢谢", "好的谢谢",
    }
    if source in smalltalk_tokens:
        return True
    # Very short casual utterances should not trigger tool actions.
    return len(source) <= 4 and any(token in source for token in ["你好", "hi", "ok", "在吗"])


def _material_agent_undo_stack_key(task_id: str) -> str:
    return f"material_agent_undo_stack_{task_id}"


def _material_agent_change_log_key(task_id: str) -> str:
    return f"material_agent_change_log_{task_id}"


def _material_agent_snapshot_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "line_items": _normalize_line_items(_to_editor_rows(fields.get("line_items"))),
        "amount": str(fields.get("amount") or ""),
        "auto_split_enabled": bool(fields.get("auto_split_enabled", False)),
        "llm_line_items_suggested": _normalize_line_items(_to_editor_rows(fields.get("llm_line_items_suggested"))),
        "low_confidence_review": list(fields.get("low_confidence_review") or []),
    }


def _material_agent_push_undo_snapshot(task_id: str, fields: dict[str, Any]) -> None:
    key = _material_agent_undo_stack_key(task_id)
    stack = st.session_state.setdefault(key, [])
    if not isinstance(stack, list):
        stack = []
    stack.append(_material_agent_snapshot_fields(fields))
    if len(stack) > 20:
        stack = stack[-20:]
    st.session_state[key] = stack


def _material_agent_pop_undo_snapshot(task_id: str) -> dict[str, Any] | None:
    key = _material_agent_undo_stack_key(task_id)
    stack = st.session_state.get(key)
    if not isinstance(stack, list) or not stack:
        return None
    snapshot = stack.pop()
    st.session_state[key] = stack
    return snapshot if isinstance(snapshot, dict) else None


def _material_agent_record_change(task_id: str, action: str, summary_lines: list[str], user_text: str) -> None:
    key = _material_agent_change_log_key(task_id)
    logs = st.session_state.setdefault(key, [])
    if not isinstance(logs, list):
        logs = []
    logs.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "summary_lines": list(summary_lines or []),
            "user_text": str(user_text or "").strip(),
        }
    )
    if len(logs) > 30:
        logs = logs[-30:]
    st.session_state[key] = logs


def _material_agent_recent_changes_text(task_id: str, limit: int = 5) -> str:
    logs = st.session_state.get(_material_agent_change_log_key(task_id))
    if not isinstance(logs, list) or not logs:
        return "当前会话还没有可展示的变更记录。"
    parts: list[str] = []
    for item in logs[-max(1, limit):]:
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("time") or "")
        action = str(item.get("action") or "")
        lines = list(item.get("summary_lines") or [])
        if not lines:
            lines = ["已执行变更。"]
        parts.append(f"- [{stamp}] {action}：{'; '.join(str(x) for x in lines[:3])}")
    return "最近变更：\n" + "\n".join(parts)


def _material_scope_name(task_id: str) -> str:
    return f"material_agent_{task_id}"


def _material_pending_action_spec_from_text(user_text: str, task, fields: dict[str, Any]) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    if not text:
        return None

    if any(token in text for token in ["导出报销表", "导出结果", "导出excel", "导出 excel"]):
        return {
            "action_type": "material_export",
            "summary": "确认导出当前材料费结果",
            "target": "当前任务导出",
            "risk_level": "high",
            "payload": {"command": text},
        }

    if any(token in text for token in ["应用全部修正", "应用全部建议", "覆盖当前结果", "覆盖当前分配结果"]):
        return {
            "action_type": "material_apply_all",
            "summary": "批量应用当前材料费待确认建议",
            "target": "当前任务全部待确认动作",
            "risk_level": "high",
            "payload": {"command": text},
        }

    if any(token in text for token in ["智能修复", "重新识别", "应用llm修复表", "应用llm结果", "应用对比结果"]):
        return {
            "action_type": "material_command",
            "summary": "执行一条高影响材料费调整",
            "target": text[:120],
            "risk_level": "high",
            "payload": {"command": text},
        }

    return {
        "action_type": "material_command",
        "summary": "执行一条待确认材料费调整",
        "target": text[:120],
        "risk_level": "medium",
        "payload": {"command": text},
    }


def _material_build_pending_action_from_text(user_text: str, task, fields: dict[str, Any]) -> dict[str, Any] | None:
    spec = _material_pending_action_spec_from_text(user_text, task, fields)
    if not spec:
        return None
    return _append_pending_action(
        _material_scope_name(task.id),
        action_type=str(spec.get("action_type") or ""),
        summary=str(spec.get("summary") or ""),
        target=str(spec.get("target") or ""),
        risk_level=str(spec.get("risk_level") or "medium"),
        payload=dict(spec.get("payload") or {}),
    )


def _append_material_pending_action_from_spec(scope: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    action_type = str(spec.get("action_type") or "").strip()
    summary = str(spec.get("summary") or "").strip()
    if not action_type or not summary:
        return None
    return _append_pending_action(
        scope,
        action_type=action_type,
        summary=summary,
        target=str(spec.get("target") or ""),
        risk_level=str(spec.get("risk_level") or "medium"),
        payload=dict(spec.get("payload") or {}),
    )


def _material_execute_pending_action(action: dict[str, Any], task, fields: dict[str, Any]) -> tuple[bool, str, Any, dict[str, Any]]:
    action_type = str(action.get("action_type") or "").strip()
    if action_type == "material_export":
        st.session_state[f"material_export_confirmed_{task.id}"] = True
        return True, "已确认导出。你可以直接使用下方“下载Excel/下载文本”。", task, fields

    command = str((action.get("payload") or {}).get("command") or action.get("target") or "").strip()
    if action_type == "material_apply_all":
        command = "应用LLM修复表"
    if not command:
        return False, "待确认动作缺少可执行内容。", task, fields

    handled, reply, updated_task, updated_fields = _material_agent_apply_chat_command(command, task, fields)
    if not handled:
        return False, "确认后未命中可执行动作，请补充更具体的目标。", task, fields
    return True, reply, updated_task, updated_fields


def _build_material_execution_payload(task, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task,
        "fields": fields,
        "handler": _material_agent_apply_chat_command,
    }


def _execute_material_light_edit_command(
    *,
    user_text: str,
    task,
    fields: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    return _execute_agent_command(
        AgentCommand(
            command_type="material_light_edit",
            payload={
                "user_text": str(user_text or "").strip(),
                **_build_material_execution_payload(task, fields),
            },
            summary="执行材料费轻修正",
            risk_level="low",
            requires_confirmation=False,
            created_by="material_workbench",
        )
    )


def _execute_material_pending_action_command(
    *,
    action: dict[str, Any],
    task,
    fields: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    return _execute_agent_command(
        AgentCommand(
            command_type="material_pending_action",
            payload={
                "action": dict(action or {}),
                "task": task,
                "fields": fields,
                "handler": _material_execute_pending_action,
                "set_export_confirmed": _material_mark_export_confirmed,
                "export_flag_key": f"material_export_confirmed_{task.id}",
            },
            summary=str(action.get("summary") or "执行材料费待确认动作"),
            risk_level=str(action.get("risk_level") or "medium"),
            requires_confirmation=False,
            created_by="material_workbench",
        )
    )


def _material_agent_build_row_diff(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], max_lines: int = 8) -> list[str]:
    result: list[str] = []
    total = max(len(old_rows), len(new_rows))
    for idx in range(total):
        old = dict(old_rows[idx]) if idx < len(old_rows) else {}
        new = dict(new_rows[idx]) if idx < len(new_rows) else {}
        row_no = idx + 1
        if not old and new:
            result.append(f"第{row_no}行新增：{new.get('item_name', '')}")
        elif old and not new:
            result.append(f"第{row_no}行删除：{old.get('item_name', '')}")
        else:
            changed_fields: list[str] = []
            for field_name, label in [
                ("item_name", "项目名称"),
                ("spec", "规格型号"),
                ("quantity", "数量"),
                ("unit", "单位"),
                ("line_total_with_tax", "每项含税总价"),
            ]:
                if str(old.get(field_name) or "") != str(new.get(field_name) or ""):
                    changed_fields.append(label)
            if changed_fields:
                result.append(f"第{row_no}行更新：{'、'.join(changed_fields)}")
        if len(result) >= max_lines:
            break
    return result


def _material_agent_extract_json_object(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        payload = json.loads(source)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", source)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _material_agent_action_row_no(action: dict[str, Any], row_count: int) -> int | None:
    raw = action.get("row_no", action.get("row", action.get("index")))
    try:
        row_no = int(raw)
    except (TypeError, ValueError):
        row_no = None
    if row_no is None:
        row_ref = str(action.get("row_ref") or "").strip()
        if row_ref:
            idx = _material_agent_resolve_row_index(row_ref, row_count)
            if idx is not None:
                return idx + 1
        return None
    if row_no < 1:
        return None
    return row_no


def _material_agent_normalize_update_fields(raw_fields: Any) -> dict[str, str]:
    if not isinstance(raw_fields, dict):
        return {}

    alias_map = {
        "item_name": "item_name",
        "项目名称": "item_name",
        "项目名": "item_name",
        "name": "item_name",
        "spec": "spec",
        "规格": "spec",
        "规格型号": "spec",
        "quantity": "quantity",
        "数量": "quantity",
        "unit": "unit",
        "单位": "unit",
        "line_total_with_tax": "line_total_with_tax",
        "每项含税总价": "line_total_with_tax",
        "含税总价": "line_total_with_tax",
        "金额": "line_total_with_tax",
    }
    output: dict[str, str] = {}
    for key, value in raw_fields.items():
        target = alias_map.get(str(key).strip())
        if not target:
            continue
        text = "" if value is None else str(value).strip()
        if target == "line_total_with_tax":
            text = _format_amount(_safe_float(text))
        elif target == "quantity":
            text = _normalize_quantity(text)
        output[target] = text
    return output


def _material_agent_parse_actions_llm(
    user_text: str,
    task,
    fields: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_scope = "material"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()
    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    row_preview = []
    for idx, row in enumerate(rows[:8], start=1):
        row_preview.append(
            {
                "row_no": idx,
                "item_name": row.get("item_name"),
                "spec": row.get("spec"),
                "quantity": row.get("quantity"),
                "unit": row.get("unit"),
                "line_total_with_tax": row.get("line_total_with_tax"),
            }
        )

    system_prompt = (
        "你是材料费表格动作解析器。"
        "请把用户自然语言转换为可执行JSON，仅输出JSON对象，不要解释。"
        "JSON格式: {\"actions\":[...],\"reason\":\"...\"}。"
        "action只允许: update_row, batch_update, delete_row, add_row, reidentify, open_compare, apply_compare, undo_last, show_changes, none。"
        "字段只允许: item_name,spec,quantity,unit,line_total_with_tax。"
        "row_no从1开始。"
    )
    user_payload = {
        "task_id": getattr(task, "id", ""),
        "filename": getattr(task, "original_filename", ""),
        "row_count": len(rows),
        "row_preview": row_preview,
        "user_text": str(user_text or ""),
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    try:
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.0},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 60))
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
        data = _material_agent_extract_json_object(content)
        if isinstance(data, dict):
            actions = data.get("actions")
            if isinstance(actions, list):
                _record_llm_outcome(metric_scope, True)
                return [item for item in actions if isinstance(item, dict)]
    except Exception:
        _record_llm_outcome(metric_scope, False)
        return []
    _record_llm_outcome(metric_scope, False)
    return []


def _material_agent_apply_actions_from_llm(
    user_text: str,
    task,
    fields: dict[str, Any],
) -> tuple[bool, str, Any, dict[str, Any]]:
    actions = _material_agent_parse_actions_llm(user_text, task, fields)
    if not actions:
        return False, "", task, fields

    first_action = str((actions[0] or {}).get("action") or "").strip().lower()
    if first_action in {"", "none"}:
        return False, "", task, fields

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    if first_action == "show_changes":
        return True, _material_agent_recent_changes_text(task.id), task, fields

    if first_action == "undo_last":
        snapshot = _material_agent_pop_undo_snapshot(task.id)
        if not snapshot:
            return True, "没有可撤销的最近修改。", task, fields
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = _normalize_line_items(_to_editor_rows(snapshot.get("line_items")))
        new_fields["amount"] = str(snapshot.get("amount") or "")
        new_fields["llm_line_items_suggested"] = _normalize_line_items(_to_editor_rows(snapshot.get("llm_line_items_suggested")))
        new_fields["low_confidence_review"] = list(snapshot.get("low_confidence_review") or [])
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"撤销失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        _material_agent_record_change(task.id, "undo_last", ["已回滚到上一步。"], user_text)
        return True, "已撤销上一步修改。", updated_task, updated_fields

    if first_action == "open_compare":
        st.session_state[_material_rule_llm_compare_dialog_state_key(task.id)] = True
        return True, "已打开“智能修复对比（规则 vs LLM）”弹窗。", task, fields

    if first_action == "apply_compare":
        llm_rows = _normalize_line_items(_to_editor_rows(fields.get("llm_line_items_suggested")))
        if not llm_rows:
            return True, "当前没有可应用的LLM修复表，请先点“智能修复对比（规则 vs LLM）”生成。", task, fields
        _material_agent_push_undo_snapshot(task.id, fields)
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = llm_rows
        new_fields["amount"] = _format_amount(_line_items_total(llm_rows))
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"应用LLM修复表失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        diff_lines = _material_agent_build_row_diff(rows, llm_rows)
        _material_agent_record_change(task.id, "apply_compare", diff_lines or ["已应用LLM修复表。"], user_text)
        return True, "已将LLM修复表应用到主表。", updated_task, updated_fields

    if first_action == "reidentify":
        result = material_usecase.reprocess_and_export(task.id)
        if not result.ok:
            return True, f"重新识别失败：{result.message}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        line_count = len(_normalize_line_items(_to_editor_rows(updated_fields.get("line_items"))))
        _material_agent_record_change(task.id, "reidentify", [f"重新识别后共 {line_count} 行。"], user_text)
        return True, f"已重新识别，当前识别到明细 {line_count} 行。", updated_task, updated_fields

    # Row-level actions (update/delete/add/batch_update)
    working_rows = [dict(row) for row in rows]
    summaries: list[str] = []
    for action in actions:
        act = str(action.get("action") or "").strip().lower()
        if act not in {"update_row", "delete_row", "add_row", "batch_update"}:
            continue

        if act == "batch_update":
            updates = action.get("updates")
            if not isinstance(updates, list):
                continue
            for item in updates:
                if not isinstance(item, dict):
                    continue
                row_no = _material_agent_action_row_no(item, len(working_rows))
                if row_no is None or row_no < 1 or row_no > len(working_rows):
                    continue
                fields_patch = _material_agent_normalize_update_fields(item.get("fields") or {})
                if not fields_patch:
                    continue
                working_rows[row_no - 1].update(fields_patch)
                summaries.append(f"第{row_no}行更新：{','.join(fields_patch.keys())}")
            continue

        if act == "update_row":
            row_no = _material_agent_action_row_no(action, len(working_rows))
            if row_no is None or row_no < 1 or row_no > len(working_rows):
                continue
            fields_patch = _material_agent_normalize_update_fields(action.get("fields") or {})
            if not fields_patch:
                continue
            working_rows[row_no - 1].update(fields_patch)
            summaries.append(f"第{row_no}行更新：{','.join(fields_patch.keys())}")
            continue

        if act == "delete_row":
            row_no = _material_agent_action_row_no(action, len(working_rows))
            if row_no is None or row_no < 1 or row_no > len(working_rows):
                continue
            removed = dict(working_rows[row_no - 1])
            del working_rows[row_no - 1]
            summaries.append(f"删除第{row_no}行：{removed.get('item_name', '')}")
            continue

        if act == "add_row":
            fields_patch = _material_agent_normalize_update_fields(action.get("fields") or {})
            if not fields_patch.get("item_name"):
                continue
            payload = {
                "item_name": fields_patch.get("item_name", ""),
                "spec": fields_patch.get("spec", ""),
                "quantity": fields_patch.get("quantity", ""),
                "unit": fields_patch.get("unit", ""),
                "line_total_with_tax": fields_patch.get("line_total_with_tax", ""),
            }
            working_rows.append(payload)
            summaries.append(f"新增第{len(working_rows)}行：{payload.get('item_name', '')}")

    new_rows = _normalize_line_items(working_rows)
    if not new_rows:
        return True, "未识别到可执行的修改动作。你可以说：第3行数量改为20。", task, fields

    if new_rows == rows:
        return True, "没有检测到实际变更（可能行号或字段未命中）。", task, fields

    _material_agent_push_undo_snapshot(task.id, fields)
    new_fields = dict(fields)
    new_fields["auto_split_enabled"] = False
    new_fields["line_items"] = new_rows
    total = _line_items_total(new_rows)
    if total is not None:
        new_fields["amount"] = _format_amount(total)
    ok, err = _material_agent_apply_updates(task.id, new_fields)
    if not ok:
        return True, f"执行失败：{err}", task, fields

    updated_task = material_usecase.get_task(task.id) or task
    updated_fields = _material_agent_extract_fields(updated_task)
    diff_lines = _material_agent_build_row_diff(rows, new_rows)
    _material_agent_record_change(task.id, first_action, diff_lines or summaries or ["已应用修改。"], user_text)
    summary_text = "；".join((diff_lines or summaries)[:4]) if (diff_lines or summaries) else "已应用修改。"
    return True, f"已执行：{summary_text}", updated_task, updated_fields


def _material_agent_apply_chat_command(
    user_text: str,
    task,
    fields: dict[str, Any],
) -> tuple[bool, str, Any, dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return True, "我在。你可以先告诉我哪里看起来不对，我会先解释并给出可执行修改。", task, fields

    # Casual chat should not be interpreted as executable actions.
    if _material_agent_is_smalltalk(text):
        return True, "你好，我在。你可以直接说你的疑问或目标，例如“最后一行规格像混进项目名了”。", task, fields

    action_intent = _material_agent_has_action_intent(text)
    if action_intent:
        llm_handled, llm_reply, llm_task, llm_fields = _material_agent_apply_actions_from_llm(text, task, fields)
        if llm_handled:
            return llm_handled, llm_reply, llm_task, llm_fields

    if any(token in text for token in ["查看最近变更", "最近变更", "变更记录"]):
        return True, _material_agent_recent_changes_text(task.id), task, fields

    if any(token in text for token in ["撤销上一步", "撤销", "回滚"]):
        snapshot = _material_agent_pop_undo_snapshot(task.id)
        if not snapshot:
            return True, "没有可撤销的最近修改。", task, fields
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = _normalize_line_items(_to_editor_rows(snapshot.get("line_items")))
        new_fields["amount"] = str(snapshot.get("amount") or "")
        new_fields["llm_line_items_suggested"] = _normalize_line_items(_to_editor_rows(snapshot.get("llm_line_items_suggested")))
        new_fields["low_confidence_review"] = list(snapshot.get("low_confidence_review") or [])
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"撤销失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        _material_agent_record_change(task.id, "undo_last", ["已回滚到上一步。"], text)
        return True, "已撤销上一步修改。", updated_task, updated_fields

    if any(token in text for token in ["打开对比", "智能修复对比", "规则vsllm", "规则 vs llm"]):
        st.session_state[_material_rule_llm_compare_dialog_state_key(task.id)] = True
        return True, "已打开“智能修复对比（规则 vs LLM）”弹窗。", task, fields

    if any(token in text for token in ["应用llm修复表", "应用llm结果", "应用对比结果"]):
        llm_rows = _normalize_line_items(_to_editor_rows(fields.get("llm_line_items_suggested")))
        if not llm_rows:
            return True, "当前没有可应用的LLM修复表，请先点“智能修复对比（规则 vs LLM）”生成。", task, fields
        old_rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
        _material_agent_push_undo_snapshot(task.id, fields)
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = llm_rows
        new_fields["amount"] = _format_amount(_line_items_total(llm_rows))
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"应用LLM修复表失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        _material_agent_record_change(task.id, "apply_compare", _material_agent_build_row_diff(old_rows, llm_rows), text)
        return True, "已将LLM修复表应用到主表。", updated_task, updated_fields

    if "重新识别" in text:
        result = material_usecase.reprocess_and_export(task.id)
        if not result.ok:
            return True, f"重新识别失败：{result.message}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        line_count = len(_normalize_line_items(_to_editor_rows(updated_fields.get("line_items"))))
        return True, f"已重新识别，当前识别到明细 {line_count} 行。", updated_task, updated_fields

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    if not rows:
        rows = []

    delete_match = re.search(r"删除\s*(?:第\s*[0-9一二两三四五六七八九十]+\s*行|最后一行|末行|首行|第一行|倒数第\s*[0-9一二两三四五六七八九十]+\s*行)", text)
    if delete_match:
        idx = _material_agent_resolve_row_index(text, len(rows))
        if idx is None:
            return True, "未识别到要删除的行号，请用“第N行/最后一行/倒数第N行”。", task, fields
        if idx < 0 or idx >= len(rows):
            return True, "行号超出范围。", task, fields
        removed = rows.pop(idx)
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"删除失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        return True, f"已删除第{idx + 1}行：{removed.get('item_name', '')}。", updated_task, updated_fields

    if "新增一行" in text or "添加一行" in text:
        key_map = {
            "项目名称": "item_name",
            "项目名": "item_name",
            "规格": "spec",
            "规格型号": "spec",
            "数量": "quantity",
            "单位": "unit",
            "金额": "line_total_with_tax",
            "含税总价": "line_total_with_tax",
            "每项含税总价": "line_total_with_tax",
        }
        pairs = re.findall(r"(项目名称|项目名|规格|规格型号|数量|单位|金额|含税总价|每项含税总价)\s*[=:：]\s*([^,，;；]+)", text)
        payload = {"item_name": "", "spec": "", "quantity": "", "unit": "", "line_total_with_tax": ""}
        for key, raw_value in pairs:
            target = key_map.get(key)
            if not target:
                continue
            value = raw_value.strip()
            if target == "line_total_with_tax":
                value = _format_amount(_safe_float(value))
            payload[target] = value

        if not payload["item_name"]:
            return True, "新增一行至少要提供项目名称。示例：`新增一行 项目名称=电子元件*连接器, 规格=Y50EX, 数量=20, 单位=只, 金额=2396.04`", task, fields

        updated_rows = [dict(row) for row in rows]
        updated_rows.append(payload)
        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = updated_rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"新增失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        return True, "已新增1行并保存。", updated_task, updated_fields

    row_updates = _material_agent_extract_row_updates(text)
    if row_updates:
        idx = _material_agent_resolve_row_index(text, len(rows))
        if idx is None:
            return True, "未识别到行号，请用“第N行/最后一行/倒数第N行”。", task, fields
        if idx < 0 or idx >= len(rows):
            return True, "行号超出范围。", task, fields

        updated_rows = [dict(row) for row in rows]
        for field_name, value in row_updates.items():
            updated_rows[idx][field_name] = value

        new_fields = dict(fields)
        new_fields["auto_split_enabled"] = False
        new_fields["line_items"] = updated_rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"更新失败：{err}", task, fields
        updated_task = material_usecase.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        changed_fields = []
        if "item_name" in row_updates:
            changed_fields.append("项目名称")
        if "spec" in row_updates:
            changed_fields.append("规格型号")
        if "quantity" in row_updates:
            changed_fields.append("数量")
        if "unit" in row_updates:
            changed_fields.append("单位")
        if "line_total_with_tax" in row_updates:
            changed_fields.append("每项含税总价")
        fields_text = "、".join(changed_fields) if changed_fields else "字段"
        return True, f"已更新第{idx + 1}行：{fields_text}。", updated_task, updated_fields

    if any(token in text for token in ["智能修复", "agent修复", "llm修复", "可疑行修复", "案例修复"]):
        return _material_agent_run_llm_fix(task, fields)

    if _material_agent_looks_like_edit_intent(text):
        return (
            True,
            "我先不直接改动，避免误操作。你可以告诉我“哪一行、哪个字段、希望改成什么”，我会先给你确认再执行。",
            task,
            fields,
        )

    return False, "", task, fields


def _generate_material_agent_reply_llm(
    user_text: str,
    task,
    fields: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str | None:
    metric_scope = "material"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    hints = _material_agent_quality_hints(fields)
    raw_text = str(getattr(task, "raw_text", "") or "")

    rag_bundle = material_usecase.build_material_references(fields, raw_text)
    policy_refs = list(rag_bundle.get("policy_refs") or [])
    case_hits = list(rag_bundle.get("case_hits") or [])
    case_lines = []
    for hit in case_hits[:3]:
        category = str((hit.get("metadata") or {}).get("expense_category") or "")
        case_lines.append(f"{hit.get('score', 0):.2f} | {category} | {str(hit.get('content') or '')[:80]}")

    context = {
        "task_id": task.id,
        "filename": task.original_filename,
        "invoice_number": fields.get("invoice_number"),
        "invoice_date": fields.get("invoice_date"),
        "amount": fields.get("amount"),
        "seller": fields.get("seller"),
        "buyer": fields.get("buyer"),
        "processing_mode": fields.get("processing_mode"),
        "long_mode_stats": fields.get("long_mode_stats"),
        "line_count": len(rows),
        "line_items": rows[:120],
        "quality_hints": hints,
        "low_confidence_review": list(fields.get("low_confidence_review") or [])[:40],
        "llm_agent_stats": dict(fields.get("llm_agent_stats") or {}),
        "policy_refs": policy_refs[:4],
        "case_refs": case_lines,
    }
    context_text = json.dumps(context, ensure_ascii=False)

    system_prompt = (
        "你是材料费报销助手。"
        "你的职责是：检查漏项风险、项目名称与规格是否混杂、金额一致性，并给出下一步可执行建议。"
        "禁止编造未给出的发票字段。"
        "如果用户只是问候，先自然回应，再引导下一步。"
        "回答简短自然；在清单、核对、修复建议场景再使用要点。"
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for item in messages[-6:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        llm_messages.append({"role": role, "content": content[:1200]})

    llm_messages.append(
        {
            "role": "user",
            "content": f"当前材料上下文(JSON)：\n{context_text}\n\n用户问题：{user_text}",
        }
    )

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": llm_messages,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()
        if content:
            _record_llm_outcome(metric_scope, True)
            return content
    except Exception:
        pass

    prompt = "\n\n".join(f"[{item.get('role')}] {item.get('content')}" for item in llm_messages[-8:])
    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("response") or "").strip()
        if content:
            _record_llm_outcome(metric_scope, True)
            return content
    except Exception:
        _record_llm_outcome(metric_scope, False)
        return None

    _record_llm_outcome(metric_scope, False)
    return None


def _build_material_review_view(review_items: list[Any]) -> list[dict[str, Any]]:
    review_view = []
    for item in review_items:
        if not isinstance(item, dict):
            continue
        confidence_raw = item.get("confidence")
        try:
            confidence_text = f"{float(confidence_raw) * 100:.1f}%"
        except (TypeError, ValueError):
            confidence_text = str(confidence_raw or "")
        review_view.append(
            {
                "行号": item.get("row_no"),
                "项目名称": item.get("item_name"),
                "规格型号": item.get("spec"),
                "建议项目名称": item.get("suggested_item_name"),
                "建议规格型号": item.get("suggested_spec"),
                "置信度": confidence_text,
                "风险类型": "、".join(item.get("risk_types") or []),
                "原因": item.get("reason"),
            }
        )
    return review_view


def _render_material_pending_queue(
    *,
    task,
    fields: dict[str, Any],
    scope: str,
    pending_actions: list[dict[str, Any]],
    task_messages: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    pa1, pa2, pa3 = st.columns(3)
    apply_all_pending = pa1.button(
        "应用全部建议",
        key=f"material_pending_apply_all_{task.id}",
        use_container_width=True,
        disabled=not pending_actions,
    )
    undo_last = pa2.button("撤销上一步", key=f"material_pending_undo_{task.id}", use_container_width=True)
    clear_pending = pa3.button(
        "清空待确认",
        key=f"material_pending_clear_{task.id}",
        use_container_width=True,
        disabled=not pending_actions,
    )

    if clear_pending:
        _clear_pending_actions(scope)
        st.success("已清空待确认修改。")
        st.rerun()

    if undo_last:
        handled, reply, updated_task, updated_fields = _material_agent_apply_chat_command("撤销上一步", task, fields)
        if handled:
            _clear_last_applied_action(scope)
            task_messages.append(
                {
                    "role": "assistant",
                    "content": _compose_three_stage_reply(
                        "好的，我理解你要回退刚才的修改。",
                        reply,
                        "你可以继续指出要调整的行，或让我先解释当前风险点。",
                    ),
                }
            )
            task = updated_task or task
            fields = updated_fields or fields
            st.rerun()
        st.info("当前没有可撤销的上一步。")

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
                target_key = f"material_pending_target_{task.id}_{action_id}"
                current_target = str(action.get("target") or "")
                edited_target = c1.text_input("目标值（可改）", value=current_target, key=target_key, label_visibility="collapsed")
                if edited_target != current_target:
                    payload = dict(action.get("payload") or {})
                    if payload.get("command"):
                        payload["command"] = edited_target
                    _update_pending_action(scope, action_id, {"target": edited_target, "payload": payload})
                    action["target"] = edited_target
                    action["payload"] = payload

                confirm_clicked = c2.button("确认", key=f"material_pending_confirm_{task.id}_{action_id}", use_container_width=True)
                cancel_clicked = c3.button("取消", key=f"material_pending_cancel_{task.id}_{action_id}", use_container_width=True)
                if cancel_clicked:
                    _remove_pending_action(scope, action_id)
                    st.rerun()
                if confirm_clicked:
                    ok, result_payload, msg = _execute_material_pending_action_command(
                        action=action,
                        task=task,
                        fields=fields,
                    )
                    if not ok:
                        st.warning(msg)
                    else:
                        _record_last_applied_action(scope, action)
                        _remove_pending_action(scope, action_id)
                        task = result_payload.get("task") or task
                        fields = result_payload.get("fields") or fields
                        task_messages.append(
                            {
                                "role": "assistant",
                                "content": _compose_three_stage_reply(
                                    "好的，我已按你的确认执行。",
                                    msg,
                                    "如果你希望，我可以继续检查漏项和名称/规格混杂风险。",
                                ),
                            }
                        )
                        st.rerun()
    else:
        st.info("当前没有待确认动作。高风险操作会先放在这里，确认后再执行。")

    if apply_all_pending and pending_actions:
        success_count = 0
        failed_lines: list[str] = []
        for action in list(pending_actions):
            ok, result_payload, msg = _execute_material_pending_action_command(
                action=action,
                task=task,
                fields=fields,
            )
            if ok:
                success_count += 1
                _record_last_applied_action(scope, action)
                _remove_pending_action(scope, str(action.get("action_id") or ""))
                task = result_payload.get("task") or task
                fields = result_payload.get("fields") or fields
            else:
                failed_lines.append(msg)
        summary = f"已批量应用 {success_count} 条建议。"
        if failed_lines:
            summary += "\n" + "\n".join(f"- {line}" for line in failed_lines[:5])
        task_messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "我已经收到你的批量确认。",
                    summary,
                    "你可以继续微调具体行，或者直接导出当前结果。",
                ),
            }
        )
        st.rerun()

    return task, fields


def _render_material_chat_thread(
    *,
    task,
    fields: dict[str, Any],
    rows: list[dict[str, Any]],
    quality_hints: list[str],
    pending_actions: list[dict[str, Any]],
    scope: str,
    task_messages: list[dict[str, Any]],
) -> None:
    _render_agent_metric_caption("material")
    _render_chat_messages(task_messages, stream_state_key=f"material_chat_streamed_{task.id}")

    user_input = st.chat_input(
        "直接说你的问题或修改意图（例如：最后一行规格和项目名混了）",
        key=f"material_agent_chat_input_{task.id}",
    )
    if not user_input:
        return

    task_messages.append({"role": "user", "content": user_input})
    plan_ok, plan_payload, plan_summary, plan_commands = _run_conversation_agent_task(
        "plan_material_turn",
        {
            "user_text": user_input,
            "intent_parser": classify_user_message_intent,
            "intent_context": {
                "domain": "material",
                "line_count": len(rows),
                "pending_count": len(pending_actions),
                "quality_hint_count": len(quality_hints),
            },
            "pending_action_builder": _material_pending_action_spec_from_text,
            "reply_llm": _generate_material_agent_reply_llm,
            "task": task,
            "fields": fields,
            "messages": task_messages,
            "row_count": len(rows),
            "quality_hint_count": len(quality_hints),
            "pending_count": len(pending_actions),
            "execution_payload": _build_material_execution_payload(task, fields),
        },
    )
    planned_intent = dict(plan_payload.get("intent") or {})
    intent_type = str(planned_intent.get("intent_type") or "chat")
    needs_confirmation = bool(planned_intent.get("needs_confirmation"))

    if not plan_ok:
        task_messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "我看到了你的输入。",
                    plan_summary or "这轮对话规划没有成功，先按普通说明处理。",
                    "你可以重试一次，或者直接告诉我哪一行、哪个字段、希望改成什么。",
                ),
            }
        )
        st.rerun()

    if intent_type == "strong_action" and needs_confirmation:
        action = _append_material_pending_action_from_spec(
            scope,
            dict(plan_payload.get("pending_action_spec") or {}),
        )
        if action:
            task_messages.append(
                {
                    "role": "assistant",
                    "content": str(plan_payload.get("reply") or "").strip()
                    or _compose_three_stage_reply(
                        "我理解你的操作意图，这一步影响范围较大。",
                        f"我已把它放入右侧待确认区：{action.get('summary') or '待确认动作'}。",
                        "你可以在右侧逐条确认，或点“应用全部建议”。",
                    ),
                }
            )
            st.rerun()

    if intent_type == "light_edit":
        command = plan_commands[0] if plan_commands else AgentCommand(
            command_type="material_light_edit",
            payload={
                "user_text": user_input,
                **_build_material_execution_payload(task, fields),
            },
            summary="执行材料费轻修正",
            risk_level="low",
            requires_confirmation=False,
            created_by="material_workbench_fallback",
        )
        edit_ok, edit_payload, edit_summary = _execute_agent_command(command)
        if edit_ok and bool(edit_payload.get("handled")):
            _record_last_applied_action(
                scope,
                {
                    "action_id": uuid4().hex,
                    "action_type": "material_light_edit",
                    "summary": str(edit_summary or "轻量修正"),
                },
            )
        reply_ok, reply_payload, _, _ = _run_conversation_agent_task(
            "compose_material_edit_reply",
            {
                "execution_ok": edit_ok and bool(edit_payload.get("handled")),
                "execution_summary": edit_summary,
            },
        )
        task_messages.append(
            {
                "role": "assistant",
                "content": str(reply_payload.get("reply") or "").strip()
                if reply_ok
                else _compose_three_stage_reply(
                    "好的，我已经理解并执行了这条轻量修正。",
                    str(edit_summary or "已更新当前表格。"),
                    "如需恢复，我可以撤销刚才的修改；也可以继续帮你检查剩余风险。",
                ),
            }
        )
        st.rerun()

    task_messages.append(
        {
            "role": "assistant",
            "content": str(plan_payload.get("reply") or "").strip()
            or _compose_three_stage_reply(
                "收到，我先按对话方式给你解释。",
                "我先解释当前判断，不会直接改数据。你可以继续追问原因，或告诉我希望改成什么。",
                "如果你希望我执行修改，我会先判断风险：低风险可直接改，高风险先进入待确认区。",
            ),
        }
    )
    st.rerun()


def _render_material_conversation_agent() -> None:
    st.subheader("材料费工作台")
    st.caption("上传材料费发票后，Agent 自动抽取并生成明细表；历史任务可从左侧直接切换。")

    uploaded = st.file_uploader(
        "上传材料费发票（PDF，可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="material_agent_upload_files",
    )
    page_uploaded_files = _as_uploaded_list(uploaded)
    upload_list = list(page_uploaded_files)
    guide_payload, guide_files = _get_guide_handoff_for_flow("material")
    if guide_files:
        upload_list = _merge_uploaded_lists(upload_list, _as_uploaded_list(guide_files))
        st.info(f"已从首页引导带入 {len(guide_files)} 份材料，可直接点击“Agent识别材料发票”。")
        _render_included_file_list(
            flow_label="材料费流程",
            page_uploaded_files=page_uploaded_files,
            guide_files=guide_files,
            merged_files=upload_list,
        )
    if guide_payload:
        with st.expander("首页引导摘要（已带入）", expanded=False):
            st.json(guide_payload)

    task_ids = st.session_state.setdefault("material_agent_task_ids", [])
    if not isinstance(task_ids, list):
        task_ids = []
        st.session_state["material_agent_task_ids"] = task_ids

    action1, action2, action3 = st.columns(3)
    process_clicked = action1.button("Agent识别材料发票", use_container_width=True, key="material_agent_process")
    clear_tasks_clicked = action2.button("清空材料任务缓存", use_container_width=True, key="material_agent_clear_tasks")
    clear_chat_clicked = action3.button("清空材料会话", use_container_width=True, key="material_agent_clear_chat")

    if clear_tasks_clicked:
        for tid in list(st.session_state.get("material_agent_task_ids", []) or []):
            st.session_state.pop(_pending_actions_key(_material_scope_name(str(tid))), None)
            st.session_state.pop(_material_agent_undo_stack_key(str(tid)), None)
            _clear_last_applied_action(_material_scope_name(str(tid)))
        st.session_state.pop("material_agent_task_ids", None)
        st.session_state.pop("material_agent_chat_map", None)
        task_hub.set_selected_material_task("")
        st.success("已清空材料任务缓存。")
        st.rerun()
    if clear_chat_clicked:
        st.session_state.pop("material_agent_chat_map", None)
        st.success("已清空材料会话。")
        st.rerun()

    if process_clicked:
        if not upload_list:
            st.warning("请先上传 PDF。")
        else:
            with st.spinner("正在识别材料发票，并执行 LLM 自动修复..."):
                process_result: usecase_dto.MaterialBatchProcessResult = material_usecase.process_uploaded_files(upload_list)
                if process_result.task_ids:
                    merged = list(dict.fromkeys(process_result.task_ids + task_ids))
                    st.session_state["material_agent_task_ids"] = merged
                    task_hub.set_selected_material_task(process_result.task_ids[0])
            if process_result.prepare_errors:
                st.warning(
                    "以下文件的自动 LLM 修复未完成，可稍后点“智能修复对比（规则 vs LLM）”补跑：\n\n- "
                    + "\n- ".join(process_result.prepare_errors[:8])
                )
            st.success("材料费任务已更新。")
            st.rerun()

    valid_tasks = []
    for task_id in st.session_state.get("material_agent_task_ids", []):
        task = material_usecase.get_task(task_id)
        if task is not None:
            valid_tasks.append(task)
    st.session_state["material_agent_task_ids"] = [task.id for task in valid_tasks]

    if not valid_tasks:
        st.info("先上传材料费发票并点击“Agent识别材料发票”。")
        return

    selected_task_id = task_hub.get_selected_material_task_id()
    if selected_task_id and all(task.id != selected_task_id for task in valid_tasks):
        selected_task_id = ""
    if not selected_task_id:
        selected_task_id = valid_tasks[0].id
        task_hub.set_selected_material_task(selected_task_id)

    with st.expander("切换材料任务（兼容入口）", expanded=False):
        options = {f"{task.original_filename} | {task.id} | {task.status}": task.id for task in valid_tasks}
        current_label = next((label for label, value in options.items() if value == selected_task_id), list(options.keys())[0])
        selected_label = st.selectbox("选择当前材料任务", options=list(options.keys()), index=list(options.keys()).index(current_label), key="material_agent_selected_task")
        next_task_id = options[selected_label]
        if next_task_id != selected_task_id and st.button("切换到所选任务", use_container_width=True, key="material_agent_switch_task"):
            task_hub.set_selected_material_task(next_task_id)
            st.rerun()

    task = material_usecase.get_task(selected_task_id)
    if task is None:
        st.error("任务不存在。")
        return

    fields = _material_agent_extract_fields(task)
    scope = _material_scope_name(task.id)
    pending_actions = [item for item in _get_pending_actions(scope) if str(item.get("status") or "pending") == "pending"]
    compare_dialog_key = _material_rule_llm_compare_dialog_state_key(task.id)
    if bool(st.session_state.get(compare_dialog_key)):
        _render_material_rule_llm_compare_dialog(task.id)

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    row_total = _line_items_total(rows)
    amount_value = _safe_float(fields.get("amount"))
    display_rows = [{"row_no": idx + 1, **row} for idx, row in enumerate(rows)]
    display_rows_cn = [
        {
            "行号": idx + 1,
            "项目名称(含星号)": str(row.get("item_name") or ""),
            "规格型号": str(row.get("spec") or ""),
            "数量": str(row.get("quantity") or ""),
            "单位": str(row.get("unit") or ""),
            "每项含税总价": str(row.get("line_total_with_tax") or ""),
        }
        for idx, row in enumerate(rows)
    ]

    quality_hints = _material_agent_quality_hints(fields)
    stage_label = "待确认" if pending_actions else ("可导出" if task.status in {"completed", "corrected"} else "处理中")
    summary_text = (
        f"当前明细 {len(rows)} 行，质量提示 {len(quality_hints)} 条，待确认动作 {len(pending_actions)} 条。"
    )
    workbench.render_case_header(
        title=str(task.original_filename or "材料任务"),
        task_type_label="材料费",
        stage_label=stage_label,
        goal="整理当前材料费发票并复核风险",
        summary=summary_text,
        issue_text=f"当前发现 {len(quality_hints)} 条质量风险。" if quality_hints else "",
        next_step="继续在中间对话里指出具体行和字段，或在下方确认待处理建议。",
    )
    workbench.render_material_result_summary(
        amount_text=_format_amount(amount_value) if amount_value is not None else "-",
        row_count=len(rows),
        quality_hint_count=len(quality_hints),
        pending_count=len(pending_actions),
        processing_mode=str(fields.get("processing_mode") or fields.get("extraction_source") or "default"),
    )

    chat_map = st.session_state.setdefault("material_agent_chat_map", {})
    if not isinstance(chat_map, dict):
        chat_map = {}
        st.session_state["material_agent_chat_map"] = chat_map

    task_messages = chat_map.setdefault(
        task.id,
        [
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "我已经接管这张材料费发票。",
                    "我会持续显示当前状态、风险和待确认动作；默认先解释，不会盲目改数据。",
                    "你可以先问“这张有什么问题”，也可以说“最后一行规格和项目名混了，帮我拆开”。",
                ),
            }
        ],
    )

    # 如果已有规则表/LLM表差异，自动生成一条可确认建议，不直接执行。
    diff_count = _material_agent_rule_llm_diff_count(fields)
    if diff_count > 0:
        has_compare_action = any(
            str((item.get("payload") or {}).get("command") or "") == "应用LLM修复表" for item in pending_actions
        )
        if not has_compare_action:
            _append_pending_action(
                scope,
                action_type="material_command",
                summary=f"建议应用LLM修复结果（检测到 {diff_count} 处差异）",
                target="应用LLM修复表",
                risk_level="medium",
                payload={"command": "应用LLM修复表"},
            )
            pending_actions = [item for item in _get_pending_actions(scope) if str(item.get("status") or "pending") == "pending"]

    if guide_files or guide_payload:
        token_map = st.session_state.setdefault("material_agent_handoff_summary_token_map", {})
        if not isinstance(token_map, dict):
            token_map = {}
            st.session_state["material_agent_handoff_summary_token_map"] = token_map
        handoff_token = (
            f"material|{task.id}|{str(st.session_state.get('guide_handoff_entered_at') or '')}"
            f"|{len(_as_uploaded_list(guide_files))}"
        )
        if str(token_map.get(task.id) or "") != handoff_token:
            task_messages.append(
                {
                    "role": "assistant",
                    "content": _build_material_handoff_status_reply(
                        task=task,
                        rows=rows,
                        amount_value=amount_value,
                        row_total=row_total,
                        quality_hints=quality_hints,
                        guide_files=guide_files,
                        guide_payload=guide_payload,
                    ),
                }
            )
            token_map[task.id] = handoff_token
    review_items = list(fields.get("low_confidence_review") or [])
    review_view = _build_material_review_view(review_items)
    slot_like = {
        "项目名称为空": sum(1 for row in rows if not str(row.get("item_name") or "").strip()),
        "规格为空": sum(1 for row in rows if not str(row.get("spec") or "").strip()),
        "数量为空": sum(1 for row in rows if not str(row.get("quantity") or "").strip()),
        "金额为空": sum(1 for row in rows if not str(row.get("line_total_with_tax") or "").strip()),
    }

    center_col, right_col = st.columns([1.7, 1], gap="large")

    with center_col:
        st.markdown("### Agent 对话")
        _render_material_chat_thread(
            task=task,
            fields=fields,
            rows=rows,
            quality_hints=quality_hints,
            pending_actions=pending_actions,
            scope=scope,
            task_messages=task_messages,
        )

    with right_col:
        overview_tab, result_tab, files_tab, pending_tab = st.tabs(["概览", "结果", "文件", "待确认"])

        with overview_tab:
            if quality_hints:
                st.warning("发现质量风险：")
                for hint in quality_hints[:8]:
                    st.markdown(f"- {hint}")
            else:
                st.success("当前明细质量检查通过。")
            st.dataframe(
                [{"项": key, "数量": value} for key, value in slot_like.items()],
                use_container_width=True,
                hide_index=True,
            )
            if review_items:
                st.info(f"人工复核区：{len(review_items)} 条低置信度项。")
                open_key = _material_review_dialog_state_key(task.id)
                if open_key not in st.session_state:
                    st.session_state[open_key] = True
                if st.button(
                    "打开质量风险复核弹窗（双表对比）",
                    use_container_width=True,
                    key=f"material_review_open_{task.id}",
                ):
                    st.session_state[open_key] = True
                if bool(st.session_state.get(open_key)):
                    _render_material_review_dialog(task.id)
            else:
                st.session_state.pop(_material_review_dialog_state_key(task.id), None)

            if st.button("智能修复对比（规则 vs LLM）", use_container_width=True, key=f"material_agent_llm_compare_{task.id}"):
                latest_task = material_usecase.get_task(task.id) or task
                latest_fields = _material_agent_extract_fields(latest_task)
                baseline_rows = _normalize_line_items(_to_editor_rows(latest_fields.get("rule_line_items_baseline")))
                llm_rows = _normalize_line_items(_to_editor_rows(latest_fields.get("llm_line_items_suggested")))
                if not baseline_rows or not llm_rows:
                    with st.spinner("首次生成 LLM 修复结果并构建对比..."):
                        handled, reply, updated_task, updated_fields = _material_agent_run_llm_fix(latest_task, latest_fields)
                    if not handled:
                        st.error(reply)
                    else:
                        latest_task = updated_task or latest_task
                        latest_fields = updated_fields or latest_fields
                        st.success(reply)
                        baseline_rows = _normalize_line_items(_to_editor_rows(latest_fields.get("rule_line_items_baseline")))
                        llm_rows = _normalize_line_items(_to_editor_rows(latest_fields.get("llm_line_items_suggested")))
                if baseline_rows and llm_rows:
                    st.session_state[compare_dialog_key] = True
                    st.rerun()
                st.error("暂未生成可对比的规则表/LLM表。")

            _render_export_download(task, key_scope="material_agent")

        with result_tab:
            st.dataframe(display_rows_cn, use_container_width=True, hide_index=True)
            if review_view:
                st.markdown("**低置信度复核项**")
                st.dataframe(review_view, use_container_width=True, hide_index=True)

        with files_tab:
            with st.expander("查看抽取结果(JSON)", expanded=False):
                st.json(task.extracted_data or {})
            if guide_payload:
                with st.expander("查看首页立案摘要", expanded=False):
                    st.json(guide_payload)

        with pending_tab:
            task, fields = _render_material_pending_queue(
                task=task,
                fields=fields,
                scope=scope,
                pending_actions=pending_actions,
                task_messages=task_messages,
            )




def render_material_conversation_agent() -> None:
    _render_material_conversation_agent()
