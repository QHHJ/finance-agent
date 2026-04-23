from __future__ import annotations

import html
from typing import Any

import streamlit as st


def inject_workbench_styles() -> None:
    st.markdown(
        """
<style>
.wb-shell-note {
  color: #64748b;
  font-size: 0.92rem;
}
.wb-title {
  font-size: 1.35rem;
  font-weight: 700;
  color: #0f172a;
}
.wb-subtitle {
  color: #475569;
  font-size: 0.95rem;
}
.wb-chip {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.82rem;
  margin-right: 0.35rem;
  margin-bottom: 0.25rem;
}
.wb-stat-label {
  color: #64748b;
  font-size: 0.82rem;
}
.wb-stat-value {
  color: #0f172a;
  font-size: 1.05rem;
  font-weight: 700;
}
.wb-card-title {
  color: #0f172a;
  font-weight: 700;
  margin-bottom: 0.35rem;
}
.wb-card-muted {
  color: #64748b;
  font-size: 0.9rem;
}
.wb-file-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.5rem;
}
.wb-file-pill {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  border: 1px solid #dbeafe;
  background: #f8fbff;
  color: #1e3a8a;
  font-size: 0.82rem;
  line-height: 1.2;
}
.wb-section-gap {
  margin-top: 0.9rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_case_header(
    *,
    title: str,
    task_type_label: str,
    stage_label: str,
    goal: str,
    summary: str,
    issue_text: str,
    next_step: str,
) -> None:
    with st.container(border=True):
        st.markdown(f"<div class='wb-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(
            " ".join(
                [
                    f"<span class='wb-chip'>任务类型：{task_type_label}</span>",
                    f"<span class='wb-chip'>当前阶段：{stage_label}</span>",
                ]
            ),
            unsafe_allow_html=True,
        )
        if goal:
            st.markdown(f"<div class='wb-subtitle'>当前目标：{goal}</div>", unsafe_allow_html=True)
        if summary:
            st.markdown(f"<div class='wb-shell-note'>当前摘要：{summary}</div>", unsafe_allow_html=True)
        if issue_text:
            st.warning(issue_text)
        if next_step:
            st.info(f"下一步建议：{next_step}")


def render_stat_strip(items: list[tuple[str, Any]]) -> None:
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            (
                f"<div class='wb-stat-label'>{label}</div>"
                f"<div class='wb-stat-value'>{value}</div>"
            ),
            unsafe_allow_html=True,
        )


def render_recommendation_card(
    *,
    recommended_flow_label: str,
    route_reason: str,
    file_count: int,
    identified_summary: str,
    can_enter: bool,
    file_names: list[str] | None = None,
    show_entry_buttons: bool = False,
    travel_button_label: str = "进入差旅",
    material_button_label: str = "进入材料",
    travel_button_key: str | None = None,
    material_button_key: str | None = None,
    travel_disabled: bool = False,
    material_disabled: bool = False,
) -> tuple[bool, bool]:
    travel_clicked = False
    material_clicked = False
    clean_files = [str(name or "").strip() for name in list(file_names or []) if str(name or "").strip()]

    with st.container(border=True):
        st.markdown("<div class='wb-card-title'>推荐结果</div>", unsafe_allow_html=True)
        st.markdown(f"**推荐流程**：{recommended_flow_label}")
        st.markdown(f"**判断依据**：{route_reason or '等待更多信息'}")
        st.markdown(f"**已上传材料**：{file_count} 份")
        st.markdown(f"**识别摘要**：{identified_summary or '暂无明显材料类型'}")
        if can_enter:
            st.success("当前可以直接进入正式工作台。")
        else:
            st.info("当前还在收集信息，可以继续补文件或提问。")

        st.markdown("<div class='wb-section-gap'></div>", unsafe_allow_html=True)
        render_uploaded_file_digest(clean_files, empty_text="当前还没有带入材料。")

        if show_entry_buttons:
            st.markdown(
                "<div class='wb-card-muted wb-section-gap'>保持在首页继续聊，或直接进入对应工作台。</div>",
                unsafe_allow_html=True,
            )
            action_cols = st.columns(2)
            travel_clicked = action_cols[0].button(
                travel_button_label,
                key=travel_button_key or "wb_enter_travel",
                use_container_width=True,
                disabled=travel_disabled,
            )
            material_clicked = action_cols[1].button(
                material_button_label,
                key=material_button_key or "wb_enter_material",
                use_container_width=True,
                disabled=material_disabled,
            )

    return travel_clicked, material_clicked


def render_trip_board(assignment: dict[str, Any]) -> None:
    sections = [
        (
            "去程",
            [
                ("票据", len(list(assignment.get("go_ticket") or []))),
                ("支付", len(list(assignment.get("go_payment") or []))),
                ("明细", len(list(assignment.get("go_detail") or []))),
            ],
            assignment.get("go_ticket_amount"),
            assignment.get("go_payment_amount"),
        ),
        (
            "返程",
            [
                ("票据", len(list(assignment.get("return_ticket") or []))),
                ("支付", len(list(assignment.get("return_payment") or []))),
                ("明细", len(list(assignment.get("return_detail") or []))),
            ],
            assignment.get("return_ticket_amount"),
            assignment.get("return_payment_amount"),
        ),
        (
            "酒店",
            [
                ("发票", len(list(assignment.get("hotel_invoice") or []))),
                ("支付", len(list(assignment.get("hotel_payment") or []))),
                ("订单", len(list(assignment.get("hotel_order") or []))),
            ],
            assignment.get("hotel_invoice_amount"),
            assignment.get("hotel_payment_amount"),
        ),
    ]
    cols = st.columns(3)
    for col, (title, items, left_amount, right_amount) in zip(cols, sections):
        with col.container(border=True):
            st.markdown(f"<div class='wb-card-title'>{title}</div>", unsafe_allow_html=True)
            for label, count in items:
                st.markdown(f"- {label}：{count}")
            if left_amount is not None or right_amount is not None:
                st.caption(f"金额参考：{left_amount or '-'} / {right_amount or '-'}")


def render_material_result_summary(
    *,
    amount_text: str,
    row_count: int,
    quality_hint_count: int,
    pending_count: int,
    processing_mode: str,
) -> None:
    render_stat_strip(
        [
            ("总金额", amount_text or "-"),
            ("明细行数", row_count),
            ("质量提示", quality_hint_count),
            ("待确认", pending_count),
            ("识别模式", processing_mode or "default"),
        ]
    )


def render_uploaded_file_digest(file_names: list[str], *, empty_text: str = "当前还没有上传材料。") -> None:
    st.markdown("<div class='wb-card-title'>已带入材料</div>", unsafe_allow_html=True)
    clean_files = [str(name or "").strip() for name in list(file_names or []) if str(name or "").strip()]
    if not clean_files:
        st.markdown(f"<div class='wb-card-muted'>{empty_text}</div>", unsafe_allow_html=True)
        return

    pills = "".join(
        f"<span class='wb-file-pill'>{html.escape(name)}</span>"
        for name in clean_files[:10]
    )
    st.markdown(f"<div class='wb-file-pills'>{pills}</div>", unsafe_allow_html=True)
    if len(clean_files) > 10:
        st.caption(f"还有 {len(clean_files) - 10} 份材料未展开。")
