from __future__ import annotations

import base64
import json
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any
from datetime import datetime
import zipfile

import requests
import streamlit as st
from pypdf import PdfReader

from app.runtime import init_runtime
from app.services import extractor, local_runner

LINE_ITEM_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
TRUE_VALUES = {"1", "true", "yes", "on"}


def _task_label(task) -> str:
    return f"{task.id} | {task.original_filename} | {task.status}"


def _policy_label(policy) -> str:
    return f"{policy.id} | {policy.name} | {str(policy.created_at).split('.')[0]}"


def _parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _get_default(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    return "" if value is None else str(value)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip()
        text = text.replace("−", "-").replace("—", "-")
        text = text.replace(",", "").replace("，", "").replace("¥", "").replace("￥", "")
        text = re.sub(r"[^\d.\-]", "", text)
        if text in {"", ".", "-", "-."}:
            return None
        return float(text)
    except ValueError:
        return None


def _format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _to_editor_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        # pandas.DataFrame
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _normalize_quantity(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def _normalize_line_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item_name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        quantity = _normalize_quantity(row.get("quantity"))
        unit = str(row.get("unit") or "").strip()
        line_total = _format_amount(_safe_float(row.get("line_total_with_tax")))

        if not any([item_name, spec, quantity, unit, line_total]):
            continue

        normalized.append(
            {
                "item_name": item_name,
                "spec": spec,
                "quantity": quantity,
                "unit": unit,
                "line_total_with_tax": line_total,
            }
        )
    return normalized


def _line_items_total(rows: list[dict[str, str]]) -> float | None:
    values = [_safe_float(row.get("line_total_with_tax")) for row in rows]
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers)


def _get_initial_line_items(extracted_fields: dict[str, Any]) -> list[dict[str, str]]:
    raw_items = extracted_fields.get("line_items")
    rows = _normalize_line_items(_to_editor_rows(raw_items))
    if rows:
        return rows

    item_content = str(extracted_fields.get("item_content") or "").strip()
    if item_content:
        return [
            {
                "item_name": item_content,
                "spec": "",
                "quantity": "",
                "unit": "",
                "line_total_with_tax": _get_default(extracted_fields, "amount"),
            }
        ]
    return []


@st.cache_data(show_spinner=False)
def _extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join(chunk for chunk in pages if chunk)


def _env_flag_true(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in TRUE_VALUES


def _extract_amount_from_filename(file_name: str) -> float | None:
    name = file_name or ""
    stem = Path(name).stem

    # 1) Strong signal: number followed by "元".
    match = re.search(r"(\d+(?:\.\d{1,2})?)\s*元", stem)
    if match:
        return _safe_float(match.group(1))

    # 2) Fallback: parse numeric tokens in filename, useful for names like
    # "酒店支付凭证8131.44.jpg" or "支付219.png".
    tokens = re.findall(r"\d+(?:\.\d{1,2})?", stem)
    if not tokens:
        return None

    candidates: list[float] = []
    for token in tokens:
        value = _safe_float(token)
        if value is None or value <= 0:
            continue
        # Exclude obvious date/time or ids (too long integers).
        if "." not in token and len(token) >= 6:
            continue
        if value > 200000:
            continue
        candidates.append(value)

    if not candidates:
        return None
    # Prefer the last numeric token in filename, usually closest to amount label.
    return candidates[-1]


def _extract_amount_from_text(raw_text: str) -> float | None:
    text = raw_text or ""
    amount_token = r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"

    def _is_valid_amount(amount: float, token: str) -> bool:
        if amount == 0:
            return False
        if abs(amount) > 200000:
            return False
        cleaned = token.replace(",", "").replace("，", "").replace("−", "-")
        if re.fullmatch(r"-?\d{4}", cleaned):
            year = int(cleaned)
            if 1900 <= abs(year) <= 2099:
                return False
        return True

    def _pick_best(candidates: list[float]) -> float | None:
        if not candidates:
            return None
        return max(candidates, key=lambda value: abs(value))

    payment_candidates: list[float] = []
    patterns = [
        rf"(?:支付金额|实付金额|付款金额|交易金额|订单金额|应付金额|合计支付|支付合计|实付|支付|payment amount|paid amount|amount paid|transaction amount)[^\d¥￥$-]{{0,20}}[¥￥$]?\s*({amount_token})",
        rf"(?:付款|支付|payment|paid)[^\d¥￥$-]{{0,14}}[¥￥$]?\s*({amount_token})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = match.group(1)
            amount = _safe_float(token)
            if amount is not None and _is_valid_amount(amount, token):
                payment_candidates.append(amount)

    best = _pick_best(payment_candidates)
    if best is not None:
        return best

    currency_candidates: list[float] = []
    for match in re.finditer(rf"[¥￥$]\s*({amount_token})", text, flags=re.IGNORECASE):
        token = match.group(1)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            currency_candidates.append(amount)
    for match in re.finditer(rf"({amount_token})\s*元", text, flags=re.IGNORECASE):
        token = match.group(1)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            currency_candidates.append(amount)

    best = _pick_best(currency_candidates)
    if best is not None:
        return best

    # Last fallback for model outputs without symbols/keywords, e.g. "payment_amount: -1850.00".
    generic_candidates: list[float] = []
    for match in re.finditer(amount_token, text):
        token = match.group(0)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            generic_candidates.append(amount)
    best = _pick_best(generic_candidates)
    if best is not None:
        return best
    return None


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_payment_amount(value: Any) -> float | None:
    amount = _safe_float(value)
    if amount is None:
        return None
    amount = abs(amount)
    return amount if amount > 0 else None


def _pick_payment_candidate(values: list[Any]) -> float | None:
    normalized: list[float] = []
    for value in values:
        amount = _normalize_payment_amount(value)
        if amount is None:
            continue
        if amount > 200000:
            continue
        normalized.append(amount)
    if not normalized:
        return None
    return max(normalized)


def _extract_payment_amount_from_model_output(content: str) -> float | None:
    parsed = _extract_json_from_text(content)
    if parsed:
        for key in ("payment_amount", "paid_amount", "amount", "transaction_amount"):
            if key in parsed:
                amount = _normalize_payment_amount(parsed.get(key))
                if amount is not None:
                    return amount
        for list_key in ("amount_candidates", "amounts", "candidates", "money_list"):
            value = parsed.get(list_key)
            if isinstance(value, list):
                picked = _pick_payment_candidate(value)
                if picked is not None:
                    return picked

    key_match = re.search(
        r"(?:payment_amount|paid_amount|paymentAmount|transaction_amount)\s*[:=]\s*[\"']?([-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)",
        content or "",
        flags=re.IGNORECASE,
    )
    if key_match:
        amount = _normalize_payment_amount(key_match.group(1))
        if amount is not None:
            return amount

    amount = _extract_amount_from_text(content)
    return _normalize_payment_amount(amount)


@st.cache_data(show_spinner=False)
def _extract_payment_amount_with_ollama(file_bytes: bytes) -> float | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    encoded = base64.b64encode(file_bytes).decode("utf-8")

    prompts = [
        (
            "你是财务助手。请从支付记录截图中提取“本次支付金额（人民币）”。"
            "只输出 JSON：{\"payment_amount\": \"123.45\"}。"
            "若无法识别，输出 {\"payment_amount\": null}。"
        ),
        (
            "请识别截图里所有看起来像金额的数字（可带负号、逗号和小数），"
            "只输出 JSON：{\"amount_candidates\": [\"-8131.44\", \"20.00\"]}。"
        ),
    ]

    for prompt in prompts:
        try:
            payload_chat = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=(8, 45))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

        try:
            payload_generate = {
                "model": model,
                "stream": False,
                "prompt": prompt,
                "images": [encoded],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

    return None


@st.cache_data(show_spinner=False)
def _extract_payment_amount_with_ollama_text(raw_text: str) -> float | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None
    text = (raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    prompts = [
        (
            "你是财务助手。请从以下支付记录文本中提取“本次支付金额（人民币）”。"
            "只输出 JSON：{\"payment_amount\": \"123.45\"}。"
            "若无法识别，输出 {\"payment_amount\": null}。\n\n"
            f"支付记录文本：\n{text[:12000]}"
        ),
        (
            "请从下面文本里找出所有看起来像金额的数字（可带负号、逗号和小数），"
            "只输出 JSON：{\"amount_candidates\": [\"-8131.44\", \"20.00\"]}。\n\n"
            f"支付记录文本：\n{text[:12000]}"
        ),
    ]

    for prompt in prompts:
        try:
            payload_generate = {
                "model": model,
                "stream": False,
                "prompt": prompt,
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

    return None


def _auto_extract_amount_from_ticket(uploaded_file) -> float | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
            extracted = extractor.extract_invoice_fields(raw_text)
            amount = _safe_float(extracted.get("amount"))
            if amount is not None:
                return amount
            return _extract_amount_from_text(raw_text)
        except Exception:
            return _extract_amount_from_filename(uploaded_file.name)

    # 对图片票据优先用视觉模型识别。
    amount = _extract_payment_amount_with_ollama(file_bytes)
    if amount is not None:
        return amount
    return _extract_amount_from_filename(uploaded_file.name)


def _auto_extract_amount_from_payment(uploaded_file) -> float | None:
    if uploaded_file is None:
        return None

    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
            amount = _extract_payment_amount_with_ollama_text(raw_text)
            if amount is not None:
                return amount
        except Exception:
            return None
    else:
        amount = _extract_payment_amount_with_ollama(file_bytes)
        if amount is not None:
            return amount

    return None


def _as_uploaded_list(uploaded_value) -> list[Any]:
    if uploaded_value is None:
        return []
    if isinstance(uploaded_value, list):
        return [item for item in uploaded_value if item is not None]
    return [uploaded_value]


def _files_signature(files: list[Any]) -> str:
    parts: list[str] = []
    for file in files:
        name = str(getattr(file, "name", ""))
        size = str(getattr(file, "size", ""))
        parts.append(f"{name}:{size}")
    return "|".join(parts)


def _aggregate_auto_amount(files: list[Any], extractor_func) -> tuple[float | None, int, int]:
    total = 0.0
    recognized = 0
    for file in files:
        amount = extractor_func(file)
        if amount is None:
            continue
        recognized += 1
        total += amount

    total_files = len(files)
    if recognized == 0:
        return None, 0, total_files
    return total, recognized, total_files


def _sync_amount_state(prefix: str, role: str, uploaded_files: list[Any], auto_amount: float | None) -> str:
    file_state_key = f"{prefix}_{role}_file_signature"
    amount_state_key = f"{prefix}_{role}_amount"
    current_signature = _files_signature(uploaded_files)

    if st.session_state.get(file_state_key) != current_signature:
        st.session_state[file_state_key] = current_signature
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    elif amount_state_key not in st.session_state:
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    elif _safe_float(st.session_state.get(amount_state_key)) is None and auto_amount is not None:
        # Preserve manual corrections, but auto-fill when previous value is empty.
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    return amount_state_key


def _render_amount_match_check(prefix: str, ticket_label: str, ticket_files, payment_files) -> dict[str, Any]:
    ticket_file_list = _as_uploaded_list(ticket_files)
    payment_file_list = _as_uploaded_list(payment_files)

    ticket_auto_amount, ticket_recognized, ticket_total = _aggregate_auto_amount(
        ticket_file_list, _auto_extract_amount_from_ticket
    )
    payment_auto_amount, payment_recognized, payment_total = _aggregate_auto_amount(
        payment_file_list, _auto_extract_amount_from_payment
    )

    ticket_amount_key = _sync_amount_state(prefix, "ticket", ticket_file_list, ticket_auto_amount)
    payment_amount_key = _sync_amount_state(prefix, "payment", payment_file_list, payment_auto_amount)

    left, right = st.columns(2)
    left.text_input(f"{ticket_label}金额（合计）", key=ticket_amount_key)
    right.text_input("支付记录金额（合计）", key=payment_amount_key)

    if ticket_auto_amount is not None:
        left.caption(
            f"已自动识别 {ticket_recognized}/{ticket_total} 份，合计：{_format_amount(ticket_auto_amount)}"
        )
    elif ticket_total > 0:
        left.caption(f"已上传 {ticket_total} 份，自动识别 0 份。")
    if payment_auto_amount is not None:
        right.caption(
            f"已自动识别 {payment_recognized}/{payment_total} 份，合计：{_format_amount(payment_auto_amount)}"
        )
    elif payment_total > 0:
        right.caption(f"已上传 {payment_total} 份，自动识别 0 份。")

    ticket_amount = _safe_float(st.session_state.get(ticket_amount_key))
    payment_amount = _safe_float(st.session_state.get(payment_amount_key))

    checked = ticket_amount is not None and payment_amount is not None
    matched = checked and abs(ticket_amount - payment_amount) <= 0.01

    if checked and matched:
        st.success(f"{ticket_label}与支付记录金额一致。")
    elif checked:
        st.error(
            f"{ticket_label}与支付记录金额不一致："
            f"{_format_amount(ticket_amount)} vs {_format_amount(payment_amount)}"
        )
    elif ticket_total > 0 and payment_total > 0:
        st.warning("已上传票据和支付记录，但尚未形成可比对金额，请补充或修正金额。")

    return {
        "ticket_amount": ticket_amount,
        "payment_amount": payment_amount,
        "ticket_files_total": ticket_total,
        "ticket_files_recognized": ticket_recognized,
        "payment_files_total": payment_total,
        "payment_files_recognized": payment_recognized,
        "checked": checked,
        "matched": matched,
    }


def _render_travel_transport_section(section_title: str, prefix: str) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown(f"#### {section_title}")
        transport_type = st.radio(
            "交通方式",
            options=["飞机", "高铁"],
            horizontal=True,
            key=f"{prefix}_transport_type",
        )
        ticket_label = "机票发票" if transport_type == "飞机" else "高铁报销凭证"

        ticket_files = st.file_uploader(
            f"上传{ticket_label}（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_ticket_file",
        )
        payment_files = st.file_uploader(
            "上传支付记录（微信/支付宝/银行，PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_payment_file",
        )
        ticket_detail_files = st.file_uploader(
            "上传机票明细（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_ticket_detail_file",
        )

        amount_check = _render_amount_match_check(prefix, ticket_label, ticket_files, payment_files)

        missing: list[str] = []
        if not ticket_files:
            missing.append(ticket_label)
        if not payment_files:
            missing.append("支付记录")
        if not ticket_detail_files:
            missing.append("机票明细")

        if missing:
            st.info(f"待补充：{'、'.join(missing)}")

        return {
            "section": section_title,
            "transport_type": transport_type,
            "ticket_label": ticket_label,
            "ticket_uploaded": bool(ticket_files),
            "payment_uploaded": bool(payment_files),
            "ticket_detail_uploaded": bool(ticket_detail_files),
            "ticket_file_count": len(ticket_files or []),
            "payment_file_count": len(payment_files or []),
            "ticket_detail_file_count": len(ticket_detail_files or []),
            "complete": not missing,
            **amount_check,
        }


def _render_travel_hotel_section(prefix: str) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown("#### 3) 酒店报销")
        no_cost_stay = st.checkbox(
            "在出差地住宿不产生费用（如亲友家住宿）",
            value=False,
            key=f"{prefix}_no_cost_stay",
        )

        if no_cost_stay:
            explanation = st.text_area(
                "情况说明（必填）",
                height=100,
                key=f"{prefix}_no_cost_explanation",
                placeholder="请说明住宿地点、日期范围及未产生费用原因。",
            )
            complete = bool(explanation.strip())
            if complete:
                st.success("已提供无费用住宿情况说明。")
            else:
                st.warning("请填写无费用住宿情况说明。")
            return {
                "section": "酒店",
                "no_cost_stay": True,
                "complete": complete,
                "checked": True,
                "matched": True,
                "ticket_amount": None,
                "payment_amount": None,
            }

        invoice_files = st.file_uploader(
            "上传酒店发票（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_invoice",
        )
        payment_files = st.file_uploader(
            "上传酒店支付记录（微信/支付宝/银行，PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_payment",
        )
        order_files = st.file_uploader(
            "上传酒店平台订单截图（如携程）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_order",
        )

        amount_check = _render_amount_match_check(prefix, "酒店发票", invoice_files, payment_files)

        missing: list[str] = []
        if not invoice_files:
            missing.append("酒店发票")
        if not payment_files:
            missing.append("支付记录")
        if not order_files:
            missing.append("酒店订单截图")

        if missing:
            st.info(f"待补充：{'、'.join(missing)}")

        return {
            "section": "酒店",
            "no_cost_stay": False,
            "invoice_uploaded": bool(invoice_files),
            "payment_uploaded": bool(payment_files),
            "order_uploaded": bool(order_files),
            "invoice_file_count": len(invoice_files or []),
            "payment_file_count": len(payment_files or []),
            "order_file_count": len(order_files or []),
            "complete": not missing,
            **amount_check,
        }


def _render_travel_summary(
    go_section: dict[str, Any],
    return_section: dict[str, Any],
    hotel_section: dict[str, Any],
) -> None:
    st.subheader("差旅流程校验结果")

    issues: list[str] = []
    tips: list[str] = []

    if not go_section.get("complete"):
        issues.append("去程交通材料不完整。")
    if not return_section.get("complete"):
        issues.append("返程交通材料不完整。")
    if not hotel_section.get("complete"):
        issues.append("酒店材料不完整。")

    if go_section.get("complete") and return_section.get("complete"):
        st.success("交通闭环已满足：去程与返程材料均已上传。")
    else:
        st.error("交通闭环未满足：需同时提供去程与返程交通材料。")

    for section_name, section_data in [("去程交通", go_section), ("返程交通", return_section), ("酒店", hotel_section)]:
        if section_data.get("checked") and not section_data.get("matched"):
            issues.append(f"{section_name}票据金额与支付记录金额不一致。")
        elif section_data.get("complete") and not section_data.get("checked"):
            tips.append(f"{section_name}已上传材料，但金额尚未形成有效匹配。")

    if issues:
        st.error("存在高优先级问题，请修正后再提交报销。")
        for issue in issues:
            st.markdown(f"- {issue}")
    else:
        st.success("当前材料满足基础提交要求。")

    if tips:
        st.warning("提示：")
        for tip in tips:
            st.markdown(f"- {tip}")

    summary = {
        "go_transport": go_section,
        "return_transport": return_section,
        "hotel": hotel_section,
        "all_required_uploaded": not issues,
    }
    with st.expander("查看差旅流程结构化结果(JSON)", expanded=False):
        st.json(summary)


def _sanitize_export_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "差旅报销材料"


def _safe_uploaded_filename(name: str, default_stem: str) -> str:
    raw = Path(name or "").name
    if not raw:
        raw = default_stem
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", raw).strip()
    return cleaned or default_stem


def _amount_suffix(amount: float | None) -> str:
    if amount is None:
        return "金额未知"
    if abs(amount - round(amount)) <= 0.01:
        return f"{int(round(amount))}元"
    return f"{amount:.2f}元"


def _zip_ensure_dir(zip_file: zipfile.ZipFile, dir_path: str) -> None:
    normalized = dir_path.replace("\\", "/").rstrip("/") + "/"
    zip_file.writestr(normalized, b"")


def _zip_write_uploaded_files(zip_file: zipfile.ZipFile, target_dir: str, files: list[Any]) -> None:
    _zip_ensure_dir(zip_file, target_dir)
    for idx, uploaded in enumerate(files, start=1):
        original_name = str(getattr(uploaded, "name", ""))
        safe_name = _safe_uploaded_filename(original_name, f"file_{idx}")
        stored_name = f"{idx:02d}_{safe_name}"
        zip_file.writestr(f"{target_dir}/{stored_name}", uploaded.getvalue())


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
    root_name = _sanitize_export_name(package_name)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        go_root = f"{root_name}/出差去程交通报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程机票发票_{_amount_suffix(go_ticket_amount)}",
            go_ticket_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程支付记录_{_amount_suffix(go_payment_amount)}",
            go_payment_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程机票明细",
            go_detail_files,
        )

        return_root = f"{root_name}/出差返程交通报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程机票发票_{_amount_suffix(return_ticket_amount)}",
            return_ticket_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程支付记录_{_amount_suffix(return_payment_amount)}",
            return_payment_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程机票明细",
            return_detail_files,
        )

        hotel_root = f"{root_name}/酒店报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{hotel_root}/酒店发票_{_amount_suffix(hotel_invoice_amount)}",
            hotel_invoice_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{hotel_root}/支付记录_{_amount_suffix(hotel_payment_amount)}",
            hotel_payment_files,
        )
        _zip_write_uploaded_files(zip_file, f"{hotel_root}/订单截图", hotel_order_files)
    buffer.seek(0)
    return buffer.getvalue()


def _render_travel_package_export() -> None:
    st.subheader("差旅材料打包导出")
    default_name = f"差旅报销材料_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_name = st.text_input(
        "压缩包名称（无需 .zip）",
        value=default_name,
        key="travel_export_package_name",
    )

    go_ticket_files = _as_uploaded_list(st.session_state.get("travel_go_ticket_file"))
    go_payment_files = _as_uploaded_list(st.session_state.get("travel_go_payment_file"))
    go_detail_files = _as_uploaded_list(st.session_state.get("travel_go_ticket_detail_file"))

    return_ticket_files = _as_uploaded_list(st.session_state.get("travel_return_ticket_file"))
    return_payment_files = _as_uploaded_list(st.session_state.get("travel_return_payment_file"))
    return_detail_files = _as_uploaded_list(st.session_state.get("travel_return_ticket_detail_file"))

    hotel_invoice_files = _as_uploaded_list(st.session_state.get("travel_hotel_hotel_invoice"))
    hotel_payment_files = _as_uploaded_list(st.session_state.get("travel_hotel_hotel_payment"))
    hotel_order_files = _as_uploaded_list(st.session_state.get("travel_hotel_hotel_order"))

    go_ticket_amount = _safe_float(st.session_state.get("travel_go_ticket_amount"))
    go_payment_amount = _safe_float(st.session_state.get("travel_go_payment_amount"))
    return_ticket_amount = _safe_float(st.session_state.get("travel_return_ticket_amount"))
    return_payment_amount = _safe_float(st.session_state.get("travel_return_payment_amount"))
    hotel_invoice_amount = _safe_float(st.session_state.get("travel_hotel_ticket_amount"))
    hotel_payment_amount = _safe_float(st.session_state.get("travel_hotel_payment_amount"))

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


def _render_policy_section() -> None:
    st.subheader("1) 制度管理")
    policy_files = st.file_uploader(
        "上传制度 PDF（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="policy_files",
    )
    if st.button("保存制度文件", use_container_width=True):
        if not policy_files:
            st.warning("请先选择制度文件。")
            return
        for file in policy_files:
            local_runner.upload_policy_pdf(file.name, file.getvalue())
        st.success(f"已入库制度文件：{len(policy_files)} 份")
        st.rerun()

    policies = local_runner.list_policies(limit=200)
    if not policies:
        st.info("当前没有制度文件。")
        return

    selected_policy_label = st.selectbox(
        "已上传制度（可删除）",
        options=[_policy_label(policy) for policy in policies],
        key="policy_delete_select",
    )
    confirm_delete = st.checkbox("确认删除所选制度", value=False, key="policy_delete_confirm")
    if st.button("删除所选制度", use_container_width=True, type="secondary"):
        if not confirm_delete:
            st.warning("请先勾选确认删除。")
            return
        policy_id = int(selected_policy_label.split(" | ")[0])
        ok = local_runner.delete_policy(policy_id)
        if ok:
            st.success("制度文件已删除（数据库记录和本地 PDF 均已删除）。")
            st.rerun()
        else:
            st.error("删除失败：制度不存在。")


def _render_invoice_upload_section() -> None:
    st.subheader("2) 上传并处理发票")
    invoice_files = st.file_uploader(
        "上传发票/报销附件 PDF（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="invoice_files",
    )
    auto_process = st.checkbox("上传后自动处理", value=True)
    if st.button("开始处理", use_container_width=True):
        if not invoice_files:
            st.warning("请先选择发票文件。")
            return
        for file in invoice_files:
            local_runner.create_and_process_task(
                file.name,
                file.getvalue(),
                auto_process=auto_process,
                auto_export=True,
            )
        st.success(f"已提交任务：{len(invoice_files)} 个。处理完成后会自动导出 Excel 和文本。")
        st.rerun()


def _render_export_download(task) -> None:
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
                )
        else:
            cols[1].warning("文本路径已记录，但文件不存在。")


def _render_manual_correction(task) -> None:
    st.subheader("4) 人工修正（可直接编辑明细表）")

    final_data = task.final_data or {}
    extracted_fields = final_data.get("extracted_fields") or task.extracted_data or {}
    initial_line_items = _get_initial_line_items(extracted_fields)

    with st.form(f"correction_form_{task.id}", clear_on_submit=False):
        expense_category = st.text_input("费用类别", value=_get_default(final_data, "expense_category"))
        required_materials = st.text_area(
            "所需材料（每行一条）",
            value="\n".join(final_data.get("required_materials", [])),
            height=100,
        )
        risk_points = st.text_area(
            "风险判断（每行一条）",
            value="\n".join(final_data.get("risk_points", [])),
            height=100,
        )

        st.markdown("**票头字段修正**")
        left, right = st.columns(2)
        invoice_number = left.text_input("发票号码", value=_get_default(extracted_fields, "invoice_number"))
        invoice_date = right.text_input("开票日期", value=_get_default(extracted_fields, "invoice_date"))
        amount = left.text_input("总金额（含税）", value=_get_default(extracted_fields, "amount"))
        tax_amount = right.text_input("税额", value=_get_default(extracted_fields, "tax_amount"))
        seller = left.text_input("销售方", value=_get_default(extracted_fields, "seller"))
        buyer = right.text_input("购买方", value=_get_default(extracted_fields, "buyer"))
        bill_type = left.text_input("票据类型", value=_get_default(extracted_fields, "bill_type"))
        item_content = right.text_input("项目内容", value=_get_default(extracted_fields, "item_content"))

        st.markdown("**明细表修正（支持增删改）**")
        edited_line_items = st.data_editor(
            initial_line_items,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "item_name": st.column_config.TextColumn("项目名称(含星号)", required=False, width="large"),
                "spec": st.column_config.TextColumn("规格型号", required=False, width="medium"),
                "quantity": st.column_config.TextColumn("数量", required=False, width="small"),
                "unit": st.column_config.TextColumn("单位", required=False, width="small"),
                "line_total_with_tax": st.column_config.TextColumn("每项含税总价", required=False, width="medium"),
            },
            key=f"line_items_editor_{task.id}",
        )

        normalized_preview = _normalize_line_items(_to_editor_rows(edited_line_items))
        line_total_sum = _line_items_total(normalized_preview)
        st.caption(f"当前明细合计（按“每项含税总价”求和）：{_format_amount(line_total_sum) if line_total_sum is not None else '-'}")
        sync_amount = st.checkbox("提交时用明细合计覆盖“总金额（含税）”", value=True)

        submit = st.form_submit_button("保存修正并重新导出", use_container_width=True)

    if not submit:
        return

    normalized_line_items = _normalize_line_items(_to_editor_rows(edited_line_items))
    if sync_amount:
        line_total_sum = _line_items_total(normalized_line_items)
        if line_total_sum is not None:
            amount = _format_amount(line_total_sum)

    corrected_fields = {
        "invoice_number": invoice_number.strip() or None,
        "invoice_date": invoice_date.strip() or None,
        "amount": amount.strip() or None,
        "tax_amount": tax_amount.strip() or None,
        "seller": seller.strip() or None,
        "buyer": buyer.strip() or None,
        "bill_type": bill_type.strip() or None,
        "item_content": item_content.strip() or None,
        "line_items": normalized_line_items,
    }

    corrections = {
        "expense_category": expense_category.strip() or None,
        "required_materials": _parse_lines(required_materials),
        "risk_points": _parse_lines(risk_points),
        "extracted_fields": corrected_fields,
    }

    local_runner.apply_corrections(task.id, corrections)
    local_runner.export_task(task.id, export_format="both")
    st.success("修正已保存，并已重新导出。")
    st.rerun()


def _render_task_section() -> None:
    st.subheader("3) 任务结果")
    tasks = local_runner.list_tasks(limit=200)
    if not tasks:
        st.info("暂无任务。")
        return

    selected_label = st.selectbox("选择任务", options=[_task_label(task) for task in tasks], index=0)
    task_id = selected_label.split(" | ")[0]
    task = local_runner.get_task(task_id)
    if task is None:
        st.error("任务不存在。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("状态", task.status)
    col2.metric("创建时间", str(task.created_at).split(".")[0])
    col3.metric("更新时间", str(task.updated_at).split(".")[0])

    c1, c2 = st.columns(2)
    if c1.button("重新处理该任务", use_container_width=True):
        local_runner.process_task(task.id)
        local_runner.export_task(task.id, export_format="both")
        st.success("任务已重新处理并重新导出。")
        st.rerun()
    if c2.button("导出 Excel + 文本", use_container_width=True):
        result = local_runner.export_task(task.id, export_format="both")
        st.success(f"导出完成：{result}")
        st.rerun()

    _render_export_download(task)

    with st.expander("查看抽取结果(JSON)", expanded=False):
        st.json(task.extracted_data or {})
    with st.expander("查看系统建议(JSON)", expanded=False):
        st.json(task.suggestion_data or {})
    with st.expander("查看最终结果(JSON)", expanded=False):
        st.json(task.final_data or {})

    _render_manual_correction(task)


def _render_material_flow() -> None:
    _render_policy_section()
    st.divider()
    _render_invoice_upload_section()
    st.divider()
    _render_task_section()


def _render_travel_flow() -> None:
    st.subheader("差旅费流程")
    st.caption("差旅流程不导出材料费样式 Excel，仅做材料完整性与金额一致性校验。")
    st.markdown(
        "- 去程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 返程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 酒店：发票 + 支付记录 + 平台订单截图（无住宿费用时需提供情况说明）"
    )

    go_section = _render_travel_transport_section("1) 出差去程交通报销", "travel_go")
    return_section = _render_travel_transport_section("2) 出差返程交通报销", "travel_return")
    hotel_section = _render_travel_hotel_section("travel_hotel")
    _render_travel_summary(go_section, return_section, hotel_section)
    st.divider()
    _render_travel_package_export()


def main() -> None:
    st.set_page_config(page_title="Finance Agent", layout="wide")
    st.title("财务 Agent（本地工具版）")
    st.caption("按费用类型选择流程：材料费 / 差旅费")

    init_runtime()

    flow_mode = st.radio(
        "选择报销流程",
        options=["材料费流程", "差旅费流程"],
        horizontal=True,
    )
    st.divider()

    if flow_mode == "材料费流程":
        _render_material_flow()
    else:
        _render_travel_flow()


if __name__ == "__main__":
    main()
