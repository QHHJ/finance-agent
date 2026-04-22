from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any
from datetime import datetime
from uuid import uuid4

import requests
import streamlit as st
from pypdf import PdfReader

from app.ui import home_router
from app.usecases import dto as usecase_dto
from app.usecases import material_agent as material_usecase
from app.usecases import travel_agent as travel_usecase

LINE_ITEM_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
TRUE_VALUES = {"1", "true", "yes", "on"}
TRAVEL_DOC_TYPES = {
    "transport_ticket",
    "transport_payment",
    "flight_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "unknown",
}
TRAVEL_RAG_KEYWORDS = [
    "差旅",
    "报销",
    "机票",
    "高铁",
    "火车",
    "交通",
    "酒店",
    "住宿",
    "订单截图",
    "行程单",
    "支付记录",
    "支付凭证",
    "发票",
    "票据",
    "入账",
    "抬头",
    "税号",
]
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


def _parse_json_object_loose(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        parsed = json.loads(source)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", source)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _pending_actions_key(scope: str) -> str:
    return f"{scope}_pending_actions"


def _last_action_key(scope: str) -> str:
    return f"{scope}_last_applied_action"


def _get_pending_actions(scope: str) -> list[dict[str, Any]]:
    key = _pending_actions_key(scope)
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


def _set_pending_actions(scope: str, actions: list[dict[str, Any]]) -> None:
    st.session_state[_pending_actions_key(scope)] = [dict(item) for item in actions if isinstance(item, dict)]


def _append_pending_action(
    scope: str,
    *,
    action_type: str,
    summary: str,
    target: str = "",
    risk_level: str = "medium",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions = _get_pending_actions(scope)
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
    _set_pending_actions(scope, actions)
    return action


def _update_pending_action(scope: str, action_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    target = str(action_id or "").strip()
    if not target:
        return None
    actions = _get_pending_actions(scope)
    updated = None
    for item in actions:
        if str(item.get("action_id") or "") != target:
            continue
        item.update(dict(patch or {}))
        updated = item
        break
    _set_pending_actions(scope, actions)
    return updated


def _remove_pending_action(scope: str, action_id: str) -> bool:
    target = str(action_id or "").strip()
    if not target:
        return False
    actions = _get_pending_actions(scope)
    filtered = [item for item in actions if str(item.get("action_id") or "") != target]
    changed = len(filtered) != len(actions)
    if changed:
        _set_pending_actions(scope, filtered)
    return changed


def _clear_pending_actions(scope: str) -> None:
    _set_pending_actions(scope, [])


def _record_last_applied_action(scope: str, action: dict[str, Any]) -> None:
    entry = usecase_dto.LastAppliedAction(
        action_id=str(action.get("action_id") or uuid4().hex),
        action_type=str(action.get("action_type") or ""),
        summary=str(action.get("summary") or ""),
        scope=scope,
        applied_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ).to_dict()
    st.session_state[_last_action_key(scope)] = entry


def _get_last_applied_action(scope: str) -> dict[str, Any] | None:
    value = st.session_state.get(_last_action_key(scope))
    return dict(value) if isinstance(value, dict) else None


def _clear_last_applied_action(scope: str) -> None:
    st.session_state.pop(_last_action_key(scope), None)


def _compose_three_stage_reply(understand: str, status_change: str, next_step: str) -> str:
    part1 = str(understand or "好的，我理解了。").strip()
    part2 = str(status_change or "当前状态保持不变。").strip()
    part3 = str(next_step or "你可以继续告诉我希望我怎么处理。").strip()
    return f"{part1}\n\n{part2}\n\n{part3}"


def _inject_ui_styles() -> None:
    st.markdown(
        """
<style>
/* Chat input: make boundary obvious and consistent */
div[data-testid="stChatInput"] {
  border: 1.5px solid #c7d2fe !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06) !important;
  padding: 4px 8px !important;
  margin-top: 6px !important;
}

div[data-testid="stChatInput"]:focus-within {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16) !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
  background: #ffffff !important;
  color: #0f172a !important;
}

div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] input::placeholder {
  color: #6b7280 !important;
  opacity: 1 !important;
}

/* Generic input field visibility (for fallback text boxes) */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  background: #ffffff !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _infer_intent_with_llm(message: str, domain: str) -> usecase_dto.IntentParseResult | None:
    text = str(message or "").strip()
    if not text:
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()
    prompt = (
        "你是意图分类器。仅返回JSON对象，不要输出其他内容。"
        "intent_type 只能是: chat, light_edit, strong_action, ambiguous。"
        "risk_level 只能是: low, medium, high。"
        "needs_confirmation 为布尔值。"
        "is_actionable 为布尔值。"
        "示例: {\"intent_type\":\"chat\",\"is_actionable\":false,\"risk_level\":\"low\",\"needs_confirmation\":false,\"reason\":\"...\"}\n"
        f"domain={domain}\n"
        f"message={text}\n"
    )
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
        return None

    parsed = _parse_json_object_loose(content)
    if not parsed:
        return None
    intent = str(parsed.get("intent_type") or "").strip().lower()
    if intent not in {"chat", "light_edit", "strong_action", "ambiguous"}:
        return None
    risk = str(parsed.get("risk_level") or "low").strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "low"
    return usecase_dto.IntentParseResult(
        intent_type=intent,
        is_actionable=bool(parsed.get("is_actionable")),
        risk_level=risk,
        needs_confirmation=bool(parsed.get("needs_confirmation")),
        reason=str(parsed.get("reason") or ""),
    )


def classify_user_message_intent(message: str, context: dict[str, Any] | None = None) -> usecase_dto.IntentParseResult:
    text = str(message or "").strip()
    lower = text.lower()
    domain = str((context or {}).get("domain") or "generic")
    if not text:
        return usecase_dto.IntentParseResult(intent_type="chat", reason="empty_message")

    strong_tokens = [
        "应用全部修正",
        "应用全部建议",
        "覆盖当前分配结果",
        "重新归并",
        "批量覆盖",
        "批量应用",
        "导出报销表",
        "导出结果",
    ]
    if any(token in text for token in strong_tokens):
        return usecase_dto.IntentParseResult(
            intent_type="strong_action",
            is_actionable=True,
            risk_level="high",
            needs_confirmation=True,
            reason="matched_strong_tokens",
        )

    ambiguous_tokens = ["不太对", "怪怪的", "不对劲", "再看看", "你再看看", "感觉有问题", "这个有问题"]
    if any(token in text for token in ambiguous_tokens):
        return usecase_dto.IntentParseResult(
            intent_type="ambiguous",
            is_actionable=False,
            risk_level="low",
            needs_confirmation=False,
            reason="matched_ambiguous_tokens",
        )

    chat_tokens = ["还缺什么", "为什么", "哪里金额不一致", "怎么分配", "说明", "解释", "全不全", "齐不齐", "有哪些问题"]
    if "?" in text or "？" in text or any(token in text for token in chat_tokens):
        return usecase_dto.IntentParseResult(
            intent_type="chat",
            is_actionable=False,
            risk_level="low",
            needs_confirmation=False,
            reason="matched_chat_tokens",
        )

    if domain == "travel":
        relabel_markers = ["这个是", "这张是", "应该是", "应归为", "算", "归到", "归类到", "改为", "改成", "类型改为", "类型是"]
        doc_markers = [
            "机票明细",
            "酒店发票",
            "交通票据",
            "支付记录",
            "酒店订单",
            "订单截图",
            "机票发票",
            "高铁报销凭证",
            "发票",
            "票据",
            "支付凭证",
        ]
        has_file_name_hint = bool(re.search(r"\.(pdf|jpg|jpeg|png|webp)\b", lower))
        has_relabel_phrase = any(token in text for token in relabel_markers) or re.search(
            r"(?:是|改为|改成|归为|归类为).{0,10}(?:发票|票据|明细|支付|订单|截图)",
            text,
        )
        if _is_reclassify_command(text):
            bulk_marker = any(token in text for token in ["全部", "所有", "这批", "批量", "都"])
            return usecase_dto.IntentParseResult(
                intent_type="strong_action" if bulk_marker else "light_edit",
                is_actionable=True,
                risk_level="medium" if bulk_marker else "low",
                needs_confirmation=bulk_marker,
                reason="travel_reidentify",
            )
        if has_relabel_phrase and any(token in text for token in doc_markers):
            return usecase_dto.IntentParseResult(
                intent_type="light_edit",
                is_actionable=True,
                risk_level="low",
                needs_confirmation=False,
                reason="travel_manual_relabel",
            )
        if has_file_name_hint and any(token in text for token in ["发票", "票据", "明细", "支付", "订单截图", "订单"]):
            return usecase_dto.IntentParseResult(
                intent_type="light_edit",
                is_actionable=True,
                risk_level="low",
                needs_confirmation=False,
                reason="travel_filetype_short_edit",
            )
        amount_set_match = re.search(
            r"(?:金额|总价|价税合计|含税|小写|支付金额)[^\d¥￥\-]{0,12}"
            r"(?:改为|改成|是|为|设为|写成|填成|调整为|=)\s*[¥￥]?\s*-?\d",
            text,
            flags=re.IGNORECASE,
        )
        if amount_set_match:
            return usecase_dto.IntentParseResult(
                intent_type="light_edit",
                is_actionable=True,
                risk_level="low",
                needs_confirmation=False,
                reason="travel_amount_edit",
            )
        if any(token in text for token in ["一趟", "同一趟", "同一次", "去程和返程"]):
            return usecase_dto.IntentParseResult(
                intent_type="strong_action",
                is_actionable=True,
                risk_level="medium",
                needs_confirmation=True,
                reason="travel_grouping_adjust",
            )

    if domain == "material":
        if any(token in text for token in ["删除", "新增一行", "添加一行", "改为", "应为", "设置为", "设为"]) and any(
            token in text for token in ["第", "行", "最后一行", "倒数第", "项目名称", "规格", "数量", "单位", "金额"]
        ):
            return usecase_dto.IntentParseResult(
                intent_type="light_edit",
                is_actionable=True,
                risk_level="low",
                needs_confirmation=False,
                reason="material_row_edit",
            )
        if any(token in text for token in ["重新识别", "智能修复", "应用llm修复表", "应用llm结果", "应用对比结果"]):
            return usecase_dto.IntentParseResult(
                intent_type="strong_action",
                is_actionable=True,
                risk_level="high",
                needs_confirmation=True,
                reason="material_high_impact",
            )

    llm_guess = _infer_intent_with_llm(text, domain)
    if llm_guess is not None:
        return llm_guess

    return usecase_dto.IntentParseResult(
        intent_type="chat",
        is_actionable=False,
        risk_level="low",
        needs_confirmation=False,
        reason="default_chat_fallback",
    )


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


@st.cache_data(show_spinner=False)
def _extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join(chunk for chunk in pages if chunk)


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


def _env_flag_true(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in TRUE_VALUES


def _vl_model() -> str:
    return os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")


def _text_model() -> str:
    return os.getenv("OLLAMA_TEXT_MODEL") or os.getenv("OLLAMA_CHAT_MODEL") or _vl_model()


def _chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL") or _text_model()


def _current_model_config() -> dict[str, Any]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    vl_model = _vl_model()
    text_model = _text_model()
    chat_model = _chat_model()
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return {
        "use_ollama_vl": _env_flag_true("USE_OLLAMA_VL"),
        "base_url": base_url,
        "vl_model": vl_model,
        "text_model": text_model,
        "chat_model": chat_model,
        "embed_model": embed_model,
    }


def _get_ollama_runtime_rows(base_url: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=(4, 12))
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return [], str(exc)

    rows: list[dict[str, Any]] = []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return rows, None

    for item in models:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "模型": str(item.get("name") or ""),
                "处理器占用": str(item.get("details", {}).get("processor") or item.get("processor") or ""),
                "上下文": str(item.get("details", {}).get("context_length") or item.get("context") or ""),
                "驻留到": str(item.get("expires_at") or item.get("until") or ""),
                "大小": str(item.get("size") or ""),
            }
        )
    return rows, None


def _render_model_runtime_panel() -> None:
    cfg = _current_model_config()
    with st.expander("当前模型与运行状态", expanded=False):
        st.markdown(f"- 视觉抽取模型：`{cfg['vl_model']}`")
        st.markdown(f"- 文本抽取/修复模型：`{cfg['text_model']}`")
        st.markdown(f"- 对话模型：`{cfg['chat_model']}`")
        st.markdown(f"- 向量模型：`{cfg['embed_model']}`")
        st.markdown(f"- Ollama 地址：`{cfg['base_url']}`")
        st.markdown(f"- 启用视觉抽取：`{cfg['use_ollama_vl']}`")

        rows, err = _get_ollama_runtime_rows(cfg["base_url"])
        if err:
            st.caption(f"未能获取运行中模型状态：{err}")
            return
        if not rows:
            st.caption("当前没有模型驻留（或 Ollama 暂无活跃会话）。")
            return
        st.dataframe(rows, use_container_width=True, hide_index=True)


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


def _sha1_of_bytes(data: bytes) -> str:
    if not data:
        return ""
    return hashlib.sha1(data).hexdigest()


@st.cache_data(show_spinner=False)
def _extract_image_text_with_ollama(file_bytes: bytes) -> str:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return ""

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _vl_model()
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    prompt = (
        "请做OCR，只输出图片中的主要文本内容，尽量保持阅读顺序。"
        "不要解释，不要总结，不要输出JSON。"
    )

    for mode in ("chat", "generate"):
        try:
            if mode == "chat":
                payload = {
                    "model": model,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = str((resp.json().get("message") or {}).get("content") or "").strip()
            else:
                payload = {
                    "model": model,
                    "stream": False,
                    "prompt": prompt,
                    "images": [encoded],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = str(resp.json().get("response") or "").strip()
            if content:
                content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                return content[:4000]
        except Exception:
            continue
    return ""


def _normalize_confidence(value: Any) -> float | None:
    confidence = _safe_float(value)
    if confidence is None:
        return None
    if confidence > 1:
        confidence /= 100.0
    if confidence < 0 or confidence > 1:
        return None
    return confidence


def _extract_travel_retrieval_terms(raw_text: str) -> list[str]:
    text = (raw_text or "").strip().lower()
    if not text:
        return TRAVEL_RAG_KEYWORDS[:8]

    terms: list[str] = []
    for keyword in TRAVEL_RAG_KEYWORDS:
        if keyword.lower() in text:
            terms.append(keyword)

    # Additional numeric/context terms that often appear in payment/ticket pages.
    extra = re.findall(r"[a-z0-9\-/]{3,24}", text)
    for token in extra:
        if token in terms:
            continue
        if token in {"http", "https", "www"}:
            continue
        terms.append(token)
        if len(terms) >= 20:
            break
    return terms[:20] if terms else TRAVEL_RAG_KEYWORDS[:8]


@st.cache_data(show_spinner=False)
def _load_travel_policy_corpus() -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    try:
        policies = travel_usecase.list_policies(limit=200)
    except Exception:
        return corpus

    for policy in policies:
        name = str(getattr(policy, "name", "") or "").strip() or "policy"
        raw_text = str(getattr(policy, "raw_text", "") or "").strip()
        if not raw_text:
            continue
        text = re.sub(r"\s+", " ", raw_text)
        if len(text) > 40000:
            text = text[:40000]
        corpus.append({"name": name, "text": text})
    return corpus


def _policy_snippet_around_terms(text: str, terms: list[str], max_chars: int = 420) -> str:
    if not text:
        return ""
    for term in terms:
        if not term:
            continue
        idx = text.find(term)
        if idx < 0:
            continue
        start = max(0, idx - 120)
        end = min(len(text), idx + max_chars - 120)
        return text[start:end]
    return text[:max_chars]


def _build_travel_rag_context(raw_text: str) -> str:
    try:
        context = travel_usecase.build_travel_policy_context(raw_text, top_k=3)
        if context:
            return context
    except Exception:
        pass

    corpus = _load_travel_policy_corpus()
    if not corpus:
        return ""

    terms = _extract_travel_retrieval_terms(raw_text)
    snippets: list[str] = []
    for idx, doc in enumerate(corpus[:2], start=1):
        snippet = _policy_snippet_around_terms(doc["text"], terms, max_chars=420)
        snippets.append(f"[Policy#{idx} score=fallback name={doc['name']}] {snippet}")
    return "\n".join(snippets)


def _rule_confidence_for_doc_type(doc_type: str, raw_text: str) -> int:
    features = _travel_signal_features("", raw_text)
    hotel_hit = features["hotel_hit"]
    transport_hit = features["transport_hit"]
    invoice_hit = features["invoice_hit"]
    payment_hit = features["payment_hit"]
    detail_hit = features["detail_hit"]
    order_hit = features["order_strong_hit"]

    if doc_type == "hotel_order":
        return 3 if order_hit else 0
    if doc_type == "flight_detail":
        return 3 if detail_hit else 0
    if doc_type == "transport_ticket":
        if invoice_hit and transport_hit and not hotel_hit:
            return 3
        if invoice_hit and transport_hit:
            return 2
        return 1 if transport_hit else 0
    if doc_type == "hotel_invoice":
        if invoice_hit and hotel_hit and not transport_hit:
            return 3
        if invoice_hit and hotel_hit:
            return 2
        return 1 if hotel_hit else 0
    if doc_type == "transport_payment":
        if payment_hit and transport_hit and not hotel_hit:
            return 3
        if payment_hit and transport_hit:
            return 2
        return 1 if payment_hit else 0
    if doc_type == "hotel_payment":
        if payment_hit and hotel_hit and not transport_hit:
            return 3
        if payment_hit and hotel_hit:
            return 2
        return 1 if payment_hit else 0
    return 0


def _resolve_travel_doc_type(
    rule_guess: str,
    llm_guess: str,
    llm_confidence: float | None,
    raw_text: str,
) -> tuple[str, str]:
    normalized_rule = rule_guess if rule_guess in TRAVEL_DOC_TYPES else "unknown"
    normalized_llm = llm_guess if llm_guess in TRAVEL_DOC_TYPES else "unknown"

    if normalized_llm == "unknown":
        if normalized_rule != "unknown":
            return normalized_rule, "rule"
        return "unknown", "rule_unknown"

    if normalized_rule == "unknown":
        return normalized_llm, "llm"

    if normalized_llm == normalized_rule:
        return normalized_llm, "llm+rule_agree"

    rule_strength = _rule_confidence_for_doc_type(normalized_rule, raw_text)
    merged = re.sub(r"\s+", "", (raw_text or "").lower())
    invoice_core_count = _keyword_score(
        merged,
        ["电子发票", "发票号码", "开票日期", "购买方", "销售方", "价税合计"],
    )
    payment_core_count = _keyword_score(
        merged,
        ["账单详情", "交易成功", "支付时间", "付款方式", "商品说明", "交易单号", "交易号", "收单机构", "商户单号", "账单分类"],
    )

    # Guard 1: invoice-like documents should not be overridden to payment by a high-confidence LLM guess.
    if normalized_rule in {"transport_ticket", "hotel_invoice"} and normalized_llm in {"transport_payment", "hotel_payment"}:
        if invoice_core_count >= 2 and payment_core_count <= 1:
            return normalized_rule, "rule_guard_invoice"

    # Guard 2: structured detail pages should not be overridden to payment easily.
    if normalized_rule in {"flight_detail", "hotel_order"} and normalized_llm in {"transport_payment", "hotel_payment"}:
        if rule_strength >= 2:
            return normalized_rule, "rule_guard_structured"

    # For strongly structured screens (itinerary detail / hotel order detail),
    # keep strict rule guard to avoid frequent "payment" over-classification.
    if rule_strength >= 3 and normalized_rule in {"flight_detail", "hotel_order"}:
        return normalized_rule, "rule_guard_strict"
    if rule_strength >= 3 and (llm_confidence is None or llm_confidence < 0.8):
        return normalized_rule, "rule_guard"
    if llm_confidence is not None and llm_confidence < 0.45 and rule_strength >= 2:
        return normalized_rule, "rule_guard"

    return normalized_llm, "llm_override"


def _extract_doc_type_from_case_hit(hit: dict[str, Any]) -> str:
    meta = dict(hit.get("metadata") or {})
    doc_type = str(meta.get("doc_type") or "").strip()
    if doc_type in TRAVEL_DOC_TYPES:
        return doc_type

    content = str(hit.get("content") or "")
    match = re.search(r"doc_type\s*:\s*([a-z_]+)", content)
    if not match:
        return "unknown"
    guessed = str(match.group(1) or "").strip()
    return guessed if guessed in TRAVEL_DOC_TYPES else "unknown"


def _lookup_learned_doc_type_override(
    file_sha1: str,
    file_name: str,
    signal_text: str,
    current_doc_type: str,
) -> tuple[str | None, str]:
    normalized_current = current_doc_type if current_doc_type in TRAVEL_DOC_TYPES else "unknown"
    key = str(file_sha1 or "").strip()
    name = str(file_name or "").strip()
    signal = str(signal_text or "").strip()

    if key:
        try:
            exact_hits = travel_usecase.retrieve_travel_case_hits(
                query=f"file_sha1:{key}",
                top_k=4,
                metadata_filter={"case_kind": "file_doc_type", "file_sha1": key},
            )
        except Exception:
            exact_hits = []
        for hit in exact_hits:
            doc_type = _extract_doc_type_from_case_hit(hit)
            if doc_type in TRAVEL_DOC_TYPES and doc_type != "unknown":
                return doc_type, "learned_file_hash"

    if name:
        try:
            name_hits = travel_usecase.retrieve_travel_case_hits(
                query=name,
                top_k=6,
            )
        except Exception:
            name_hits = []
        name_lower = name.lower()
        for hit in name_hits:
            title = str(hit.get("title") or "").lower()
            meta = dict(hit.get("metadata") or {})
            meta_name = str(meta.get("file_name") or "").lower()
            if name_lower and name_lower not in title and name_lower != meta_name:
                continue
            doc_type = _extract_doc_type_from_case_hit(hit)
            if doc_type in TRAVEL_DOC_TYPES and doc_type != "unknown":
                return doc_type, "learned_file_name"

    if not signal:
        return None, ""

    try:
        semantic_hits = travel_usecase.retrieve_travel_case_hits(
            query=signal[:1500],
            top_k=8,
            metadata_filter={"case_kind": "file_doc_type"},
        )
    except Exception:
        semantic_hits = []

    vote_scores: dict[str, float] = {}
    for hit in semantic_hits:
        doc_type = _extract_doc_type_from_case_hit(hit)
        if doc_type not in TRAVEL_DOC_TYPES or doc_type == "unknown":
            continue
        meta = dict(hit.get("metadata") or {})
        score = float(hit.get("score") or 0.0)
        if str(meta.get("source") or "").startswith("manual"):
            score += 0.12
        if str(meta.get("reason") or "").startswith("manual"):
            score += 0.08
        if str(meta.get("file_sha1") or "") == key and key:
            score += 1.0
        vote_scores[doc_type] = vote_scores.get(doc_type, 0.0) + max(0.0, score)

    if not vote_scores:
        return None, ""

    ordered = sorted(vote_scores.items(), key=lambda item: item[1], reverse=True)
    best_doc_type, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = best_score - second_score

    if normalized_current == "unknown" and best_score >= 0.75 and margin >= 0.15:
        return best_doc_type, "learned_case"

    pair = {"transport_payment", "flight_detail"}
    if normalized_current in pair and best_doc_type in pair and best_doc_type != normalized_current:
        if best_score >= 0.82 and margin >= 0.12:
            return best_doc_type, "learned_case_pair_fix"

    return None, ""


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


def _extract_invoice_total_with_tax_from_text(raw_text: str) -> float | None:
    text = str(raw_text or "").strip()
    if not text:
        return None

    normalized = text.replace("（", "(").replace("）", ")")
    amount_pattern = r"([¥￥]?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"

    # Highest priority: 价税合计(小写)金额。
    for pattern in [
        rf"价税合计\s*\(\s*小写\s*\)\s*[:：]?\s*{amount_pattern}",
        rf"\(\s*小写\s*\)\s*[:：]?\s*{amount_pattern}",
        rf"小写\s*[:：]?\s*{amount_pattern}",
    ]:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        amount = _normalize_payment_amount(match.group(1))
        if amount is not None:
            return amount

    # Next: explicit 价税合计数值。
    match = re.search(
        rf"价税合计\s*[:：]?\s*{amount_pattern}",
        normalized,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        amount = _normalize_payment_amount(match.group(1))
        if amount is not None:
            return amount

    # Common invoice line: 合计 不含税 税额 => 含税总价 = 两者之和。
    total_line_match = re.search(
        rf"(?:^|[\s\r\n])合计[^\n\r]{{0,80}}?{amount_pattern}\s+{amount_pattern}",
        normalized,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if total_line_match:
        no_tax_total = _normalize_payment_amount(total_line_match.group(1))
        tax_total = _normalize_payment_amount(total_line_match.group(2))
        if no_tax_total is not None and tax_total is not None:
            return float(no_tax_total + tax_total)

    return None


@st.cache_data(show_spinner=False)
def _extract_payment_amount_with_ollama(file_bytes: bytes) -> float | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _vl_model()
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
    model = _vl_model()
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
            total_with_tax = _extract_invoice_total_with_tax_from_text(raw_text)
            if total_with_tax is not None:
                return total_with_tax

            extracted = material_usecase.extract_invoice_fields(raw_text)
            amount = _safe_float(extracted.get("amount"))
            tax_amount = _safe_float(extracted.get("tax_amount"))
            if amount is not None and tax_amount is not None:
                # Heuristic: when extracted amount is no-tax and tax exists, amount+tax should
                # better match ticket reimbursement than amount itself for transport/hotel invoices.
                has_small_label = re.search(
                    r"(?:价税合计\s*[（(]小写[)）]|[（(]小写[)）])",
                    raw_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if not has_small_label and tax_amount > 0 and abs(tax_amount) < abs(amount):
                    candidate_total = float(amount + tax_amount)
                    if candidate_total > 0:
                        return candidate_total
            if amount is not None:
                return amount
            amount = _extract_amount_from_text(raw_text)
            if amount is not None:
                return amount
            # Last text-only fallback uses model parsing, still based on content.
            return _extract_payment_amount_with_ollama_text(raw_text)
        except Exception:
            return None

    # 对图片票据先做 OCR+规则抽取“价税合计(小写)”，再回退到模型金额识别。
    image_text = _extract_image_text_with_ollama(file_bytes)
    total_with_tax = _extract_invoice_total_with_tax_from_text(image_text)
    if total_with_tax is not None:
        return total_with_tax

    amount = _extract_payment_amount_with_ollama(file_bytes)
    if amount is not None:
        return amount
    return None


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
    return travel_usecase.as_uploaded_list(uploaded_value)


def _files_signature(files: list[Any]) -> str:
    parts: list[str] = []
    for file in files:
        name = str(getattr(file, "name", ""))
        size = str(getattr(file, "size", ""))
        parts.append(f"{name}:{size}")
    return "|".join(parts)


def _travel_file_key(name: str, size: Any) -> str:
    return f"{str(name or '').strip()}:{str(size or '').strip()}"


def _uploaded_file_key(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    return _travel_file_key(
        str(getattr(uploaded_file, "name", "")),
        getattr(uploaded_file, "size", ""),
    )


def _profile_file_key(profile: dict[str, Any]) -> str:
    file_obj = profile.get("file")
    size = getattr(file_obj, "size", "") if file_obj is not None else ""
    return _travel_file_key(str(profile.get("name") or ""), size)


def _uploaded_file_size_label(size_value: Any) -> str:
    try:
        size = int(size_value or 0)
    except Exception:
        size = 0
    if size <= 0:
        return "-"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _render_included_file_list(
    *,
    flow_label: str,
    page_uploaded_files: list[Any],
    guide_files: list[Any],
    merged_files: list[Any],
) -> None:
    guide_list = _as_uploaded_list(guide_files)
    if not guide_list:
        return
    page_list = _as_uploaded_list(page_uploaded_files)
    merged_list = _as_uploaded_list(merged_files)
    guide_keys = {_uploaded_file_key(item) for item in guide_list}
    page_keys = {_uploaded_file_key(item) for item in page_list}
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(merged_list, start=1):
        key = _uploaded_file_key(item)
        source_tags: list[str] = []
        if key in guide_keys:
            source_tags.append("首页带入")
        if key in page_keys:
            source_tags.append("当前页上传")
        rows.append(
            {
                "序号": idx,
                "文件名": str(getattr(item, "name", "") or ""),
                "大小": _uploaded_file_size_label(getattr(item, "size", 0)),
                "来源": " + ".join(source_tags) if source_tags else "流程缓存",
            }
        )

    with st.expander(f"查看{flow_label}已纳入文件清单（共 {len(rows)} 份）", expanded=True):
        st.caption("说明：上传控件仅显示当前页手动选择文件；首页带入文件也会出现在下表并参与处理。")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _set_manual_override_for_profile(manual_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    doc_type = str(profile.get("doc_type") or "").strip()
    if not key or doc_type not in TRAVEL_DOC_TYPES:
        return
    manual_overrides[key] = doc_type


def _remember_manual_overrides(manual_overrides: dict[str, str], profiles: list[dict[str, Any]]) -> int:
    updated = 0
    for profile in profiles:
        source = str(profile.get("source") or "")
        if source not in {"manual_chat", "manual_table", "manual_persist"}:
            continue
        key = _profile_file_key(profile)
        doc_type = str(profile.get("doc_type") or "").strip()
        if not key or doc_type not in TRAVEL_DOC_TYPES:
            continue
        if manual_overrides.get(key) == doc_type:
            continue
        manual_overrides[key] = doc_type
        updated += 1
    return updated


def _remove_manual_override_for_profile(manual_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    if key:
        manual_overrides.pop(key, None)


def _prune_manual_overrides(manual_overrides: dict[str, str], pool_files: list[Any]) -> None:
    valid_keys = {_uploaded_file_key(file) for file in pool_files if file is not None}
    for key in list(manual_overrides.keys()):
        if key not in valid_keys:
            manual_overrides.pop(key, None)


def _apply_manual_overrides_to_profiles(profiles: list[dict[str, Any]], manual_overrides: dict[str, str]) -> int:
    if not manual_overrides:
        return 0
    changed = 0
    for profile in profiles:
        key = _profile_file_key(profile)
        target_doc_type = str(manual_overrides.get(key) or "").strip()
        if target_doc_type not in TRAVEL_DOC_TYPES:
            continue
        current_doc_type = str(profile.get("doc_type") or "unknown")
        current_source = str(profile.get("source") or "")
        if current_doc_type != target_doc_type or not current_source.startswith("manual"):
            profile["doc_type"] = target_doc_type
            profile["source"] = "manual_persist"
            changed += 1
    return changed


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


def _parse_date_value(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_candidate_dates(text: str) -> list[datetime]:
    content = text or ""
    candidates: list[datetime] = []

    for match in re.finditer(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", content):
        year, month, day = match.groups()
        try:
            candidates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue

    for match in re.finditer(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", content):
        year, month, day = match.groups()
        try:
            candidates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue

    # Keep unique dates and preserve order.
    seen: set[str] = set()
    unique: list[datetime] = []
    for dt in candidates:
        key = dt.strftime("%Y-%m-%d")
        if key in seen:
            continue
        seen.add(key)
        unique.append(dt)
    return unique


def _pick_primary_date(file_name: str, raw_text: str) -> datetime | None:
    text_dates = _extract_candidate_dates(raw_text)
    if text_dates:
        return text_dates[0]
    file_dates = _extract_candidate_dates(file_name)
    if file_dates:
        return file_dates[0]
    return None


def _keyword_score(text: str, keywords: list[str]) -> int:
    return sum(1 for key in keywords if key in text)


def _is_generic_evidence(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    compact = re.sub(r"\s+", "", value)
    generic_markers = {
        "shortreason",
        "reason",
        "unknown",
        "n/a",
        "na",
        "none",
        "无法判断",
        "无法识别",
        "不确定",
        "看不清",
        "无",
        "空",
    }
    if compact in generic_markers:
        return True
    if len(compact) <= 3:
        return True
    return False


def _infer_doc_type_from_visual_keywords(text: str) -> str:
    merged = re.sub(r"\s+", "", (text or "").lower())
    if not merged:
        return "unknown"

    hotel_order_keys = [
        "携程旅行",
        "飞猪旅行",
        "同程旅行",
        "美团酒店",
        "酒店订单",
        "订单详情",
        "订单号",
        "晚明细",
        "费用明细",
        "取消政策",
        "已完成",
        "在线付",
    ]
    flight_detail_keys = [
        "价格明细",
        "机建",
        "燃油",
        "票价",
        "普通成人",
        "退改签",
        "行程单",
        "航段",
        "机票详情",
    ]
    payment_core_keys = [
        "账单详情",
        "交易成功",
        "支付时间",
        "付款方式",
        "商品说明",
        "收单机构",
        "交易单号",
        "交易号",
        "账单分类",
    ]
    transport_keys = [
        "机票",
        "航班",
        "高铁",
        "火车",
        "铁路",
        "航空",
        "客票",
        "代订机票",
        "机建",
        "燃油",
    ]
    hotel_keys = [
        "酒店",
        "住宿",
        "房费",
        "宾馆",
        "旅馆",
        "华住",
        "汉庭",
        "全季",
        "如家",
        "亚朵",
    ]
    invoice_keys = [
        "电子发票",
        "发票号码",
        "开票日期",
        "购买方",
        "销售方",
        "价税合计",
        "税额",
        "经纪代理服务",
        "客运服务",
        "代订机票费",
    ]

    if any(k in merged for k in hotel_order_keys):
        return "hotel_order"
    if any(k in merged for k in flight_detail_keys):
        return "flight_detail"

    payment_core_count = _keyword_score(merged, payment_core_keys)
    payment_hit = payment_core_count > 0
    transport_hit = any(k in merged for k in transport_keys)
    hotel_hit = any(k in merged for k in hotel_keys)
    invoice_core_keys = ["电子发票", "发票号码", "开票日期", "购买方", "销售方", "价税合计"]
    invoice_core_count = _keyword_score(merged, invoice_core_keys)
    invoice_hit = any(k in merged for k in invoice_keys) or ("发票" in merged)

    # Strong invoice header markers should win over payment-like words.
    if invoice_core_count >= 2:
        if "代订机票费" in merged or (transport_hit and not hotel_hit):
            return "transport_ticket"
        if "住宿服务" in merged or (hotel_hit and not transport_hit):
            return "hotel_invoice"
        if transport_hit:
            return "transport_ticket"
        if hotel_hit:
            return "hotel_invoice"

    if invoice_hit:
        if transport_hit and not hotel_hit:
            return "transport_ticket"
        if hotel_hit and not transport_hit:
            return "hotel_invoice"
        if "代订机票费" in merged:
            return "transport_ticket"
        if "住宿服务" in merged:
            return "hotel_invoice"

    # Payment classification requires stronger payment evidence, otherwise it is too noisy.
    if payment_core_count >= 2 and invoice_core_count == 0:
        if hotel_hit and not transport_hit:
            return "hotel_payment"
        if transport_hit and not hotel_hit:
            return "transport_payment"
        if "在线付" in merged and hotel_hit:
            return "hotel_payment"
        return "transport_payment"

    if payment_hit and invoice_core_count == 0:
        if hotel_hit and not transport_hit:
            return "hotel_payment"
        return "transport_payment"

    if hotel_hit and not transport_hit:
        return "hotel_invoice"
    if transport_hit and not hotel_hit:
        return "transport_ticket"
    return "unknown"


def _travel_signal_features(file_name: str, raw_text: str) -> dict[str, bool]:
    combined = f"{file_name}\n{raw_text}".lower()
    combined = re.sub(r"\s+", "", combined)

    hotel_hit = any(
        key in combined
        for key in ["酒店", "住宿", "房费", "入住", "离店", "旅馆", "宾馆", "hotel", "checkin", "checkout"]
    )
    transport_hit = any(
        key in combined
        for key in [
            "机票",
            "航班",
            "高铁",
            "火车",
            "铁路",
            "航空",
            "乘机",
            "起飞",
            "到达",
            "客票",
            "舱位",
            "boarding",
            "flight",
            "train",
        ]
    )
    invoice_hit = any(
        key in combined
        for key in ["发票", "电子发票", "普通发票", "专用发票", "税额", "价税合计", "购买方", "销售方", "invoice"]
    )
    payment_hit = any(
        key in combined
        for key in [
            "支付凭证",
            "支付记录",
            "交易成功",
            "微信支付",
            "支付宝",
            "余额宝",
            "银行",
            "交易单号",
            "支付时间",
            "payment",
            "transaction",
            "receipt",
        ]
    )
    detail_hit = any(
        key in combined
        for key in [
            "机票明细",
            "价格明细",
            "票价明细",
            "普通成人",
            "机建",
            "燃油",
            "票价",
            "行程单",
            "电子客票行程单",
            "客票行程",
            "航段",
            "itinerary",
            "tripdetail",
        ]
    )

    platform_hit = any(key in combined for key in ["携程", "飞猪", "美团", "同程", "booking", "trip.com"])
    # Do NOT use only "订单号" / "预订" to classify as hotel order, too noisy.
    order_strong_hit = any(
        key in combined
        for key in ["订单截图", "酒店订单", "携程订单", "飞猪订单", "晚明细", "费用明细", "取消政策", "已完成"]
    ) or ("订单" in combined and (hotel_hit or platform_hit) and ("入住" in combined or "离店" in combined))

    return {
        "hotel_hit": hotel_hit,
        "transport_hit": transport_hit,
        "invoice_hit": invoice_hit,
        "payment_hit": payment_hit,
        "detail_hit": detail_hit,
        "order_strong_hit": order_strong_hit,
        "platform_hit": platform_hit,
    }


def _guess_travel_doc_type(file_name: str, raw_text: str) -> str:
    features = _travel_signal_features(file_name, raw_text)
    hotel_hit = features["hotel_hit"]
    transport_hit = features["transport_hit"]
    invoice_hit = features["invoice_hit"]
    payment_hit = features["payment_hit"]
    detail_hit = features["detail_hit"]
    order_strong_hit = features["order_strong_hit"]

    if order_strong_hit:
        return "hotel_order"

    if detail_hit:
        return "flight_detail"

    # Invoices have higher priority than payment, otherwise invoice PDFs are often misread as payment.
    if invoice_hit:
        if transport_hit and not hotel_hit:
            return "transport_ticket"
        if hotel_hit and not transport_hit:
            return "hotel_invoice"

    if payment_hit:
        if hotel_hit and not transport_hit:
            return "hotel_payment"
        if transport_hit and not hotel_hit:
            return "transport_payment"
        return "transport_payment"

    if hotel_hit and invoice_hit and not transport_hit:
        return "hotel_invoice"

    if transport_hit and invoice_hit and not hotel_hit:
        return "transport_ticket"

    if hotel_hit and not transport_hit:
        return "hotel_invoice"
    if transport_hit and not hotel_hit:
        return "transport_ticket"
    return "unknown"


def _infer_doc_type_from_invoice_fields(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "unknown"
    try:
        fields = material_usecase.extract_invoice_fields(text)
    except Exception:
        return "unknown"

    item_content = str(fields.get("item_content") or "")
    seller = str(fields.get("seller") or "")
    bill_type = str(fields.get("bill_type") or "")
    merged = f"{item_content} {seller} {bill_type}".lower()

    transport_keys = ["航空", "机票", "客运服务", "航班", "铁路", "高铁", "火车", "客票"]
    hotel_keys = ["酒店", "住宿", "房费", "旅馆", "宾馆"]

    if any(key in merged for key in transport_keys):
        return "transport_ticket"
    if any(key in merged for key in hotel_keys):
        return "hotel_invoice"
    return "unknown"


@st.cache_data(show_spinner=False)
def _classify_travel_doc_with_ollama(
    file_bytes: bytes,
    suffix: str,
    file_name: str,
    raw_text: str,
    rule_hint: str = "unknown",
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    # Keep parameter for cache key control when user requests forced re-recognition.
    _ = retry_tag

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _vl_model()
    rag_context = _build_travel_rag_context(raw_text)
    policy_block = f"\nPolicy snippets:\n{rag_context}\n" if rag_context else ""
    rule_hint_text = rule_hint if rule_hint in TRAVEL_DOC_TYPES else "unknown"
    prompt = (
        "你是差旅报销材料分类助手。必须仅根据材料内容分类，不能根据文件名猜测。\n"
        "允许的 doc_type 只有：transport_ticket, transport_payment, flight_detail, hotel_invoice, hotel_payment, hotel_order, unknown。\n"
        "判定规则（严格按内容）：\n"
        "1) 支付凭证常见关键词：账单详情、交易成功、支付时间、付款方式、商品说明、账单分类。\n"
        "2) 机票明细常见关键词：价格明细、机建、燃油、票价、普通成人、退改签。\n"
        "3) 机票发票常见关键词：电子发票、经纪代理服务*代订机票费、发票号码、购买方、销售方、价税合计。\n"
        "4) 酒店订单明细常见关键词：携程旅行/飞猪旅行、订单号、几晚明细、费用明细、取消政策、在线付。\n"
        "5) 酒店发票常见关键词：电子发票、住宿/房费、入住/离店、价税合计。\n"
        "6) 若同一图有“在线付+几晚明细+费用明细”，优先判为 hotel_order，不要判成 hotel_payment。\n"
        "7) 若出现“价格明细/普通成人/机票/燃油/机建”这类票价拆分信息，应优先判为 flight_detail；"
        "即使页面也出现订单号或行程信息，也不要判成 transport_payment。\n"
        "8) 若识别到“电子发票/发票号码/购买方/销售方/价税合计”中任意2项及以上，"
        "必须判为 transport_ticket 或 hotel_invoice，禁止判为 payment 类型。\n"
        "9) 只有出现至少两项支付凭证核心词（账单详情、交易成功、支付时间、付款方式、交易号）时，"
        "才允许判为 transport_payment/hotel_payment。\n"
        "输出必须是单个 JSON 对象，不要任何额外文本，格式如下：\n"
        '{"doc_type":"transport_payment","confidence":0.88,"amount":"2360.00","date":"2026-03-13","evidence":"命中关键词: 账单详情,交易成功,商品说明","ocr_text":"...","keywords":["账单详情","交易成功","商品说明"]}\n'
        "约束：\n"
        "- confidence 取值 0~1。\n"
        "- amount 为主要金额（支付金额或发票价税合计）；无法识别填 null。\n"
        "- date 用 YYYY-MM-DD；无法识别填 null。\n"
        "- evidence 不要写泛化词（如 short reason / unknown）。\n"
        f"规则引擎提示（content-only）: {rule_hint_text}\n"
        f"{policy_block}"
    )

    content = ""
    text = (raw_text or "").strip()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    try:
        if text:
            payload = {
                "model": model,
                "stream": False,
                "prompt": f"{prompt}\nDocument text:\n{text[:12000]}",
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
        elif suffix in image_suffixes:
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 45))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        else:
            return None
    except Exception:
        # Fallback path for Ollama versions/models where /api/chat or /api/generate behavior varies.
        try:
            if text:
                payload = {
                    "model": model,
                    "stream": False,
                    "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:12000]}"}],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = (resp.json().get("message") or {}).get("content", "")
            elif suffix in image_suffixes:
                encoded = base64.b64encode(file_bytes).decode("utf-8")
                payload = {
                    "model": model,
                    "stream": False,
                    "prompt": prompt,
                    "images": [encoded],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = resp.json().get("response", "")
            else:
                return None
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None

    doc_type = str(parsed.get("doc_type") or "unknown").strip()
    if doc_type not in TRAVEL_DOC_TYPES:
        doc_type = "unknown"
    amount = _normalize_payment_amount(parsed.get("amount"))
    date_value = str(parsed.get("date") or "").strip()
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if _is_generic_evidence(evidence):
        evidence = ""
    if len(evidence) > 220:
        evidence = evidence[:217] + "..."

    ocr_text = str(parsed.get("ocr_text") or parsed.get("text") or parsed.get("recognized_text") or "").strip()
    if len(ocr_text) > 1200:
        ocr_text = ocr_text[:1200]

    keywords_text = ""
    keywords = parsed.get("keywords")
    if isinstance(keywords, list):
        tokens = [str(item).strip() for item in keywords if str(item).strip()]
        keywords_text = " ".join(tokens[:20])

    keyword_fallback_text = "\n".join(part for part in [ocr_text, keywords_text, evidence] if part)
    keyword_guess = _infer_doc_type_from_visual_keywords(keyword_fallback_text)
    if doc_type == "unknown" and keyword_guess != "unknown":
        doc_type = keyword_guess
        if confidence is None:
            confidence = 0.72
        if not evidence:
            evidence = f"关键词兜底命中: {keyword_guess}"
        else:
            evidence = f"{evidence}; 关键词兜底: {keyword_guess}"

    if len(evidence) > 220:
        evidence = evidence[:217] + "..."
    if doc_type == "unknown":
        confidence = None
    return {
        "doc_type": doc_type,
        "amount": amount,
        "date": date_value,
        "confidence": confidence,
        "evidence": evidence,
        "ocr_text": ocr_text,
        "file_name": file_name,
    }


def _recognize_travel_file(uploaded_file: Any, index: int, retry_tag: str = "") -> dict[str, Any]:
    file_name = str(getattr(uploaded_file, "name", ""))
    suffix = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    file_sha1 = _sha1_of_bytes(file_bytes)

    raw_text = ""
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
        except Exception:
            raw_text = ""

    # Deterministic guess from content only (do not use file name for classification).
    rule_guess = _guess_travel_doc_type("", raw_text)
    invoice_field_guess = _infer_doc_type_from_invoice_fields(raw_text)
    if invoice_field_guess != "unknown":
        rule_guess = invoice_field_guess

    llm_result = _classify_travel_doc_with_ollama(
        file_bytes=file_bytes,
        suffix=suffix,
        file_name=file_name,
        raw_text=raw_text,
        rule_hint=rule_guess,
        retry_tag=retry_tag,
    )
    llm_ocr_text = str((llm_result or {}).get("ocr_text") or "").strip()
    if not llm_ocr_text and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        llm_ocr_text = _extract_image_text_with_ollama(file_bytes)
    if llm_ocr_text:
        ocr_rule_guess = _guess_travel_doc_type("", llm_ocr_text)
        if rule_guess == "unknown" and ocr_rule_guess != "unknown":
            rule_guess = ocr_rule_guess
        if rule_guess == "unknown":
            ocr_invoice_guess = _infer_doc_type_from_invoice_fields(llm_ocr_text)
            if ocr_invoice_guess != "unknown":
                rule_guess = ocr_invoice_guess

    llm_guess = str((llm_result or {}).get("doc_type") or "unknown")
    llm_confidence = _normalize_confidence((llm_result or {}).get("confidence"))
    merged_signal_text = "\n".join(part for part in [raw_text, llm_ocr_text] if part)
    guessed, source = _resolve_travel_doc_type(rule_guess, llm_guess, llm_confidence, merged_signal_text)
    learned_doc_type, learned_source = _lookup_learned_doc_type_override(
        file_sha1,
        file_name,
        merged_signal_text,
        guessed,
    )
    if learned_doc_type and learned_doc_type in TRAVEL_DOC_TYPES and learned_doc_type != guessed:
        guessed = learned_doc_type
        source = learned_source or source

    amount: float | None = None
    llm_amount = _normalize_payment_amount((llm_result or {}).get("amount"))
    if guessed in {"transport_ticket", "hotel_invoice"}:
        invoice_total_from_text = _extract_invoice_total_with_tax_from_text(raw_text) or _extract_invoice_total_with_tax_from_text(
            llm_ocr_text
        )
        if invoice_total_from_text is not None:
            amount = invoice_total_from_text
        # For invoices, reimbursement should use tax-included total from invoice text first.
        if amount is None:
            amount = _auto_extract_amount_from_ticket(uploaded_file)
        if amount is None and llm_amount is not None:
            amount = llm_amount
    elif guessed in {"transport_payment", "hotel_payment"}:
        if llm_amount is not None:
            amount = llm_amount
        else:
            amount = _auto_extract_amount_from_payment(uploaded_file)
    elif llm_amount is not None:
        amount = llm_amount

    date_obj = _pick_primary_date(file_name, merged_signal_text)
    if date_obj is None:
        llm_date = str((llm_result or {}).get("date") or "").strip()
        candidate_dates = _extract_candidate_dates(llm_date)
        if candidate_dates:
            date_obj = candidate_dates[0]

    evidence = str((llm_result or {}).get("evidence") or "").strip()

    return {
        "profile_id": f"{index}:{file_name}:{getattr(uploaded_file, 'size', '')}",
        "index": index,
        "file": uploaded_file,
        "name": file_name,
        "suffix": suffix,
        "doc_type": guessed,
        "amount": amount,
        "date_obj": date_obj,
        "date": date_obj.strftime("%Y-%m-%d") if date_obj else "",
        "slot": "unknown",
        "source": source,
        "confidence": llm_confidence,
        "evidence": evidence,
        "file_sha1": file_sha1,
        "raw_text": raw_text[:3000] if raw_text else "",
        "ocr_text": llm_ocr_text[:3000] if llm_ocr_text else "",
        "signal_text": merged_signal_text[:3500] if merged_signal_text else "",
    }


def _build_travel_file_profile(uploaded_file: Any, index: int) -> dict[str, Any]:
    return _recognize_travel_file(uploaded_file, index=index, retry_tag="")


def _sum_profile_amount(profiles: list[dict[str, Any]]) -> float | None:
    return travel_usecase.sum_profile_amount(profiles)


def _split_profiles_to_go_return(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return travel_usecase.split_profiles_to_go_return(profiles)


def _split_payment_profiles_to_go_return(
    payments: list[dict[str, Any]],
    go_target: float | None,
    return_target: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return travel_usecase.split_payment_profiles_to_go_return(payments, go_target, return_target)


def _build_assignment_from_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return travel_usecase.build_assignment_from_profiles(profiles)


def _organize_travel_materials(
    pool_files: list[Any],
    manual_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return travel_usecase.organize_materials(
        pool_files,
        build_profile=_build_travel_file_profile,
        manual_overrides=manual_overrides,
        apply_overrides=_apply_manual_overrides_to_profiles,
    )


def _slot_label(slot: str) -> str:
    mapping = {
        "go_ticket": "去程机票发票/票据",
        "go_payment": "去程支付记录",
        "go_detail": "去程机票明细",
        "return_ticket": "返程机票发票/票据",
        "return_payment": "返程支付记录",
        "return_detail": "返程机票明细",
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "unknown": "未识别",
    }
    return mapping.get(slot, slot)


def _doc_type_label(doc_type: str) -> str:
    mapping = {
        "transport_ticket": "交通票据",
        "transport_payment": "交通支付记录",
        "flight_detail": "机票明细",
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "unknown": "未知",
    }
    return mapping.get(doc_type, doc_type)


def _build_travel_agent_status(assignment: dict[str, Any]) -> dict[str, Any]:
    return travel_usecase.build_travel_agent_status(assignment)


def _travel_scope_name() -> str:
    return "travel_agent"


def _travel_undo_stack_key() -> str:
    return "travel_agent_undo_stack"


def _clone_travel_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "profile_id": profile.get("profile_id"),
        "index": profile.get("index"),
        "file": profile.get("file"),
        "name": profile.get("name"),
        "suffix": profile.get("suffix"),
        "doc_type": profile.get("doc_type"),
        "amount": profile.get("amount"),
        "date_obj": profile.get("date_obj"),
        "date": profile.get("date"),
        "slot": profile.get("slot"),
        "source": profile.get("source"),
        "confidence": profile.get("confidence"),
        "evidence": profile.get("evidence"),
        "file_sha1": profile.get("file_sha1"),
        "raw_text": profile.get("raw_text"),
        "ocr_text": profile.get("ocr_text"),
        "signal_text": profile.get("signal_text"),
    }


def _clone_travel_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in dict(assignment or {}).items():
        if isinstance(value, list):
            output[key] = list(value)
        else:
            output[key] = value
    return output


def _travel_push_undo_snapshot(
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str] | None = None,
) -> None:
    stack = st.session_state.setdefault(_travel_undo_stack_key(), [])
    if not isinstance(stack, list):
        stack = []
    stack.append(
        {
            "assignment": _clone_travel_assignment(assignment),
            "profiles": [_clone_travel_profile(p) for p in profiles],
            "manual_overrides": dict(manual_overrides or {}),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if len(stack) > 20:
        stack = stack[-20:]
    st.session_state[_travel_undo_stack_key()] = stack


def _travel_pop_undo_snapshot() -> dict[str, Any] | None:
    stack = st.session_state.get(_travel_undo_stack_key())
    if not isinstance(stack, list) or not stack:
        return None
    snapshot = stack.pop()
    st.session_state[_travel_undo_stack_key()] = stack
    return dict(snapshot) if isinstance(snapshot, dict) else None


def _travel_restore_undo_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    assignment = _clone_travel_assignment(dict(snapshot.get("assignment") or {}))
    profiles = [_clone_travel_profile(p) for p in list(snapshot.get("profiles") or []) if isinstance(p, dict)]
    manual_overrides = dict(snapshot.get("manual_overrides") or {})
    return assignment, profiles, manual_overrides


def _travel_build_pending_action_from_text(user_text: str) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    if any(token in text for token in ["应用全部修正", "应用全部建议", "覆盖当前分配结果"]):
        return _append_pending_action(
            _travel_scope_name(),
            action_type="travel_apply_all",
            summary="批量应用当前差旅整理建议",
            target="当前全部待确认建议",
            risk_level="high",
            payload={"command": text},
        )
    if any(token in text for token in ["重新归并", "重新分配", "重排分组", "同一趟"]):
        return _append_pending_action(
            _travel_scope_name(),
            action_type="travel_reorganize",
            summary="重新归并差旅材料并刷新槽位分配",
            target="全部已上传材料",
            risk_level="medium",
            payload={"command": text},
        )
    if any(token in text for token in ["导出报销表", "导出结果", "导出压缩包"]):
        return _append_pending_action(
            _travel_scope_name(),
            action_type="travel_export",
            summary="确认导出当前差旅材料压缩包",
            target="差旅导出",
            risk_level="high",
            payload={"command": text},
        )
    return _append_pending_action(
        _travel_scope_name(),
        action_type="travel_manual_confirm",
        summary="执行一条需要确认的差旅调整",
        target=text[:80],
        risk_level="medium",
        payload={"command": text},
    )


def _travel_execute_pending_action(
    action: dict[str, Any],
    pool_list: list[Any],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]]]:
    action_type = str(action.get("action_type") or "")
    if action_type == "travel_reorganize":
        new_assignment, new_profiles = _organize_travel_materials(pool_list, manual_overrides=manual_overrides)
        return True, "已完成重新归并，并刷新去程/返程/酒店分配。", new_assignment, new_profiles
    if action_type == "travel_export":
        st.session_state["travel_export_confirmed_from_chat"] = True
        return True, "已确认导出。请在下方“差旅材料打包导出”点击导出按钮。", assignment, profiles
    if action_type == "travel_apply_all":
        # 当前版本批量建议主要是确认执行入口，实际变更由具体动作触发。
        return True, "已确认批量应用请求。若有待确认分类动作，会逐条执行。", assignment, profiles
    if action_type == "travel_manual_confirm":
        command = str((action.get("payload") or {}).get("command") or action.get("target") or "").strip()
        if not command:
            return False, "待确认动作缺少可执行内容。", assignment, profiles

        recheck_count, _, recheck_error = _apply_reclassify_from_user_text(command, profiles, manual_overrides=manual_overrides)
        if recheck_error:
            return False, recheck_error, assignment, profiles
        if recheck_count > 0:
            new_assignment = _build_assignment_from_profiles(profiles)
            return True, f"已按确认内容重新识别 {recheck_count} 份材料。", new_assignment, profiles

        changed_count, _, _ = _apply_manual_relabel_from_user_text(command, profiles)
        if changed_count > 0:
            _remember_manual_overrides(manual_overrides, profiles)
            new_assignment = _build_assignment_from_profiles(profiles)
            return True, f"已按确认内容调整 {changed_count} 份材料分类。", new_assignment, profiles
        return False, "确认后未命中可执行变更，请补充更具体的目标。", assignment, profiles

    return False, f"暂不支持的动作类型：{action_type}", assignment, profiles


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    return travel_usecase.merge_uploaded_lists(first, second)


def _target_doc_type_from_user_text(user_text: str, file_name: str) -> str | None:
    text = (user_text or "").lower()
    name = (file_name or "").lower()
    merged = f"{text} {name}"

    if any(key in merged for key in ["订单截图", "酒店订单", "订单图", "hotel order"]):
        return "hotel_order"
    if any(
        key in merged
        for key in ["机票明细", "去程明细", "返程明细", "票价明细", "价格明细", "行程单", "客票行程", "itinerary", "detail"]
    ):
        return "flight_detail"
    if any(key in merged for key in ["酒店支付", "酒店支付记录", "酒店支付凭证"]):
        return "hotel_payment"
    if any(key in merged for key in ["交通支付", "机票支付", "高铁支付", "去程支付记录", "返程支付记录", "支付记录", "支付凭证"]):
        if any(key in merged for key in ["酒店", "住宿"]):
            return "hotel_payment"
        return "transport_payment"
    if any(key in merged for key in ["酒店发票", "住宿发票"]):
        return "hotel_invoice"
    if any(key in merged for key in ["机票发票", "高铁报销凭证", "交通发票", "交通票据"]):
        return "transport_ticket"
    if any(key in merged for key in ["票据"]):
        if any(key in merged for key in ["酒店", "住宿"]):
            return "hotel_invoice"
        return "transport_ticket"
    if "发票" in merged:
        if any(key in merged for key in ["酒店", "住宿"]):
            return "hotel_invoice"
        return "transport_ticket"
    return None


def _match_profiles_by_user_text(user_text: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return []

    sorted_profiles = sorted(profiles, key=lambda p: len(str(p.get("name") or "")), reverse=True)
    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for profile in sorted_profiles:
        name = str(profile.get("name") or "")
        if not name or name not in text:
            continue
        pid = str(profile.get("profile_id") or "")
        if pid and pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        matched.append(profile)
    return matched


def _extract_amount_set_value_from_text(user_text: str) -> float | None:
    text = str(user_text or "").strip()
    if not text:
        return None

    pattern = (
        r"(?:金额|总价|价税合计|含税|小写|支付金额)[^\d¥￥\-]{0,12}"
        r"(?:改为|改成|是|为|设为|写成|填成|调整为|=)\s*"
        r"([¥￥]?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_payment_amount(match.group(1))

    if any(token in text for token in ["金额", "总价", "价税合计", "含税"]):
        tail_match = re.search(
            r"(?:改为|改成|设为|写成|填成|调整为|=)\s*"
            r"([¥￥]?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*(?:元)?\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if tail_match:
            return _normalize_payment_amount(tail_match.group(1))

    return None


def _apply_manual_amount_from_user_text(
    user_text: str,
    profiles: list[dict[str, Any]],
) -> tuple[int, list[str], float | None, str | None]:
    text = str(user_text or "").strip()
    if not text:
        return 0, [], None, None

    target_amount = _extract_amount_set_value_from_text(text)
    if target_amount is None:
        return 0, [], None, None

    matched_profiles = _match_profiles_by_user_text(text, profiles)
    if not matched_profiles and len(profiles) == 1:
        matched_profiles = list(profiles)

    if not matched_profiles:
        return 0, [], target_amount, "没有匹配到要改金额的文件名。请写成：`某文件.pdf 金额是 746`。"

    changed_names: list[str] = []
    for profile in matched_profiles:
        current_amount = _normalize_payment_amount(profile.get("amount"))
        if current_amount is not None and abs(current_amount - target_amount) <= 0.01:
            continue
        profile["amount"] = float(target_amount)
        profile["source"] = "manual_chat_amount"
        profile["confidence"] = 1.0
        profile["evidence"] = f"人工对话改金额 -> {_format_amount(target_amount)}"
        changed_names.append(str(profile.get("name") or ""))

    return len(changed_names), changed_names, target_amount, None


def _parse_relabel_count_hint(user_text: str, max_count: int) -> int:
    if max_count <= 0:
        return 0
    text = (user_text or "").strip()
    if not text:
        return 0

    if any(key in text for key in ["全部", "所有", "全都", "都改", "都归类", "都算", "全部未知", "所有未知"]):
        return max_count

    digit_match = re.search(r"(\d{1,3})\s*(?:个|份|张|条|项)?", text)
    if digit_match:
        try:
            value = int(digit_match.group(1))
            if value > 0:
                return min(max_count, value)
        except ValueError:
            pass

    for zh, value in CHINESE_NUM_MAP.items():
        if any(token in text for token in [f"这{zh}", f"{zh}个", f"{zh}份", f"{zh}张", f"{zh}条", f"{zh}项"]):
            return min(max_count, value)

    return max_count


def _apply_manual_relabel_from_user_text(user_text: str, profiles: list[dict[str, Any]]) -> tuple[int, list[str], str | None]:
    text = (user_text or "").strip()
    if not text:
        return 0, [], None

    target = _target_doc_type_from_user_text(text, "")

    matched_profiles = []
    for profile in profiles:
        name = str(profile.get("name") or "")
        if name and name in text:
            matched_profiles.append(profile)
            if target is None:
                target = _target_doc_type_from_user_text(text, name)

    if not matched_profiles and target and any(key in text for key in ["未知", "未识别", "unknown"]):
        unknown_profiles = [p for p in profiles if str(p.get("doc_type") or "") == "unknown"]
        if unknown_profiles:
            count = _parse_relabel_count_hint(text, len(unknown_profiles))
            matched_profiles = unknown_profiles[:count]

    if not matched_profiles or not target:
        return 0, [], None

    if target not in TRAVEL_DOC_TYPES:
        return 0, [], None

    changed_names: list[str] = []
    for profile in matched_profiles:
        if str(profile.get("doc_type") or "") == target:
            continue
        profile["doc_type"] = target
        profile["source"] = "manual_chat"
        changed_names.append(str(profile.get("name") or ""))

    return len(changed_names), changed_names, target


def _is_reclassify_command(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    markers = ["重新识别", "再识别", "重识别", "重新分类", "再分类", "重新判定", "重跑识别"]
    return any(marker in text for marker in markers)


def _extract_target_profiles_for_reclassify(user_text: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = (user_text or "").strip()
    if not text:
        return []

    lower_text = text.lower()
    sorted_profiles = sorted(profiles, key=lambda p: len(str(p.get("name") or "")), reverse=True)
    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_profile(profile: dict[str, Any]) -> None:
        pid = str(profile.get("profile_id") or "")
        if not pid or pid in seen_ids:
            return
        seen_ids.add(pid)
        matched.append(profile)

    for profile in sorted_profiles:
        name = str(profile.get("name") or "")
        if not name:
            continue
        if name.lower() in lower_text:
            _add_profile(profile)

    tail_match = re.search(r"(?:重新识别|再识别|重识别|重新分类|再分类|重新判定|重跑识别)\s*[:：]?\s*(.+)$", text, flags=re.IGNORECASE)
    if tail_match:
        tail = tail_match.group(1).strip().strip("`\"'“”")
        if tail:
            for profile in sorted_profiles:
                name = str(profile.get("name") or "")
                if not name:
                    continue
                if tail in name or name in tail:
                    _add_profile(profile)

    if not matched and any(key in lower_text for key in ["未知", "未识别", "unknown"]):
        for profile in profiles:
            if str(profile.get("doc_type") or "") == "unknown":
                _add_profile(profile)

    return matched


def _rerun_profile_recognition(profile: dict[str, Any], retry_tag: str) -> dict[str, Any]:
    uploaded_file = profile.get("file")
    if uploaded_file is None:
        return profile
    index = int(profile.get("index") or 0)
    return _recognize_travel_file(uploaded_file, index=index, retry_tag=retry_tag)


def _apply_reclassify_from_user_text(
    user_text: str,
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str] | None = None,
) -> tuple[int, list[str], str | None]:
    if not _is_reclassify_command(user_text):
        return 0, [], None

    targets = _extract_target_profiles_for_reclassify(user_text, profiles)
    if not targets:
        return 0, [], "没有匹配到可重新识别的文件名。请直接写完整文件名，例如：`重新识别 2360元_支付凭证长春到上海机票4人.jpg`。"

    detail_lines: list[str] = []
    total = len(targets)
    for idx, profile in enumerate(targets, start=1):
        if manual_overrides is not None:
            _remove_manual_override_for_profile(manual_overrides, profile)
        name = str(profile.get("name") or "")
        before_doc = str(profile.get("doc_type") or "unknown")
        before_label = _doc_type_label(before_doc)
        refreshed = _rerun_profile_recognition(
            profile,
            retry_tag=f"manual_recheck_{datetime.now().timestamp()}_{idx}",
        )

        # In-place overwrite to keep session list object stable.
        for key in [
            "profile_id",
            "index",
            "file",
            "name",
            "suffix",
            "doc_type",
            "amount",
            "date_obj",
            "date",
            "slot",
            "source",
            "confidence",
            "evidence",
            "file_sha1",
            "raw_text",
            "ocr_text",
            "signal_text",
        ]:
            profile[key] = refreshed.get(key)

        after_doc = str(profile.get("doc_type") or "unknown")
        after_label = _doc_type_label(after_doc)
        source = str(profile.get("source") or "")
        detail_lines.append(f"{name}: {before_label} -> {after_label}（来源：{source}）")

    return total, detail_lines, None


def _generate_travel_agent_reply_rule(
    user_text: str,
    assignment: dict[str, Any],
    status: dict[str, Any],
    profiles: list[dict[str, Any]] | None = None,
) -> str:
    text = (user_text or "").strip()
    if not text:
        return "请继续描述你的问题，比如“我还缺什么”或“有哪些金额不一致”。"
    lower = text.lower()

    if lower in {"你好", "您好", "hi", "hello"}:
        return "你好。你可以直接问：还缺什么、哪里金额不一致、每个文件分到了哪里。"

    if any(key in text for key in ["缺", "补", "全不全", "齐不齐", "还差"]):
        if status["missing"]:
            return "当前还缺这些材料：\n- " + "\n- ".join(status["missing"])
        return "当前必需材料已齐全。"

    if any(key in text for key in ["问题", "风险", "不一致", "金额"]):
        if status["issues"]:
            return "当前发现的问题：\n- " + "\n- ".join(status["issues"])
        return "目前未发现金额不一致问题。"

    if any(key in text for key in ["分配", "归类", "对应", "怎么放", "分到", "放到"]):
        lines = []
        for slot in [
            "go_ticket",
            "go_payment",
            "go_detail",
            "return_ticket",
            "return_payment",
            "return_detail",
            "hotel_invoice",
            "hotel_payment",
            "hotel_order",
            "unknown",
        ]:
            files = _as_uploaded_list(assignment.get(slot))
            if files:
                names = "、".join(str(getattr(f, "name", "")) for f in files[:5])
                if len(files) > 5:
                    names += f" 等{len(files)}份"
                lines.append(f"- {_slot_label(slot)}：{names}")
        if not lines:
            return "当前还没有可分配的材料。"
        return "我目前的分配结果如下：\n" + "\n".join(lines)

    if any(key in text for key in ["为什么", "原因"]) and profiles:
        matched = []
        for profile in profiles:
            name = str(profile.get("name") or "")
            if name and name in text:
                matched.append(profile)
        if matched:
            lines = []
            for profile in matched[:8]:
                lines.append(
                    f"- {profile.get('name')} -> {_doc_type_label(str(profile.get('doc_type') or 'unknown'))}"
                    f"（来源：{profile.get('source') or 'unknown'}）"
                )
            return "当前文件识别说明：\n" + "\n".join(lines)

    if any(key in text for key in ["导出", "打包", "压缩包"]):
        return "可以直接在下方“差旅材料打包导出”输入压缩包名称，然后点击“导出差旅材料压缩包”。"

    summary = f"我这边已整理 {len(profiles or [])} 份材料。"
    if status["missing"]:
        summary += f" 还缺 {len(status['missing'])} 项。"
    if status["issues"]:
        summary += f" 发现 {len(status['issues'])} 个金额核对问题。"
    if not status["missing"] and not status["issues"]:
        summary += " 当前材料已基本齐全。"
    summary += " 你可以继续点名文件告诉我“它是什么类型”，我会直接改。"
    return summary


def _short_join_items(items: list[str], limit: int = 3, empty_text: str = "无") -> str:
    cleaned = [str(item).strip() for item in list(items or []) if str(item).strip()]
    if not cleaned:
        return empty_text
    head = cleaned[:limit]
    text = "；".join(head)
    if len(cleaned) > limit:
        text += f"（另有{len(cleaned) - limit}项）"
    return text


def _build_travel_handoff_status_reply(
    *,
    profiles: list[dict[str, Any]],
    status: dict[str, Any],
    guide_files: list[Any],
) -> str:
    type_counts: dict[str, int] = {}
    for profile in profiles:
        label = _doc_type_label(str(profile.get("doc_type") or "unknown"))
        type_counts[label] = type_counts.get(label, 0) + 1
    type_text = (
        "、".join(f"{name}{count}份" for name, count in sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))[:5])
        if type_counts
        else "暂未识别到明确类型"
    )
    missing_text = _short_join_items(list(status.get("missing") or []), limit=3, empty_text="暂无明显缺件")
    issue_text = _short_join_items(list(status.get("issues") or []), limit=2, empty_text="暂无金额核对异常")
    return _compose_three_stage_reply(
        f"已接收首页带入的 {len(_as_uploaded_list(guide_files))} 份材料，并完成当前批次差旅识别。",
        f"当前识别概览：{type_text}。缺件：{missing_text}。金额核对：{issue_text}。",
        "你可以继续说“现在还缺什么”或“哪个和哪个对不上”，我会按去程/返程/酒店逐项解释。",
    )


def _build_travel_agent_context_text(assignment: dict[str, Any], status: dict[str, Any], profiles: list[dict[str, Any]]) -> str:
    slot_keys = [
        "go_ticket",
        "go_payment",
        "go_detail",
        "return_ticket",
        "return_payment",
        "return_detail",
        "hotel_invoice",
        "hotel_payment",
        "hotel_order",
        "unknown",
    ]
    slot_summary = {}
    for key in slot_keys:
        files = _as_uploaded_list(assignment.get(key))
        slot_summary[key] = [str(getattr(f, "name", "")) for f in files[:20]]

    profile_summary = []
    for profile in profiles[:80]:
        profile_summary.append(
            {
                "name": profile.get("name"),
                "doc_type": profile.get("doc_type"),
                "slot": profile.get("slot"),
                "amount": _format_amount(_safe_float(profile.get("amount"))),
                "date": profile.get("date"),
                "source": profile.get("source"),
                "confidence": _normalize_confidence(profile.get("confidence")),
                "evidence": str(profile.get("evidence") or ""),
            }
        )

    context = {
        "missing": status.get("missing", []),
        "issues": status.get("issues", []),
        "tips": status.get("tips", []),
        "slot_summary": slot_summary,
        "amounts": {
            "go_ticket_amount": assignment.get("go_ticket_amount"),
            "go_payment_amount": assignment.get("go_payment_amount"),
            "return_ticket_amount": assignment.get("return_ticket_amount"),
            "return_payment_amount": assignment.get("return_payment_amount"),
            "hotel_invoice_amount": assignment.get("hotel_invoice_amount"),
            "hotel_payment_amount": assignment.get("hotel_payment_amount"),
        },
        "profiles": profile_summary,
    }
    return json.dumps(context, ensure_ascii=False)


def _generate_travel_agent_reply_llm(
    user_text: str,
    assignment: dict[str, Any],
    status: dict[str, Any],
    profiles: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()

    system_prompt = (
        "你是差旅报销Agent。基于给定上下文回答用户问题，禁止编造未提供的信息。"
        "优先回答：缺件、金额核对问题、分配结果、下一步补件建议。"
        "可以引用制度片段或历史样例，但只能使用已提供RAG上下文。"
        "不要重复欢迎语，不要复述“已进入模式”等固定句。"
        "输出中文，简洁，用要点列表。"
    )
    context_text = _build_travel_agent_context_text(assignment, status, profiles)
    rag_context = ""
    try:
        rag_query = f"{user_text}\n{context_text[:1200]}"
        rag_context = travel_usecase.build_travel_policy_context(rag_query, top_k=3)
    except Exception:
        rag_context = ""

    llm_messages = [{"role": "system", "content": system_prompt}]
    # Keep short recent history to retain context while controlling token size.
    for item in messages[-6:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        llm_messages.append({"role": role, "content": content[:1200]})
    llm_messages.append(
        {
            "role": "user",
            "content": (
                f"当前材料上下文(JSON)：\n{context_text}\n\n"
                f"RAG上下文：\n{rag_context or '无'}\n\n"
                f"用户问题：{user_text}"
            ),
        }
    )

    invalid_markers = ["我已进入差旅材料整理模式", "可以问我", "还缺什么、哪里金额不一致"]

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
        if content and not any(marker in content for marker in invalid_markers):
            return content
    except Exception:
        pass

    # Fallback for Ollama versions/models that don't support /api/chat reliably.
    history_lines = []
    for item in llm_messages:
        role = item.get("role")
        content = str(item.get("content") or "")
        if not content:
            continue
        history_lines.append(f"[{role}] {content}")
    prompt = "\n\n".join(history_lines[-8:])

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
        if content and not any(marker in content for marker in invalid_markers):
            return content
    except Exception:
        return None

    return None


def _render_travel_conversation_agent() -> dict[str, Any]:
    st.subheader("会话式差旅 Agent")
    st.caption("把本次差旅材料一次性上传，Agent 会自动归类、提示缺件，并支持继续对话追问。")
    st.caption("当前模式：材料识别=内容优先（LLM + Rule Guard）；对话=本地LLM（Ollama）")

    pool_files = st.file_uploader(
        "上传本次差旅全部材料（PDF/图片，可多选）",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key="travel_agent_pool_files",
    )
    page_uploaded_files = _as_uploaded_list(pool_files)
    pool_list = list(page_uploaded_files)
    guide_payload, guide_files = _get_guide_handoff_for_flow("travel")
    if guide_files:
        pool_list = _merge_uploaded_lists(pool_list, _as_uploaded_list(guide_files))
        st.info(f"已从首页引导带入 {len(guide_files)} 份材料，可直接开始整理。")
        _render_included_file_list(
            flow_label="差旅流程",
            page_uploaded_files=page_uploaded_files,
            guide_files=guide_files,
            merged_files=pool_list,
        )
    if guide_payload:
        with st.expander("首页引导摘要（已带入）", expanded=False):
            st.json(guide_payload)
    manual_overrides = st.session_state.setdefault("travel_agent_manual_overrides", {})
    if not isinstance(manual_overrides, dict):
        manual_overrides = {}
        st.session_state["travel_agent_manual_overrides"] = manual_overrides
    _prune_manual_overrides(manual_overrides, pool_list)
    current_signature = _files_signature(pool_list)

    action_left, action_mid, action_right = st.columns(3)
    refresh_clicked = action_left.button("Agent整理材料", use_container_width=True, key="travel_agent_refresh")
    clear_chat_clicked = action_mid.button("清空会话记录", use_container_width=True, key="travel_agent_clear_chat")
    clear_cache_clicked = action_right.button("清空识别缓存", use_container_width=True, key="travel_agent_clear_cache")

    if clear_chat_clicked:
        st.session_state.pop("travel_agent_messages", None)
    if clear_cache_clicked:
        st.cache_data.clear()
        st.session_state.pop("travel_agent_pool_signature", None)
        st.session_state.pop("travel_agent_assignment", None)
        st.session_state.pop("travel_agent_profiles", None)
        st.session_state.pop("travel_agent_manual_overrides", None)
        st.session_state.pop(_travel_undo_stack_key(), None)
        _clear_pending_actions(_travel_scope_name())
        _clear_last_applied_action(_travel_scope_name())
        st.success("识别缓存已清空，请重新点击“Agent整理材料”。")
        st.rerun()

    need_rebuild = refresh_clicked or st.session_state.get("travel_agent_pool_signature") != current_signature
    if need_rebuild:
        with st.spinner("Agent 正在识别并分配材料..."):
            assignment, profiles = _organize_travel_materials(pool_list, manual_overrides=manual_overrides)
        st.session_state["travel_agent_pool_signature"] = current_signature
        st.session_state["travel_agent_assignment"] = assignment
        st.session_state["travel_agent_profiles"] = profiles

    assignment = st.session_state.get("travel_agent_assignment", {})
    profiles = st.session_state.get("travel_agent_profiles", [])
    if pool_list and not assignment:
        assignment, profiles = _organize_travel_materials(pool_list, manual_overrides=manual_overrides)
        st.session_state["travel_agent_pool_signature"] = current_signature
        st.session_state["travel_agent_assignment"] = assignment
        st.session_state["travel_agent_profiles"] = profiles

    messages = st.session_state.setdefault("travel_agent_messages", [])
    if not messages:
        messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "我看到了，你可以把这次差旅材料都交给我。",
                    "我会先按内容做分类和归组，再持续显示缺件、异常与待确认动作。",
                    "你可以先问“现在还缺什么”，也可以说“这张应是机票明细”。",
                ),
            }
        )

    if not pool_list:
        st.info("请先上传差旅材料，我会自动分类到去程/返程/酒店，并告诉你缺什么。")
        return {"missing": [], "issues": [], "tips": [], "complete": False}

    status = _build_travel_agent_status(assignment)
    pending_actions = [item for item in _get_pending_actions(_travel_scope_name()) if str(item.get("status") or "pending") == "pending"]
    if guide_files or guide_payload:
        handoff_token = (
            f"travel|{str(st.session_state.get('guide_handoff_entered_at') or '')}"
            f"|{_files_signature(_as_uploaded_list(guide_files))}|{current_signature}"
        )
        last_token = str(st.session_state.get("travel_agent_handoff_summary_token") or "")
        if handoff_token and handoff_token != last_token:
            messages.append(
                {
                    "role": "assistant",
                    "content": _build_travel_handoff_status_reply(
                        profiles=profiles,
                        status=status,
                        guide_files=guide_files,
                    ),
                }
            )
            st.session_state["travel_agent_handoff_summary_token"] = handoff_token

    st.markdown("### 当前任务状态")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("任务类型", "差旅")
    s2.metric("已识别材料", len(profiles))
    s3.metric("缺件数", len(status.get("missing") or []))
    s4.metric("异常数", len(status.get("issues") or []))
    s5.metric("待确认动作", len(pending_actions))
    s6.metric("状态", "可提交" if status.get("complete") else "待补充")

    with st.expander("查看分类与归组摘要", expanded=False):
        type_counts: dict[str, int] = {}
        slot_counts: dict[str, int] = {}
        for profile in profiles:
            type_key = _doc_type_label(str(profile.get("doc_type") or "unknown"))
            slot_key = _slot_label(str(profile.get("slot") or "unknown"))
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            slot_counts[slot_key] = slot_counts.get(slot_key, 0) + 1
        st.markdown("**当前分类结果**")
        st.dataframe(
            [{"类别": key, "数量": value} for key, value in sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**当前归组结果**")
        st.dataframe(
            [{"槽位": key, "数量": value} for key, value in sorted(slot_counts.items(), key=lambda x: (-x[1], x[0]))],
            hide_index=True,
            use_container_width=True,
        )

    if status["missing"]:
        st.warning("仍缺材料：" + "、".join(status["missing"]))
    else:
        st.success("必需材料已齐全。")

    if status["issues"]:
        st.error("发现核对问题：")
        for issue in status["issues"]:
            st.markdown(f"- {issue}")
    elif status["complete"]:
        st.success("金额核对通过，可以导出材料。")

    for tip in status["tips"]:
        st.info(tip)

    profile_rows = []
    for profile in profiles:
        confidence = _normalize_confidence(profile.get("confidence"))
        evidence = str(profile.get("evidence") or "").strip()
        if len(evidence) > 60:
            evidence = evidence[:57] + "..."
        profile_rows.append(
            {
                "文件名": profile.get("name"),
                "识别类型": _doc_type_label(str(profile.get("doc_type") or "unknown")),
                "分配槽位": _slot_label(str(profile.get("slot") or "unknown")),
                "金额": _format_amount(_safe_float(profile.get("amount"))),
                "日期": profile.get("date") or "",
                "识别来源": profile.get("source") or "",
                "置信度": f"{confidence * 100:.0f}%" if confidence is not None else "",
                "识别依据": evidence,
            }
        )
    if profile_rows:
        st.dataframe(profile_rows, use_container_width=True, hide_index=True)
    st.caption("你可以自然表达：比如“这张应该算机票明细”“为什么分到返程支付记录”“还缺什么”。")

    if profiles:
        doc_type_to_label = {
            "transport_ticket": "交通票据",
            "transport_payment": "交通支付记录",
            "flight_detail": "机票明细",
            "hotel_invoice": "酒店发票",
            "hotel_payment": "酒店支付记录",
            "hotel_order": "酒店订单截图",
            "unknown": "未知",
        }
        label_to_doc_type = {v: k for k, v in doc_type_to_label.items()}
        with st.expander("手工修正分类（可选）", expanded=False):
            edit_rows = []
            for profile in profiles:
                edit_rows.append(
                    {
                        "profile_id": profile.get("profile_id"),
                        "文件名": profile.get("name"),
                        "识别类型": doc_type_to_label.get(str(profile.get("doc_type") or "unknown"), "未知"),
                    }
                )
            edited_rows = st.data_editor(
                edit_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "profile_id": st.column_config.TextColumn("profile_id", disabled=True, width="small"),
                    "文件名": st.column_config.TextColumn("文件名", disabled=True, width="large"),
                    "识别类型": st.column_config.SelectboxColumn(
                        "识别类型",
                        options=list(doc_type_to_label.values()),
                        required=True,
                        width="medium",
                    ),
                },
                key="travel_agent_manual_editor",
            )
            if st.button("应用修正并重新分配", key="travel_agent_apply_manual", use_container_width=True):
                snapshot_assignment = _clone_travel_assignment(assignment)
                snapshot_profiles = [_clone_travel_profile(p) for p in profiles]
                snapshot_manual_overrides = dict(manual_overrides)
                id_to_profile = {str(p.get("profile_id")): p for p in profiles}
                changed = 0
                for row in edited_rows:
                    pid = str(row.get("profile_id") or "")
                    label = str(row.get("识别类型") or "")
                    target_doc_type = label_to_doc_type.get(label, "unknown")
                    profile = id_to_profile.get(pid)
                    if not profile:
                        continue
                    if str(profile.get("doc_type") or "") == target_doc_type:
                        continue
                    profile["doc_type"] = target_doc_type
                    profile["source"] = "manual_table"
                    _set_manual_override_for_profile(manual_overrides, profile)
                    changed += 1

                if changed > 0:
                    _travel_push_undo_snapshot(snapshot_assignment, snapshot_profiles, snapshot_manual_overrides)
                    assignment = _build_assignment_from_profiles(profiles)
                    st.session_state["travel_agent_assignment"] = assignment
                    st.session_state["travel_agent_profiles"] = profiles
                    travel_usecase.learn_from_profiles(profiles, assignment, reason="manual_table")
                    _record_last_applied_action(
                        _travel_scope_name(),
                        {
                            "action_id": uuid4().hex,
                            "action_type": "manual_table",
                            "summary": f"手工修正分类 {changed} 条",
                        },
                    )
                    st.success(_compose_three_stage_reply("好，我理解了你的修正。", f"我已经应用了 {changed} 条分类调整。", "如果需要，我可以撤销上一步或继续帮你检查缺件。"))
                    st.rerun()
                else:
                    st.info("没有检测到分类变更。")

    st.markdown("### 待确认修改")
    pending_actions = [item for item in _get_pending_actions(_travel_scope_name()) if str(item.get("status") or "pending") == "pending"]
    pleft, pmid, pright = st.columns(3)
    apply_all = pleft.button("应用全部建议", key="travel_pending_apply_all", use_container_width=True, disabled=not pending_actions)
    undo_last = pmid.button("撤销上一步", key="travel_pending_undo_last", use_container_width=True)
    clear_pending = pright.button("清空待确认", key="travel_pending_clear_all", use_container_width=True, disabled=not pending_actions)

    if clear_pending:
        _clear_pending_actions(_travel_scope_name())
        st.success("已清空待确认修改。")
        st.rerun()

    if undo_last:
        snapshot = _travel_pop_undo_snapshot()
        if not snapshot:
            st.info("当前没有可撤销的上一步。")
        else:
            assignment, profiles, manual_overrides = _travel_restore_undo_snapshot(snapshot)
            st.session_state["travel_agent_assignment"] = assignment
            st.session_state["travel_agent_profiles"] = profiles
            st.session_state["travel_agent_manual_overrides"] = manual_overrides
            _clear_last_applied_action(_travel_scope_name())
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
            st.rerun()

    if pending_actions:
        for action in pending_actions:
            action_id = str(action.get("action_id") or "")
            if not action_id:
                continue
            with st.container():
                c1, c2, c3 = st.columns([7, 1, 1])
                c1.markdown(
                    f"**{action.get('summary') or '待确认动作'}**  \n"
                    f"风险：`{action.get('risk_level') or 'medium'}`｜创建时间：`{action.get('created_at') or '-'}"
                )
                target_key = f"travel_pending_target_{action_id}"
                current_target = str(action.get("target") or "")
                edited_target = c1.text_input("目标值（可改）", value=current_target, key=target_key, label_visibility="collapsed")
                if edited_target != current_target:
                    payload = dict(action.get("payload") or {})
                    if payload.get("command"):
                        payload["command"] = edited_target
                    _update_pending_action(_travel_scope_name(), action_id, {"target": edited_target, "payload": payload})
                    action["target"] = edited_target
                    action["payload"] = payload

                confirm_clicked = c2.button("确认", key=f"travel_pending_confirm_{action_id}", use_container_width=True)
                cancel_clicked = c3.button("取消", key=f"travel_pending_cancel_{action_id}", use_container_width=True)

                if cancel_clicked:
                    _remove_pending_action(_travel_scope_name(), action_id)
                    st.rerun()

                if confirm_clicked:
                    _travel_push_undo_snapshot(assignment, profiles, manual_overrides)
                    ok, msg, assignment, profiles = _travel_execute_pending_action(
                        action,
                        pool_list,
                        assignment,
                        profiles,
                        manual_overrides,
                    )
                    if ok:
                        st.session_state["travel_agent_assignment"] = assignment
                        st.session_state["travel_agent_profiles"] = profiles
                        _record_last_applied_action(_travel_scope_name(), action)
                        _remove_pending_action(_travel_scope_name(), action_id)
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
                        st.rerun()
                    else:
                        st.warning(msg)

    else:
        st.info("当前没有待确认动作。高风险指令会先放在这里，确认后再执行。")

    if apply_all and pending_actions:
        _travel_push_undo_snapshot(assignment, profiles, manual_overrides)
        success_count = 0
        failed_lines: list[str] = []
        for action in list(pending_actions):
            ok, msg, assignment, profiles = _travel_execute_pending_action(action, pool_list, assignment, profiles, manual_overrides)
            if ok:
                success_count += 1
                _record_last_applied_action(_travel_scope_name(), action)
                _remove_pending_action(_travel_scope_name(), str(action.get("action_id") or ""))
            else:
                failed_lines.append(msg)
        st.session_state["travel_agent_assignment"] = assignment
        st.session_state["travel_agent_profiles"] = profiles
        if success_count > 0:
            try:
                travel_usecase.learn_from_profiles(profiles, assignment, reason="pending_apply_all")
            except Exception:
                pass
        summary = f"已批量执行 {success_count} 条待确认动作。"
        if failed_lines:
            summary += "\n" + "\n".join(f"- {line}" for line in failed_lines[:4])
        messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "好的，我已经处理你的批量确认。",
                    summary,
                    "你可以继续微调分类，或直接导出当前结果。",
                ),
            }
        )
        st.rerun()

    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_input = st.chat_input("例如：我现在还缺什么？", key="travel_agent_chat_input")
    if user_input:
        messages.append({"role": "user", "content": user_input})
        intent = classify_user_message_intent(
            user_input,
            {
                "domain": "travel",
                "missing_count": len(status.get("missing") or []),
                "issue_count": len(status.get("issues") or []),
                "pending_count": len(pending_actions),
            },
        )

        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            action = _travel_build_pending_action_from_text(user_input)
            if action:
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "我理解你的意图，这是一个会影响整体结果的操作。",
                            f"我已把它放进“待确认修改”：{action.get('summary') or '待确认动作'}。",
                            "你可以在待确认区逐条确认，或者点“应用全部建议”。",
                        ),
                    }
                )
                st.rerun()

        if intent.intent_type == "light_edit":
            _travel_push_undo_snapshot(assignment, profiles, manual_overrides)
            with st.spinner("正在应用轻量修正..."):
                recheck_count, _, recheck_error = _apply_reclassify_from_user_text(
                    user_input,
                    profiles,
                    manual_overrides=manual_overrides,
                )
            if recheck_error:
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "我看到了你的修正意图。",
                            f"这次还没执行成功：{recheck_error}",
                            "你可以换一个更完整的文件名，或让我先展示当前分配给你确认。",
                        ),
                    }
                )
                st.rerun()

            changed_count, changed_names, target_doc_type = _apply_manual_relabel_from_user_text(user_input, profiles)
            amount_changed_count, amount_changed_names, manual_amount, amount_error = _apply_manual_amount_from_user_text(
                user_input, profiles
            )
            if amount_error:
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "我看到了你在改金额。",
                            amount_error,
                            "你可以直接写“文件名 金额是 746”，我会立刻写回并重新核对金额差异。",
                        ),
                    }
                )
                st.rerun()

            total_changed = int(recheck_count) + int(changed_count) + int(amount_changed_count)
            if total_changed > 0:
                _remember_manual_overrides(manual_overrides, profiles)
                assignment = _build_assignment_from_profiles(profiles)
                status = _build_travel_agent_status(assignment)
                st.session_state["travel_agent_assignment"] = assignment
                st.session_state["travel_agent_profiles"] = profiles
                try:
                    travel_usecase.learn_from_profiles(profiles, assignment, reason="manual_chat")
                except Exception:
                    pass
                changed_preview = "、".join(changed_names[:3]) if changed_names else ""
                if changed_count > 3:
                    changed_preview += f" 等{changed_count}个文件"
                change_text = f"我已完成 {total_changed} 项轻量修正。"
                if changed_preview:
                    change_text += f" 主要调整：{changed_preview}。"
                if target_doc_type:
                    change_text += f" 目标类型：{_doc_type_label(str(target_doc_type))}。"
                if amount_changed_count > 0:
                    amount_preview = "、".join(amount_changed_names[:3]) if amount_changed_names else ""
                    if amount_changed_count > 3:
                        amount_preview += f" 等{amount_changed_count}个文件"
                    change_text += (
                        f" 已把 {amount_changed_count} 份材料金额修正为 {_format_amount(manual_amount)}"
                        + (f"（{amount_preview}）" if amount_preview else "")
                        + "。"
                    )
                _record_last_applied_action(
                    _travel_scope_name(),
                    {
                        "action_id": uuid4().hex,
                        "action_type": "travel_light_edit",
                        "summary": f"轻修正 {total_changed} 项",
                    },
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "明白，这类修正我可以直接处理。",
                            change_text,
                            "如需恢复，我可以撤销刚才的修改；也可以继续帮你检查缺件和金额核对。",
                        ),
                    }
                )
                st.rerun()
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "我理解你的修正意图，但这次没有产生实际变更。"
                            "可能是这些文件当前已经是目标类型。"
                            "你可以让我先列出这些文件的当前类型，我再按你的要求逐条改。"
                        ),
                    }
                )
                st.rerun()

        if intent.intent_type == "ambiguous":
            summary = (
                f"当前已识别 {len(profiles)} 份材料，缺件 {len(status.get('missing') or [])} 项，"
                f"异常 {len(status.get('issues') or [])} 项。"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": _compose_three_stage_reply(
                        "我理解你觉得结果有点不对。",
                        summary,
                        "你可以告诉我具体文件名和目标类型（例如“某某.jpg 是机票明细”），我会先给出调整再请你确认。",
                    ),
                }
            )
            st.rerun()

        reply = _generate_travel_agent_reply_llm(user_input, assignment, status, profiles, messages)
        if not reply:
            reply = _generate_travel_agent_reply_rule(user_input, assignment, status, profiles)
        messages.append(
            {
                "role": "assistant",
                "content": str(reply or "").strip(),
            }
        )
        st.rerun()

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


def _render_travel_package_export() -> None:
    st.subheader("差旅材料打包导出")
    default_name = f"差旅报销材料_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_name = st.text_input(
        "压缩包名称（无需 .zip）",
        value=default_name,
        key="travel_export_package_name",
    )

    assignment = st.session_state.get("travel_agent_assignment") or {}

    go_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_ticket")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_file")),
    )
    go_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_payment")),
        _as_uploaded_list(st.session_state.get("travel_go_payment_file")),
    )
    go_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_detail")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_detail_file")),
    )

    return_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_ticket")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_file")),
    )
    return_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_payment")),
        _as_uploaded_list(st.session_state.get("travel_return_payment_file")),
    )
    return_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_detail")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_detail_file")),
    )

    hotel_invoice_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_invoice")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_invoice")),
    )
    hotel_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_payment")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_payment")),
    )
    hotel_order_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_order")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_order")),
    )

    go_ticket_amount = _safe_float(st.session_state.get("travel_go_ticket_amount"))
    go_payment_amount = _safe_float(st.session_state.get("travel_go_payment_amount"))
    return_ticket_amount = _safe_float(st.session_state.get("travel_return_ticket_amount"))
    return_payment_amount = _safe_float(st.session_state.get("travel_return_payment_amount"))
    hotel_invoice_amount = _safe_float(st.session_state.get("travel_hotel_ticket_amount"))
    hotel_payment_amount = _safe_float(st.session_state.get("travel_hotel_payment_amount"))

    if go_ticket_amount is None:
        go_ticket_amount = _safe_float(assignment.get("go_ticket_amount"))
    if go_payment_amount is None:
        go_payment_amount = _safe_float(assignment.get("go_payment_amount"))
    if return_ticket_amount is None:
        return_ticket_amount = _safe_float(assignment.get("return_ticket_amount"))
    if return_payment_amount is None:
        return_payment_amount = _safe_float(assignment.get("return_payment_amount"))
    if hotel_invoice_amount is None:
        hotel_invoice_amount = _safe_float(assignment.get("hotel_invoice_amount"))
    if hotel_payment_amount is None:
        hotel_payment_amount = _safe_float(assignment.get("hotel_payment_amount"))

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


def _material_build_pending_action_from_text(user_text: str, task, fields: dict[str, Any]) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    if not text:
        return None

    if any(token in text for token in ["导出报销表", "导出结果", "导出excel", "导出 excel"]):
        return _append_pending_action(
            _material_scope_name(task.id),
            action_type="material_export",
            summary="确认导出当前材料费结果",
            target="当前任务导出",
            risk_level="high",
            payload={"command": text},
        )

    if any(token in text for token in ["应用全部修正", "应用全部建议", "覆盖当前结果", "覆盖当前分配结果"]):
        return _append_pending_action(
            _material_scope_name(task.id),
            action_type="material_apply_all",
            summary="批量应用当前材料费待确认建议",
            target="当前任务全部待确认动作",
            risk_level="high",
            payload={"command": text},
        )

    if any(token in text for token in ["智能修复", "重新识别", "应用llm修复表", "应用llm结果", "应用对比结果"]):
        return _append_pending_action(
            _material_scope_name(task.id),
            action_type="material_command",
            summary="执行一条高影响材料费调整",
            target=text[:120],
            risk_level="high",
            payload={"command": text},
        )

    return _append_pending_action(
        _material_scope_name(task.id),
        action_type="material_command",
        summary="执行一条待确认材料费调整",
        target=text[:120],
        risk_level="medium",
        payload={"command": text},
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
                return [item for item in actions if isinstance(item, dict)]
    except Exception:
        return []
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
        "回答简短，使用要点。"
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
            return content
    except Exception:
        return None

    return None


def _render_material_conversation_agent() -> None:
    st.subheader("2) 材料费会话式 Agent")
    st.caption("上传材料费发票后，Agent 自动抽取并生成明细表。默认对话优先，动作会分级处理。")

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

    options = {f"{task.original_filename} | {task.id} | {task.status}": task.id for task in valid_tasks}
    selected_label = st.selectbox("选择当前材料任务", options=list(options.keys()), key="material_agent_selected_task")
    selected_task_id = options[selected_label]
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

    st.markdown("### 当前任务状态")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("任务类型", "材料费")
    m2.metric("明细行数", len(rows))
    m3.metric("发票总金额(含税)", _format_amount(amount_value) if amount_value is not None else "-")
    m4.metric("明细合计", _format_amount(row_total) if row_total is not None else "-")
    m5.metric("待确认动作", len(pending_actions))
    m6.metric("识别模式", str(fields.get("processing_mode") or fields.get("extraction_source") or "default"))

    quality_hints = _material_agent_quality_hints(fields)
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
    if quality_hints:
        st.warning("发现质量风险：")
        for hint in quality_hints:
            st.markdown(f"- {hint}")
    else:
        st.success("当前明细质量检查通过。")

    review_items = list(fields.get("low_confidence_review") or [])
    if review_items:
        st.info(f"人工复核区：{len(review_items)} 条低置信度项。")
        open_key = _material_review_dialog_state_key(task.id)
        if open_key not in st.session_state:
            st.session_state[open_key] = True
        open_dialog = st.button(
            "打开质量风险复核弹窗（双表对比）",
            use_container_width=True,
            key=f"material_review_open_{task.id}",
        )
        if open_dialog:
            st.session_state[open_key] = True
        if bool(st.session_state.get(open_key)):
            _render_material_review_dialog(task.id)

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
        if review_view:
            st.dataframe(review_view, use_container_width=True, hide_index=True)
    else:
        st.session_state.pop(_material_review_dialog_state_key(task.id), None)

    with st.expander("任务工作台：分类与异常摘要", expanded=False):
        slot_like = {
            "项目名称为空": sum(1 for row in rows if not str(row.get("item_name") or "").strip()),
            "规格为空": sum(1 for row in rows if not str(row.get("spec") or "").strip()),
            "数量为空": sum(1 for row in rows if not str(row.get("quantity") or "").strip()),
            "金额为空": sum(1 for row in rows if not str(row.get("line_total_with_tax") or "").strip()),
        }
        st.dataframe(
            [{"项": key, "数量": value} for key, value in slot_like.items()],
            use_container_width=True,
            hide_index=True,
        )
        if quality_hints:
            st.markdown("**异常提示**")
            for hint in quality_hints[:8]:
                st.markdown(f"- {hint}")
        else:
            st.markdown("**异常提示**\n- 当前未发现明显异常。")

    st.dataframe(display_rows_cn, use_container_width=True, hide_index=True)

    st.markdown("### 待确认修改")
    pa1, pa2, pa3 = st.columns(3)
    apply_all_pending = pa1.button("应用全部建议", key=f"material_pending_apply_all_{task.id}", use_container_width=True, disabled=not pending_actions)
    undo_last = pa2.button("撤销上一步", key=f"material_pending_undo_{task.id}", use_container_width=True)
    clear_pending = pa3.button("清空待确认", key=f"material_pending_clear_{task.id}", use_container_width=True, disabled=not pending_actions)

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
        else:
            st.info("当前没有可撤销的上一步。")

    if pending_actions:
        for action in pending_actions:
            action_id = str(action.get("action_id") or "")
            if not action_id:
                continue
            with st.container():
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
                    ok, msg, updated_task, updated_fields = _material_execute_pending_action(action, task, fields)
                    if not ok:
                        st.warning(msg)
                    else:
                        _record_last_applied_action(scope, action)
                        _remove_pending_action(scope, action_id)
                        task = updated_task or task
                        fields = updated_fields or fields
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
            ok, msg, updated_task, updated_fields = _material_execute_pending_action(action, task, fields)
            if ok:
                success_count += 1
                _record_last_applied_action(scope, action)
                _remove_pending_action(scope, str(action.get("action_id") or ""))
                task = updated_task or task
                fields = updated_fields or fields
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
        else:
            st.error("暂未生成可对比的规则表/LLM表。")

    st.markdown("### 工作台操作")
    op1, op2, op3 = st.columns(3)
    if op1.button("应用全部建议（待确认）", key=f"material_apply_all_shortcut_{task.id}", use_container_width=True, disabled=not pending_actions):
        pending_now = [item for item in _get_pending_actions(scope) if str(item.get("status") or "pending") == "pending"]
        success_count = 0
        failed_lines: list[str] = []
        for action in list(pending_now):
            ok, msg, updated_task, updated_fields = _material_execute_pending_action(action, task, fields)
            if ok:
                success_count += 1
                _record_last_applied_action(scope, action)
                _remove_pending_action(scope, str(action.get("action_id") or ""))
                task = updated_task or task
                fields = updated_fields or fields
            else:
                failed_lines.append(msg)
        summary = f"已批量应用 {success_count} 条建议。"
        if failed_lines:
            summary += "\n" + "\n".join(f"- {line}" for line in failed_lines[:5])
        task_messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "好的，我已执行工作台批量应用。",
                    summary,
                    "你可以继续让我解释异常，或直接导出结果。",
                ),
            }
        )
        st.rerun()
    if op2.button("撤销上一步（快捷）", key=f"material_undo_shortcut_{task.id}", use_container_width=True):
        handled, reply, updated_task, updated_fields = _material_agent_apply_chat_command("撤销上一步", task, fields)
        if handled:
            task_messages.append(
                {
                    "role": "assistant",
                    "content": _compose_three_stage_reply(
                        "好的，我收到撤销请求。",
                        reply,
                        "你可以继续指出要修改的行，或让我先解释当前结果。",
                    ),
                }
            )
            task = updated_task or task
            fields = updated_fields or fields
            st.rerun()
        st.info("当前没有可撤销记录。")
    op3.caption("导出请使用下方“下载Excel/下载文本”。")

    _render_export_download(task, key_scope="material_agent")

    with st.expander("查看抽取结果(JSON)", expanded=False):
        st.json(task.extracted_data or {})

    for message in task_messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_input = st.chat_input(
        "直接说你的问题或修改意图（例如：最后一行规格和项目名混了）",
        key=f"material_agent_chat_input_{task.id}",
    )
    if user_input:
        task_messages.append({"role": "user", "content": user_input})
        intent = classify_user_message_intent(
            user_input,
            {
                "domain": "material",
                "line_count": len(rows),
                "pending_count": len(pending_actions),
                "quality_hint_count": len(quality_hints),
            },
        )

        if intent.intent_type == "strong_action" and intent.needs_confirmation:
            action = _material_build_pending_action_from_text(user_input, task, fields)
            if action:
                task_messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "我理解你的操作意图，这一步影响范围较大。",
                            f"我已把它放入待确认区：{action.get('summary') or '待确认动作'}。",
                            "你可以在“待确认修改”里逐条确认，或点“应用全部建议”。",
                        ),
                    }
                )
                st.rerun()

        if intent.intent_type == "light_edit":
            handled, reply, updated_task, updated_fields = _material_agent_apply_chat_command(user_input, task, fields)
            if handled:
                _record_last_applied_action(
                    scope,
                    {
                        "action_id": uuid4().hex,
                        "action_type": "material_light_edit",
                        "summary": str(reply or "轻量修正"),
                    },
                )
                task_messages.append(
                    {
                        "role": "assistant",
                        "content": _compose_three_stage_reply(
                            "好的，我已经理解并执行了这条轻量修正。",
                            str(reply or "已更新当前表格。"),
                            "如需恢复，我可以撤销刚才的修改；也可以继续帮你检查剩余风险。",
                        ),
                    }
                )
                task = updated_task or task
                fields = updated_fields or fields
                st.rerun()

        if intent.intent_type == "ambiguous":
            task_messages.append(
                {
                    "role": "assistant",
                    "content": _compose_three_stage_reply(
                        "我理解你在表达“结果可能不太对”。",
                        f"目前明细 {len(rows)} 行，质量提示 {len(quality_hints)} 条，待确认动作 {len(pending_actions)} 条。",
                        "你可以告诉我具体行号和字段（如“第5行规格改为...”），或者让我先解释风险最高的几行。",
                    ),
                }
            )
            st.rerun()

        llm_reply = _generate_material_agent_reply_llm(user_input, task, fields, task_messages)
        if not llm_reply:
            llm_reply = "我先解释当前判断，不会直接改数据。你可以继续追问原因，或告诉我希望改成什么。"
        task_messages.append(
            {
                "role": "assistant",
                "content": _compose_three_stage_reply(
                    "收到，我先按对话方式给你解释。",
                    llm_reply,
                    "如果你希望我执行修改，我会先判断风险：低风险可直接改，高风险先进入待确认区。",
                ),
            }
        )
        st.rerun()


def _render_material_flow() -> None:
    _render_flow_back_to_home("material")
    _render_material_conversation_agent()


def _render_travel_flow() -> None:
    _render_flow_back_to_home("travel")
    st.subheader("差旅费流程")
    st.caption("差旅流程提供会话式 Agent 自动整理，也支持手工槽位补录；可导出标准归档压缩包。")
    st.markdown(
        "- 去程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 返程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 酒店：发票 + 支付记录 + 平台订单截图（无住宿费用时需提供情况说明）"
    )

    _render_travel_conversation_agent()
    st.divider()

    with st.expander("手工槽位校对（可选）", expanded=False):
        go_section = _render_travel_transport_section("1) 出差去程交通报销", "travel_go")
        return_section = _render_travel_transport_section("2) 出差返程交通报销", "travel_return")
        hotel_section = _render_travel_hotel_section("travel_hotel")
        _render_travel_summary(go_section, return_section, hotel_section)

    st.divider()
    _render_travel_package_export()


def main() -> None:
    st.set_page_config(page_title="Finance Agent", layout="wide")
    _inject_ui_styles()
    st.title("财务 Agent（本地工具版）")
    st.caption("首页引导优先：先聊天与预检查，再自动进入差旅/材料正式流程。")

    material_usecase.init_app_runtime()
    _ensure_router_state()
    _render_model_runtime_panel()
    flash_message = _pop_router_flash_message()
    if flash_message:
        st.success(flash_message)

    current_page = str(st.session_state.get("current_page") or PAGE_HOME_GUIDE)
    if current_page == PAGE_HOME_GUIDE:
        _render_home_guide_agent()
        st.divider()
        with st.expander("兼容入口：手动进入正式流程（可选）", expanded=False):
            if "flow_mode_selector" not in st.session_state:
                st.session_state["flow_mode_selector"] = "材料费流程"
            flow_mode = st.radio(
                "选择报销流程",
                options=["材料费流程", "差旅费流程"],
                horizontal=True,
                key="flow_mode_selector",
            )
            if st.button("进入所选流程", use_container_width=True, key="legacy_enter_selected_flow"):
                if flow_mode == "差旅费流程":
                    _set_current_page(PAGE_TRAVEL_FLOW, flash_message="已进入差旅正式流程。")
                else:
                    _set_current_page(PAGE_MATERIAL_FLOW, flash_message="已进入材料费正式流程。")
                st.rerun()
    elif current_page == PAGE_MATERIAL_FLOW:
        _render_material_flow()
    elif current_page == PAGE_TRAVEL_FLOW:
        _render_travel_flow()
    else:
        _set_current_page(PAGE_HOME_GUIDE)
        st.rerun()


if __name__ == "__main__":
    main()


