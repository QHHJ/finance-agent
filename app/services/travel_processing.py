from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime
from typing import Any, Callable

import requests
import streamlit as st

from app.agents import AgentCommand
from app.services import parser as ocr_parser
from app.services.ollama_config import (
    chat_model as _chat_model,
    env_flag_true as _env_flag_true,
    env_float_value as _env_float_value,
    env_int_value as _env_int_value,
    travel_doc_text_model as _travel_doc_text_model,
    vl_model as _vl_model,
)
from app.ui import task_hub
from app.ui.agent_metrics import record_llm_outcome as _record_llm_outcome
from app.ui.chat_widgets import compose_three_stage_reply as _compose_three_stage_reply
from app.ui.pending_actions import append_pending_action as _append_pending_action
from app.usecases import material_agent as material_usecase
from app.usecases import travel_agent as travel_usecase
from app.utils.json_tools import parse_json_object_loose as _parse_json_object_loose


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
TRAVEL_SLOT_TO_DOC_TYPE = {
    "go_ticket": "transport_ticket",
    "go_payment": "transport_payment",
    "go_detail": "flight_detail",
    "return_ticket": "transport_ticket",
    "return_payment": "transport_payment",
    "return_detail": "flight_detail",
    "hotel_invoice": "hotel_invoice",
    "hotel_payment": "hotel_payment",
    "hotel_order": "hotel_order",
    "unknown": "unknown",
}
TRAVEL_VALID_SLOTS = set(TRAVEL_SLOT_TO_DOC_TYPE.keys())
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


def configure_travel_processing(**dependencies: Callable[..., Any]) -> None:
    _DEPENDENCIES.update({key: value for key, value in dependencies.items() if callable(value)})


def _require_dependency(name: str) -> Callable[..., Any]:
    dependency = _DEPENDENCIES.get(name)
    if not callable(dependency):
        raise RuntimeError(f"travel_processing dependency is not configured: {name}")
    return dependency


def _run_travel_specialist_task(objective: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    return _require_dependency("run_travel_specialist_task")(objective, payload)


def _execute_agent_command(command: AgentCommand, *, scope: str | None = None) -> tuple[bool, dict[str, Any], str]:
    return _require_dependency("execute_agent_command")(command, scope=scope)


def _build_travel_execution_payload(
    *,
    pool_list: list[Any],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
) -> dict[str, Any]:
    return {
        "pool_list": pool_list,
        "assignment": assignment,
        "profiles": profiles,
        "manual_overrides": manual_overrides,
        "manual_slot_overrides": manual_slot_overrides,
        "reclassify_fn": _apply_reclassify_from_user_text,
        "slot_fn": _apply_manual_slot_from_user_text,
        "relabel_fn": _apply_manual_relabel_from_user_text,
        "amount_fn": _apply_manual_amount_from_user_text,
        "build_assignment_fn": _build_assignment_from_profiles,
        "remember_overrides_fn": _remember_manual_overrides,
        "sync_slot_overrides_fn": _sync_manual_slot_overrides,
        "learn_fn": travel_usecase.learn_from_profiles,
        "organize_fn": _organize_travel_materials,
    }


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


def _extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    return material_usecase.extract_pdf_text_from_bytes(file_bytes)


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
def _extract_image_text_with_ollama(file_bytes: bytes, suffix: str = ".png") -> str:
    if not file_bytes:
        return ""
    normalized_suffix = str(suffix or ".png").strip().lower()
    if not normalized_suffix.startswith("."):
        normalized_suffix = f".{normalized_suffix}"
    text = str(ocr_parser.parse_file_bytes(file_bytes, normalized_suffix) or "")
    return text[:4000]


def _normalize_confidence(value: Any) -> float | None:
    confidence = _safe_float(value)
    if confidence is None:
        return None
    if confidence > 1:
        confidence /= 100.0
    if confidence < 0 or confidence > 1:
        return None
    return confidence


def _normalize_travel_classify_result(parsed: dict[str, Any], file_name: str) -> dict[str, Any]:
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


def _travel_doc_classify_prompt() -> str:
    return (
        "你是差旅报销材料分类助手。必须仅根据文档内容分类，不能根据文件名猜测。\n"
        "允许的 doc_type 只有：transport_ticket, transport_payment, flight_detail, hotel_invoice, hotel_payment, hotel_order, unknown。\n"
        "分类口径：\n"
        "- transport_ticket: 交通发票/票据，常见发票号码、购买方、销售方、价税合计，项目与机票/客运相关。\n"
        "- transport_payment: 交通支付凭证，常见交易成功、支付时间、付款方式、账单详情，且不是发票版式。\n"
        "- flight_detail: 机票明细，常见价格明细、票价、机建、燃油、航段、退改签，且不是发票版式。\n"
        "- hotel_invoice: 酒店发票，常见发票号码、购买方、销售方、价税合计，项目与住宿/房费相关。\n"
        "- hotel_payment: 酒店支付凭证，常见酒店订单支付成功、付款方式、实付金额，且不是发票版式。\n"
        "- hotel_order: 酒店订单明细，常见入住、离店、几晚、房型、订单号、取消政策。\n"
        "强规则：\n"
        "- 若发票项目/服务名称为“代订机票费/客运服务/机票相关服务”，优先判为 transport_ticket。\n"
        "- 若发票项目/服务名称为“住宿服务/房费/酒店住宿”，优先判为 hotel_invoice。\n"
        "- 不要把所有电子发票默认判为酒店发票。\n"
        "输出必须是单个 JSON 对象，不要额外文本，格式如下：\n"
        '{"doc_type":"transport_payment","confidence":0.88,"amount":"2360.00","date":"2026-03-13","evidence":"根据文档主体内容判断为交通支付凭证","ocr_text":"..."}\n'
        "约束：\n"
        "- confidence 取值 0~1。\n"
        "- amount 为主要金额（支付金额或发票价税合计）；无法识别填 null。\n"
        "- date 用 YYYY-MM-DD；无法识别填 null。\n"
        "- evidence 用一句自然语言说明判断依据，不要写“命中关键词”。\n"
    )


def _travel_transport_subtype_prompt() -> str:
    return (
        "你是交通类报销材料细分类助手。只判断 transport_ticket / transport_payment / flight_detail / unknown。\n"
        "必须只看正文内容，不允许使用文件名或路径。\n"
        "判定规则：\n"
        "- 含发票号码、购买方、销售方、价税合计等发票结构，且项目/备注与客运、机票、代订机票费相关 => transport_ticket。\n"
        "- 含交易成功、支付成功、支付时间、付款方式、账单详情、实付金额，且不是发票版式 => transport_payment。\n"
        "- 含价格明细、票价、机建、燃油、航段、退改签、乘机人明细，且不是发票版式 => flight_detail。\n"
        "- 如果证据不足，输出 unknown。\n"
        "输出单个 JSON：\n"
        '{"doc_type":"flight_detail","confidence":0.84,"evidence":"正文主体为机票价格明细与航段信息"}\n'
    )


def _travel_hotel_subtype_prompt() -> str:
    return (
        "你是酒店类报销材料细分类助手。只判断 hotel_invoice / hotel_payment / hotel_order / unknown。\n"
        "必须只看正文内容，不允许使用文件名或路径。\n"
        "判定规则：\n"
        "- 含发票号码、购买方、销售方、价税合计等发票结构，且项目/备注与住宿服务、房费相关 => hotel_invoice。\n"
        "- 含支付成功、支付时间、付款方式、实付金额、账单详情，且不是发票版式 => hotel_payment。\n"
        "- 含入住、离店、几晚、房型、订单号、取消政策、最晚到店等订单详情 => hotel_order。\n"
        "- 同时有订单详情和支付信息时，若主体是在说明入住/离店/房型/订单，则优先 hotel_order。\n"
        "- 如果证据不足，输出 unknown。\n"
        "输出单个 JSON：\n"
        '{"doc_type":"hotel_order","confidence":0.82,"evidence":"正文主体为酒店入住离店与订单详情"}\n'
    )


def _travel_invoice_subtype_prompt() -> str:
    return (
        "你是差旅票据细分类助手。请只判断发票归属是交通票据还是酒店发票。\n"
        "可选 doc_type 只有：transport_ticket, hotel_invoice, unknown。\n"
        "判断依据必须来自文档文本内容，不允许根据文件名猜测。\n"
        "判定优先级（从高到低）：\n"
        "1) 项目/服务名称语义（最高优先级）：\n"
        "   - 代订机票费、客运服务、机票代理服务 => transport_ticket\n"
        "   - 住宿服务、房费、酒店住宿 => hotel_invoice\n"
        "2) 销售方语义：航空服务/票务/铁路 更偏 transport_ticket；酒店管理/酒店公司 更偏 hotel_invoice。\n"
        "3) 备注语义：航班号、飞猪订单、航段 更偏 transport_ticket；入住/离店/几晚 更偏 hotel_invoice。\n"
        "4) 当 1) 与 2)/3) 冲突时，始终以 1) 为准。\n"
        "输出单个 JSON：\n"
        '{"doc_type":"transport_ticket","confidence":0.82,"evidence":"文档主体为客运/机票相关开票内容"}\n'
        "约束：\n"
        "- confidence 范围 0~1。\n"
        "- evidence 一句话说明依据。\n"
    )


def _travel_direction_prompt(doc_type: str) -> str:
    normalized = str(doc_type or "transport_ticket").strip()
    if normalized not in {"transport_ticket", "transport_payment", "flight_detail"}:
        normalized = "transport_ticket"
    home_city = str(os.getenv("TRAVEL_HOME_CITY") or os.getenv("OLLAMA_TRAVEL_HOME_CITY") or "长春").strip()
    return (
        "你是差旅行程方向判别助手。根据文档正文判断该材料属于去程还是返程。\n"
        f"当前材料类型：{normalized}\n"
        f"默认出发/常驻城市：{home_city or '长春'}。\n"
        "可选 direction 只有：go, return, unknown。\n"
        "禁止根据文件名猜测，必须依据正文可见信息（起终点、行程方向、往返描述、日期上下文等）。\n"
        f"若正文路线是“{home_city or '长春'} -> 其他城市”，判断为 go；若正文路线是“其他城市 -> {home_city or '长春'}”，判断为 return。\n"
        "不要根据贴纸、营销文案、页面标题里的泛化词推断方向。\n"
        "输出单个 JSON：\n"
        '{"direction":"go","confidence":0.77,"evidence":"正文显示从出发地到目的地，为去程"}\n'
        "约束：\n"
        "- confidence 范围 0~1。\n"
        "- evidence 一句话说明依据。\n"
    )


def _should_use_vl_classify_fallback(raw_text: str, text_result: dict[str, Any] | None) -> bool:
    min_text_chars = max(20, _env_int_value("OLLAMA_TRAVEL_CLASSIFY_MIN_TEXT_CHARS", 50))
    confidence_threshold = _env_float_value("OLLAMA_TRAVEL_CLASSIFY_MIN_CONFIDENCE", 0.55)
    if confidence_threshold < 0:
        confidence_threshold = 0.0
    if confidence_threshold > 1:
        confidence_threshold = 1.0

    compact_len = len(re.sub(r"\s+", "", str(raw_text or "")))
    raw_text_too_short = compact_len < min_text_chars
    if raw_text_too_short:
        return True

    if not text_result:
        return True

    doc_type = str(text_result.get("doc_type") or "unknown")
    confidence = _normalize_confidence(text_result.get("confidence"))
    if doc_type == "unknown":
        return True
    if confidence is None:
        return True
    return confidence < confidence_threshold


def _travel_has_invoice_structure(raw_text: str) -> bool:
    text = str(raw_text or "")
    if not text:
        return False
    markers = ["发票号码", "购买方", "销售方", "价税合计", "税额", "开票日期"]
    score = sum(1 for token in markers if token in text)
    return score >= 2


def _count_text_hits(raw_text: str, tokens: list[str]) -> int:
    text = str(raw_text or "")
    if not text:
        return 0
    return sum(1 for token in tokens if token and token in text)


def _payment_record_signal_score(raw_text: str) -> int:
    text = str(raw_text or "")
    if not text:
        return 0
    score = _count_text_hits(
        text,
        [
            "交易成功",
            "支付成功",
            "支付时间",
            "付款方式",
            "账单详情",
            "实付",
            "支出金额",
            "订单付款",
            "付款{",
            "支付金额",
            "支付 ¥",
            "支付￥",
            "支付¥",
            "下单",
            "储蓄卡",
            "银行卡",
            "余额宝",
            "支付宝",
            "微信支付",
            "财付通",
            "收单机构",
            "商户单号",
            "商家订单号",
            "交易单号",
            "商户全称",
            "扫码退款",
            "账单管理",
            "计入收支",
            "账单服务",
        ],
    )
    if re.search(r"支付\s*[¥￥]\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?", text):
        score += 2
    if re.search(r"[-−]\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})", text):
        score += 1
    return score


def _travel_structure_doc_type_guard(raw_text: str, current_doc_type: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    current = str(current_doc_type or "unknown").strip()
    if not text:
        return "unknown", ""

    invoice_structure = _travel_has_invoice_structure(text)
    transport_context = _count_text_hits(
        text,
        ["机票", "航班", "客运", "代订机票", "机建", "燃油", "乘机", "航段", "交通", "机场", "出行", "票号"],
    ) > 0
    hotel_context = _count_text_hits(text, ["酒店", "宾馆", "旅馆", "住宿", "入住", "离店", "房型", "房费", "几晚", "间夜"]) > 0

    flight_detail_score = _count_text_hits(
        text,
        [
            "价格明细",
            "票价",
            "机建",
            "燃油",
            "航段",
            "退改签",
            "乘机人",
            "出行信息",
            "出行人",
            "票号",
            "机场",
            "总额",
            "总金额",
            "人均价",
            "普通成人",
        ],
    )
    payment_record_score = _payment_record_signal_score(text)
    transport_payment_score = payment_record_score
    hotel_payment_score = payment_record_score + _count_text_hits(text, ["已支付", "酒店预订"])
    hotel_order_score = _count_text_hits(
        text,
        [
            "入住",
            "离店",
            "几晚",
            "间夜",
            "取消政策",
            "房型",
            "最晚到店",
            "订单详情",
            "订单号",
            "费用明细",
            "晚明细",
            "在线付",
            "总计",
            "查看全部消息",
        ],
    )

    if invoice_structure:
        if current in {"transport_payment", "flight_detail"} and transport_context:
            return "transport_ticket", "结构校正: 发票版式且主体为交通/机票内容"
        if current in {"hotel_payment", "hotel_order"} and hotel_context:
            return "hotel_invoice", "结构校正: 发票版式且主体为住宿/酒店内容"
        if current in {"transport_payment", "flight_detail", "hotel_payment", "hotel_order", "unknown"} and not transport_context:
            return "hotel_invoice", "结构校正: 发票版式不应归为支付凭证或明细"
        return "unknown", ""

    if hotel_context and hotel_order_score >= 3 and hotel_order_score >= hotel_payment_score:
        return "hotel_order", f"结构校正: 酒店订单详情完整度较高({hotel_order_score})"
    if hotel_context and hotel_payment_score >= 2 and current in {"hotel_invoice", "hotel_payment", "hotel_order", "unknown"}:
        return "hotel_payment", f"结构校正: 酒店支付凭证特征较完整({hotel_payment_score})"
    if transport_context and transport_payment_score >= 2 and current in {"transport_ticket", "transport_payment", "flight_detail", "unknown"}:
        return "transport_payment", f"结构校正: 交通支付凭证特征较完整({transport_payment_score})"
    if (
        transport_context
        and current == "transport_ticket"
        and not invoice_structure
        and transport_payment_score < 2
        and _count_text_hits(text, ["价税合计", "项目名称", "合计", "机票"]) >= 2
    ):
        return "flight_detail", "结构校正: 非发票版式且主体为机票费用/项目明细"
    if (
        transport_context
        and transport_payment_score < 2
        and flight_detail_score >= 2
        and current in {"transport_ticket", "transport_payment", "unknown"}
    ):
        return "flight_detail", f"结构校正: 机票明细特征较完整({flight_detail_score})"

    return "unknown", ""


def _invoice_doc_type_guard_from_fields(raw_text: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "unknown", ""
    if not _travel_has_invoice_structure(text):
        return "unknown", ""
    try:
        fields = material_usecase.extract_invoice_fields(text)
    except Exception:
        return "unknown", ""

    item_content = str(fields.get("item_content") or "").strip()
    seller = str(fields.get("seller") or "").strip()
    bill_type = str(fields.get("bill_type") or "").strip()
    merged = " ".join(part for part in [item_content, seller, bill_type, text[:1000]] if part).lower()

    transport_signals = [
        "代订机票费",
        "客运服务",
        "机票",
        "航班",
        "航空服务",
        "飞猪订单",
        "携程机票",
    ]
    hotel_signals = [
        "住宿服务",
        "房费",
        "酒店",
        "入住",
        "离店",
        "几晚",
        "酒店管理",
    ]

    t_hits = [token for token in transport_signals if token in merged]
    h_hits = [token for token in hotel_signals if token in merged]
    if t_hits and not h_hits:
        return "transport_ticket", f"字段守卫命中交通语义: {','.join(t_hits[:3])}"
    if h_hits and not t_hits:
        return "hotel_invoice", f"字段守卫命中酒店语义: {','.join(h_hits[:3])}"

    # 项目名称语义优先于其他信号
    item_lower = item_content.lower()
    if any(token in item_lower for token in ["代订机票费", "客运服务", "机票"]):
        return "transport_ticket", f"项目名称优先: {item_content[:40]}"
    if any(token in item_lower for token in ["住宿服务", "房费", "酒店"]):
        return "hotel_invoice", f"项目名称优先: {item_content[:40]}"
    return "unknown", ""


def _post_travel_text_json(
    *,
    prompt: str,
    raw_text: str,
    timeout_env: str,
    fallback_timeout_env: str,
    default_timeout: int = 12,
    default_fallback_timeout: int = 6,
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None
    text = str(raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _travel_doc_text_model()
    timeout_sec = max(8, _env_int_value(timeout_env, default_timeout))
    fallback_timeout_sec = max(6, _env_int_value(fallback_timeout_env, default_fallback_timeout))
    content = ""

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": f"{prompt}\nDocument text:\n{text[:9000]}",
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, timeout_sec))
        resp.raise_for_status()
        content = resp.json().get("response", "")
    except Exception:
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:9000]}"}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, fallback_timeout_sec))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        except Exception:
            return None

    return _extract_json_from_text(content)


def _normalize_travel_doc_refine_result(
    parsed: dict[str, Any] | None,
    *,
    allowed_doc_types: set[str],
    file_name: str,
) -> dict[str, Any] | None:
    if not parsed:
        return None
    doc_type = str(parsed.get("doc_type") or "unknown").strip()
    if doc_type not in allowed_doc_types:
        doc_type = "unknown"
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if len(evidence) > 160:
        evidence = evidence[:157] + "..."
    return {
        "doc_type": doc_type,
        "confidence": confidence,
        "evidence": evidence,
        "file_name": file_name,
    }


@st.cache_data(show_spinner=False)
def _classify_travel_transport_subtype_with_text_llm(
    raw_text: str,
    file_name: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    _ = retry_tag
    parsed = _post_travel_text_json(
        prompt=_travel_transport_subtype_prompt(),
        raw_text=raw_text,
        timeout_env="OLLAMA_TRAVEL_TRANSPORT_REFINE_TIMEOUT",
        fallback_timeout_env="OLLAMA_TRAVEL_TRANSPORT_REFINE_FALLBACK_TIMEOUT",
        default_timeout=12,
        default_fallback_timeout=6,
    )
    return _normalize_travel_doc_refine_result(
        parsed,
        allowed_doc_types={"transport_ticket", "transport_payment", "flight_detail", "unknown"},
        file_name=file_name,
    )


@st.cache_data(show_spinner=False)
def _classify_travel_hotel_subtype_with_text_llm(
    raw_text: str,
    file_name: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    _ = retry_tag
    parsed = _post_travel_text_json(
        prompt=_travel_hotel_subtype_prompt(),
        raw_text=raw_text,
        timeout_env="OLLAMA_TRAVEL_HOTEL_REFINE_TIMEOUT",
        fallback_timeout_env="OLLAMA_TRAVEL_HOTEL_REFINE_FALLBACK_TIMEOUT",
        default_timeout=12,
        default_fallback_timeout=6,
    )
    return _normalize_travel_doc_refine_result(
        parsed,
        allowed_doc_types={"hotel_invoice", "hotel_payment", "hotel_order", "unknown"},
        file_name=file_name,
    )


@st.cache_data(show_spinner=False)
def _classify_travel_invoice_subtype_with_text_llm(
    raw_text: str,
    file_name: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    _ = retry_tag
    text = str(raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _travel_doc_text_model()
    timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_INVOICE_REFINE_TIMEOUT", 12))
    fallback_timeout_sec = max(6, _env_int_value("OLLAMA_TRAVEL_INVOICE_REFINE_FALLBACK_TIMEOUT", 6))
    prompt = _travel_invoice_subtype_prompt()
    try:
        fields = material_usecase.extract_invoice_fields(text)
    except Exception:
        fields = {}
    field_hint = {
        "item_content": str((fields or {}).get("item_content") or "").strip(),
        "seller": str((fields or {}).get("seller") or "").strip(),
        "bill_type": str((fields or {}).get("bill_type") or "").strip(),
    }
    content = ""

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": (
                f"{prompt}\n"
                f"结构化候选字段（可能有误，仅供参考）:\n{json.dumps(field_hint, ensure_ascii=False)}\n"
                f"Document text:\n{text[:9000]}"
            ),
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, timeout_sec))
        resp.raise_for_status()
        content = resp.json().get("response", "")
    except Exception:
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n"
                            f"结构化候选字段（可能有误，仅供参考）:\n{json.dumps(field_hint, ensure_ascii=False)}\n"
                            f"Document text:\n{text[:9000]}"
                        ),
                    }
                ],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, fallback_timeout_sec))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None
    doc_type = str(parsed.get("doc_type") or "unknown").strip()
    if doc_type not in {"transport_ticket", "hotel_invoice", "unknown"}:
        doc_type = "unknown"
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if len(evidence) > 160:
        evidence = evidence[:157] + "..."
    return {
        "doc_type": doc_type,
        "confidence": confidence,
        "evidence": evidence,
        "file_name": file_name,
    }


@st.cache_data(show_spinner=False)
def _classify_travel_direction_with_text_llm(
    raw_text: str,
    file_name: str,
    doc_type: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    _ = retry_tag
    text = str(raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _travel_doc_text_model()
    timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_DIRECTION_TIMEOUT", 12))
    fallback_timeout_sec = max(6, _env_int_value("OLLAMA_TRAVEL_DIRECTION_FALLBACK_TIMEOUT", 6))
    prompt = _travel_direction_prompt(doc_type)
    content = ""

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": f"{prompt}\nDocument text:\n{text[:9000]}",
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, timeout_sec))
        resp.raise_for_status()
        content = resp.json().get("response", "")
    except Exception:
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:9000]}"}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, fallback_timeout_sec))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None
    direction = str(parsed.get("direction") or "unknown").strip().lower()
    if direction not in {"go", "return", "unknown"}:
        direction = "unknown"
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if len(evidence) > 160:
        evidence = evidence[:157] + "..."
    return {
        "direction": direction,
        "confidence": confidence,
        "evidence": evidence,
        "file_name": file_name,
    }


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

    allow_filehash_override = str(os.getenv("ENABLE_TRAVEL_LEARNED_FILEHASH_OVERRIDE", "0")).strip().lower() in TRUE_VALUES
    if key and allow_filehash_override:
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

    # Disable fuzzy learned override by default to avoid cross-file overfitting.
    if not _env_flag_true("ENABLE_TRAVEL_LEARNED_FUZZY_OVERRIDE"):
        return None, ""

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
    amount_timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_AMOUNT_TIMEOUT", 14))
    amount_fallback_timeout_sec = max(6, _env_int_value("OLLAMA_TRAVEL_AMOUNT_FALLBACK_TIMEOUT", 6))
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
            resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=(8, amount_timeout_sec))
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
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, amount_fallback_timeout_sec))
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
    amount_timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_AMOUNT_TIMEOUT", 14))
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
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, amount_timeout_sec))
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
    image_text = _extract_image_text_with_ollama(file_bytes, suffix)
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


def _set_manual_slot_override_for_profile(manual_slot_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    slot = str(profile.get("manual_slot") or profile.get("slot") or "").strip()
    if not key or slot not in TRAVEL_VALID_SLOTS:
        return
    manual_slot_overrides[key] = slot


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


def _remember_manual_slot_overrides(manual_slot_overrides: dict[str, str], profiles: list[dict[str, Any]]) -> int:
    updated = 0
    for profile in profiles:
        source = str(profile.get("source") or "")
        slot = str(profile.get("manual_slot") or "").strip()
        if source not in {
            "manual_chat",
            "manual_table",
            "manual_persist",
            "manual_chat_slot",
            "manual_chat_slot_llm",
            "manual_chat_slot_rule",
            "manual_slot_persist",
        }:
            continue
        key = _profile_file_key(profile)
        if not key or slot not in TRAVEL_VALID_SLOTS:
            continue
        if manual_slot_overrides.get(key) == slot:
            continue
        manual_slot_overrides[key] = slot
        updated += 1
    return updated


def _sync_manual_slot_overrides(manual_slot_overrides: dict[str, str], profiles: list[dict[str, Any]]) -> int:
    changed = 0
    next_values: dict[str, str] = {}
    for profile in profiles:
        key = _profile_file_key(profile)
        slot = str(profile.get("manual_slot") or "").strip()
        if key and slot in TRAVEL_VALID_SLOTS:
            next_values[key] = slot
    for key in list(manual_slot_overrides.keys()):
        if key not in next_values:
            manual_slot_overrides.pop(key, None)
            changed += 1
    for key, slot in next_values.items():
        if manual_slot_overrides.get(key) != slot:
            manual_slot_overrides[key] = slot
            changed += 1
    return changed


def _remove_manual_override_for_profile(manual_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    if key:
        manual_overrides.pop(key, None)


def _remove_manual_slot_override_for_profile(manual_slot_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    if key:
        manual_slot_overrides.pop(key, None)


def _prune_manual_overrides(manual_overrides: dict[str, str], pool_files: list[Any]) -> None:
    valid_keys = {_uploaded_file_key(file) for file in pool_files if file is not None}
    for key in list(manual_overrides.keys()):
        if key not in valid_keys:
            manual_overrides.pop(key, None)


def _prune_manual_slot_overrides(manual_slot_overrides: dict[str, str], pool_files: list[Any]) -> None:
    valid_keys = {_uploaded_file_key(file) for file in pool_files if file is not None}
    for key in list(manual_slot_overrides.keys()):
        if key not in valid_keys:
            manual_slot_overrides.pop(key, None)


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


def _apply_manual_slot_overrides_to_profiles(profiles: list[dict[str, Any]], manual_slot_overrides: dict[str, str]) -> int:
    if not manual_slot_overrides:
        return 0
    changed = 0
    for profile in profiles:
        key = _profile_file_key(profile)
        target_slot = str(manual_slot_overrides.get(key) or "").strip()
        if target_slot not in TRAVEL_VALID_SLOTS:
            continue
        target_doc_type = str(TRAVEL_SLOT_TO_DOC_TYPE.get(target_slot) or "unknown")
        current_doc_type = str(profile.get("doc_type") or "unknown")
        current_slot = str(profile.get("slot") or "")
        current_manual_slot = str(profile.get("manual_slot") or "")
        if current_doc_type != target_doc_type:
            profile["doc_type"] = target_doc_type
            changed += 1
        if current_slot != target_slot or current_manual_slot != target_slot:
            profile["slot"] = target_slot
            profile["manual_slot"] = target_slot
            changed += 1
        if not str(profile.get("source") or "").startswith("manual"):
            profile["source"] = "manual_slot_persist"
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


@st.cache_data(show_spinner=False)
def _classify_travel_doc_with_text_llm(
    raw_text: str,
    file_name: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    _ = retry_tag
    text = str(raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _travel_doc_text_model()
    classify_timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_CLASSIFY_TIMEOUT", 16))
    classify_fallback_timeout_sec = max(6, _env_int_value("OLLAMA_TRAVEL_CLASSIFY_FALLBACK_TIMEOUT", 6))
    prompt = _travel_doc_classify_prompt()
    content = ""

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": f"{prompt}\nDocument text:\n{text[:12000]}",
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, classify_timeout_sec))
        resp.raise_for_status()
        content = resp.json().get("response", "")
    except Exception:
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:12000]}"}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, classify_fallback_timeout_sec))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None
    return _normalize_travel_classify_result(parsed, file_name)


@st.cache_data(show_spinner=False)
def _classify_travel_doc_with_vl_fallback(
    file_bytes: bytes,
    suffix: str,
    file_name: str,
    raw_text: str,
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    _ = retry_tag
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _vl_model()
    classify_timeout_sec = max(8, _env_int_value("OLLAMA_TRAVEL_CLASSIFY_TIMEOUT", 16))
    classify_fallback_timeout_sec = max(6, _env_int_value("OLLAMA_TRAVEL_CLASSIFY_FALLBACK_TIMEOUT", 6))
    prompt = _travel_doc_classify_prompt()
    text = str(raw_text or "").strip()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    content = ""

    try:
        if suffix in image_suffixes:
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            chat_prompt = prompt
            if text:
                chat_prompt += f"\nOCR text (for reference only):\n{text[:6000]}"
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": chat_prompt, "images": [encoded]}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, classify_timeout_sec))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        elif text:
            payload = {
                "model": model,
                "stream": False,
                "prompt": f"{prompt}\nDocument text:\n{text[:12000]}",
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, classify_timeout_sec))
            resp.raise_for_status()
            content = resp.json().get("response", "")
        else:
            return None
    except Exception:
        try:
            if suffix in image_suffixes:
                encoded = base64.b64encode(file_bytes).decode("utf-8")
                fallback_prompt = prompt
                if text:
                    fallback_prompt += f"\nOCR text (for reference only):\n{text[:6000]}"
                payload = {
                    "model": model,
                    "stream": False,
                    "prompt": fallback_prompt,
                    "images": [encoded],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, classify_fallback_timeout_sec))
                resp.raise_for_status()
                content = resp.json().get("response", "")
            elif text:
                payload = {
                    "model": model,
                    "stream": False,
                    "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:12000]}"}],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, classify_fallback_timeout_sec))
                resp.raise_for_status()
                content = (resp.json().get("message") or {}).get("content", "")
            else:
                return None
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None
    return _normalize_travel_classify_result(parsed, file_name)


def _recognize_travel_file(uploaded_file: Any, index: int, retry_tag: str = "") -> dict[str, Any]:
    t_total_start = time.perf_counter()
    file_name = str(getattr(uploaded_file, "name", ""))
    suffix = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    file_sha1 = _sha1_of_bytes(file_bytes)

    t_ocr_start = time.perf_counter()
    try:
        raw_text = str(ocr_parser.parse_file_bytes(file_bytes, suffix) or "")
    except Exception:
        raw_text = ""
    ocr_elapsed_sec = max(0.0, time.perf_counter() - t_ocr_start)

    t_classify_start = time.perf_counter()
    text_result = _classify_travel_doc_with_text_llm(
        raw_text=raw_text,
        file_name=file_name,
        retry_tag=retry_tag,
    )
    llm_result = text_result
    source = "llm_text"
    if _should_use_vl_classify_fallback(raw_text, text_result):
        vl_result = _classify_travel_doc_with_vl_fallback(
            file_bytes=file_bytes,
            suffix=suffix,
            file_name=file_name,
            raw_text=raw_text,
            retry_tag=retry_tag,
        )
        vl_doc_type = str((vl_result or {}).get("doc_type") or "unknown")
        if vl_result and vl_doc_type != "unknown":
            llm_result = vl_result
            source = "llm_vl_fallback"
        elif llm_result is None and vl_result is not None:
            llm_result = vl_result
            source = "llm_vl_fallback_unknown"

    classify_elapsed_sec = max(0.0, time.perf_counter() - t_classify_start)
    llm_guess = str((llm_result or {}).get("doc_type") or "unknown")
    llm_confidence = _normalize_confidence((llm_result or {}).get("confidence"))
    llm_ocr_text = str((llm_result or {}).get("ocr_text") or "").strip()
    merged_signal_text = "\n".join(part for part in [raw_text, llm_ocr_text] if part)
    guessed = llm_guess if llm_guess in TRAVEL_DOC_TYPES else "unknown"
    if guessed == "unknown" and llm_result is None:
        source = "llm_unavailable"
    evidence_parts: list[str] = [str((llm_result or {}).get("evidence") or "").strip()]

    # LLM second-pass binary invoice disambiguation, to reduce transport/hotel invoice混淆.
    invoice_like = guessed in {"transport_ticket", "hotel_invoice", "unknown"} or (
        "发票" in merged_signal_text or "电子发票" in merged_signal_text
    )
    if invoice_like:
        invoice_refine = _classify_travel_invoice_subtype_with_text_llm(
            raw_text=merged_signal_text or raw_text,
            file_name=file_name,
            retry_tag=retry_tag,
        )
        refine_doc_type = str((invoice_refine or {}).get("doc_type") or "unknown")
        refine_confidence = _normalize_confidence((invoice_refine or {}).get("confidence"))
        if refine_doc_type in {"transport_ticket", "hotel_invoice"}:
            min_refine_conf = _env_float_value("OLLAMA_TRAVEL_INVOICE_REFINE_MIN_CONFIDENCE", 0.62)
            if min_refine_conf < 0:
                min_refine_conf = 0.0
            if min_refine_conf > 1:
                min_refine_conf = 1.0
            if guessed == "unknown" or (refine_confidence is not None and refine_confidence >= min_refine_conf):
                if guessed != refine_doc_type:
                    guessed = refine_doc_type
                    source = f"{source}+invoice_refine"
                refine_evidence = str((invoice_refine or {}).get("evidence") or "").strip()
                if refine_evidence:
                    evidence_parts.append(f"发票二次判别: {refine_evidence}")

    field_guard_doc_type, field_guard_evidence = _invoice_doc_type_guard_from_fields(merged_signal_text or raw_text)
    if field_guard_doc_type in {"transport_ticket", "hotel_invoice"}:
        if guessed != field_guard_doc_type:
            guessed = field_guard_doc_type
            source = f"{source}+invoice_field_guard"
        if field_guard_evidence:
            evidence_parts.append(field_guard_evidence)

    invoice_structure_hit = _travel_has_invoice_structure(merged_signal_text or raw_text)
    transport_refine_needed = guessed in {"transport_ticket", "transport_payment", "flight_detail", "unknown"} or _count_text_hits(
        merged_signal_text or raw_text,
        ["机票", "航班", "机建", "燃油", "客运", "代订机票", "交通", "乘机"],
    ) > 0
    if transport_refine_needed:
        transport_refine = _classify_travel_transport_subtype_with_text_llm(
            raw_text=merged_signal_text or raw_text,
            file_name=file_name,
            retry_tag=retry_tag,
        )
        transport_doc_type = str((transport_refine or {}).get("doc_type") or "unknown")
        transport_confidence = _normalize_confidence((transport_refine or {}).get("confidence"))
        min_transport_refine_conf = _env_float_value("OLLAMA_TRAVEL_TRANSPORT_REFINE_MIN_CONFIDENCE", 0.58)
        if min_transport_refine_conf < 0:
            min_transport_refine_conf = 0.0
        if min_transport_refine_conf > 1:
            min_transport_refine_conf = 1.0
        can_apply_transport_refine = (
            transport_doc_type in {"transport_ticket", "transport_payment", "flight_detail"}
            and (guessed == "unknown" or (transport_confidence is not None and transport_confidence >= min_transport_refine_conf))
        )
        if invoice_structure_hit and transport_doc_type in {"transport_payment", "flight_detail"}:
            can_apply_transport_refine = False
        if can_apply_transport_refine:
            if guessed != transport_doc_type:
                guessed = transport_doc_type
                source = f"{source}+transport_refine"
            transport_evidence = str((transport_refine or {}).get("evidence") or "").strip()
            if transport_evidence:
                evidence_parts.append(f"交通细分: {transport_evidence}")

    hotel_refine_needed = guessed in {"hotel_invoice", "hotel_payment", "hotel_order", "unknown"} or _count_text_hits(
        merged_signal_text or raw_text,
        ["酒店", "住宿", "入住", "离店", "房费", "房型", "几晚", "间夜"],
    ) > 0
    if hotel_refine_needed:
        hotel_refine = _classify_travel_hotel_subtype_with_text_llm(
            raw_text=merged_signal_text or raw_text,
            file_name=file_name,
            retry_tag=retry_tag,
        )
        hotel_doc_type = str((hotel_refine or {}).get("doc_type") or "unknown")
        hotel_confidence = _normalize_confidence((hotel_refine or {}).get("confidence"))
        min_hotel_refine_conf = _env_float_value("OLLAMA_TRAVEL_HOTEL_REFINE_MIN_CONFIDENCE", 0.58)
        if min_hotel_refine_conf < 0:
            min_hotel_refine_conf = 0.0
        if min_hotel_refine_conf > 1:
            min_hotel_refine_conf = 1.0
        can_apply_hotel_refine = (
            hotel_doc_type in {"hotel_invoice", "hotel_payment", "hotel_order"}
            and (guessed == "unknown" or (hotel_confidence is not None and hotel_confidence >= min_hotel_refine_conf))
        )
        if invoice_structure_hit and hotel_doc_type in {"hotel_payment", "hotel_order"}:
            can_apply_hotel_refine = False
        if can_apply_hotel_refine:
            if guessed != hotel_doc_type:
                guessed = hotel_doc_type
                source = f"{source}+hotel_refine"
            hotel_evidence = str((hotel_refine or {}).get("evidence") or "").strip()
            if hotel_evidence:
                evidence_parts.append(f"酒店细分: {hotel_evidence}")

    structure_guard_doc_type, structure_guard_evidence = _travel_structure_doc_type_guard(merged_signal_text or raw_text, guessed)
    if structure_guard_doc_type in TRAVEL_DOC_TYPES and structure_guard_doc_type != "unknown":
        if guessed != structure_guard_doc_type:
            evidence_parts = []
            guessed = structure_guard_doc_type
            source = f"{source}+structure_guard"
        elif structure_guard_evidence:
            existing_evidence = " ".join(part for part in evidence_parts if str(part or "").strip())
            if structure_guard_doc_type in {"transport_payment", "hotel_payment"} and any(
                token in existing_evidence for token in ["发票特征", "交通发票", "酒店发票", "票据"]
            ):
                evidence_parts = []
            elif structure_guard_doc_type in {"flight_detail", "hotel_order"} and any(
                token in existing_evidence for token in ["支付凭证", "支付记录", "发票特征"]
            ):
                evidence_parts = []
        if structure_guard_evidence:
            evidence_parts.append(structure_guard_evidence)

    learned_doc_type, learned_source = _lookup_learned_doc_type_override(
        file_sha1,
        file_name,
        merged_signal_text,
        guessed,
    )
    if learned_doc_type and learned_doc_type in TRAVEL_DOC_TYPES and learned_doc_type != guessed:
        guessed = learned_doc_type
        source = learned_source or source
        evidence_parts.append(f"学习覆写: {learned_source}")

    manual_slot = ""
    if guessed in {"transport_ticket", "transport_payment", "flight_detail"}:
        direction_result = _classify_travel_direction_with_text_llm(
            raw_text=merged_signal_text or raw_text,
            file_name=file_name,
            doc_type=guessed,
            retry_tag=retry_tag,
        )
        direction = str((direction_result or {}).get("direction") or "unknown").strip().lower()
        direction_confidence = _normalize_confidence((direction_result or {}).get("confidence"))
        min_direction_conf = _env_float_value("OLLAMA_TRAVEL_DIRECTION_MIN_CONFIDENCE", 0.65)
        if min_direction_conf < 0:
            min_direction_conf = 0.0
        if min_direction_conf > 1:
            min_direction_conf = 1.0
        if direction in {"go", "return"} and direction_confidence is not None and direction_confidence >= min_direction_conf:
            inferred_slot = _slot_target_from_doc_type(direction, guessed)
            if inferred_slot in TRAVEL_VALID_SLOTS:
                manual_slot = inferred_slot
                source = f"{source}+direction_llm"
                dir_evidence = str((direction_result or {}).get("evidence") or "").strip()
                if dir_evidence:
                    evidence_parts.append(f"行程方向: {dir_evidence}")

    t_amount_start = time.perf_counter()
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
    amount_elapsed_sec = max(0.0, time.perf_counter() - t_amount_start)

    date_obj = _pick_primary_date(file_name, merged_signal_text)
    if date_obj is None:
        llm_date = str((llm_result or {}).get("date") or "").strip()
        candidate_dates = _extract_candidate_dates(llm_date)
        if candidate_dates:
            date_obj = candidate_dates[0]

    evidence = "; ".join(part for part in evidence_parts if str(part or "").strip())
    if len(evidence) > 220:
        evidence = evidence[:217] + "..."
    total_elapsed_sec = max(0.0, time.perf_counter() - t_total_start)

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
        "manual_slot": manual_slot,
        "source": source,
        "confidence": llm_confidence,
        "evidence": evidence,
        "file_sha1": file_sha1,
        "raw_text": raw_text[:3000] if raw_text else "",
        "ocr_text": llm_ocr_text[:3000] if llm_ocr_text else "",
        "signal_text": merged_signal_text[:3500] if merged_signal_text else "",
        "timing": {
            "ocr_sec": round(ocr_elapsed_sec, 3),
            "classify_sec": round(classify_elapsed_sec, 3),
            "amount_sec": round(amount_elapsed_sec, 3),
            "total_sec": round(total_elapsed_sec, 3),
        },
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
    auto_profiles: list[dict[str, Any]] = []
    assigned_profiles: dict[str, list[dict[str, Any]]] = {slot: [] for slot in TRAVEL_VALID_SLOTS}
    for profile in profiles:
        manual_slot = str(profile.get("manual_slot") or "").strip()
        if manual_slot in TRAVEL_VALID_SLOTS:
            expected_doc_type = str(TRAVEL_SLOT_TO_DOC_TYPE.get(manual_slot) or "unknown")
            if expected_doc_type in TRAVEL_DOC_TYPES and str(profile.get("doc_type") or "") != expected_doc_type:
                profile["doc_type"] = expected_doc_type
            profile["slot"] = manual_slot
            assigned_profiles[manual_slot].append(profile)
            continue
        auto_profiles.append(profile)

    if auto_profiles:
        travel_usecase.build_assignment_from_profiles(auto_profiles)
        for profile in auto_profiles:
            slot = str(profile.get("slot") or "unknown").strip()
            if slot not in TRAVEL_VALID_SLOTS:
                slot = "unknown"
                profile["slot"] = slot
            assigned_profiles.setdefault(slot, []).append(profile)

    return {
        "go_ticket": [p["file"] for p in assigned_profiles["go_ticket"]],
        "go_payment": [p["file"] for p in assigned_profiles["go_payment"]],
        "go_detail": [p["file"] for p in assigned_profiles["go_detail"]],
        "return_ticket": [p["file"] for p in assigned_profiles["return_ticket"]],
        "return_payment": [p["file"] for p in assigned_profiles["return_payment"]],
        "return_detail": [p["file"] for p in assigned_profiles["return_detail"]],
        "hotel_invoice": [p["file"] for p in assigned_profiles["hotel_invoice"]],
        "hotel_payment": [p["file"] for p in assigned_profiles["hotel_payment"]],
        "hotel_order": [p["file"] for p in assigned_profiles["hotel_order"]],
        "unknown": [p["file"] for p in assigned_profiles["unknown"]],
        "go_ticket_amount": _sum_profile_amount(assigned_profiles["go_ticket"]),
        "go_payment_amount": _sum_profile_amount(assigned_profiles["go_payment"]),
        "return_ticket_amount": _sum_profile_amount(assigned_profiles["return_ticket"]),
        "return_payment_amount": _sum_profile_amount(assigned_profiles["return_payment"]),
        "hotel_invoice_amount": _sum_profile_amount(assigned_profiles["hotel_invoice"]),
        "hotel_payment_amount": _sum_profile_amount(assigned_profiles["hotel_payment"]),
    }


def _organize_travel_materials(
    pool_files: list[Any],
    manual_overrides: dict[str, str] | None = None,
    manual_slot_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ok, payload, summary = _run_travel_specialist_task(
        "organize_materials",
        {
            "pool_files": list(pool_files or []),
            "build_profile": _build_travel_file_profile,
            "manual_overrides": dict(manual_overrides or {}),
            "apply_overrides": _apply_manual_overrides_to_profiles,
            "manual_slot_overrides": dict(manual_slot_overrides or {}),
            "apply_slot_overrides": _apply_manual_slot_overrides_to_profiles,
            "build_assignment": _build_assignment_from_profiles,
        },
    )
    if ok:
        assignment = dict(payload.get("assignment") or {})
        profiles = list(payload.get("profiles") or [])
        return assignment, profiles

    st.session_state["travel_agent_backend_warning"] = summary or "Travel specialist agent unavailable."
    _, profiles = travel_usecase.organize_materials(
        pool_files,
        build_profile=_build_travel_file_profile,
        manual_overrides=manual_overrides,
        apply_overrides=_apply_manual_overrides_to_profiles,
    )
    _apply_manual_slot_overrides_to_profiles(profiles, dict(manual_slot_overrides or {}))
    assignment = _build_assignment_from_profiles(profiles)
    return assignment, profiles


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
        "multiple_slots": "多个槽位",
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
    ok, payload, summary = _run_travel_specialist_task(
        "build_status",
        {"assignment": dict(assignment or {})},
    )
    if ok:
        return dict(payload.get("status") or {})
    st.session_state["travel_agent_backend_warning"] = summary or "Travel status agent unavailable."
    return travel_usecase.build_travel_agent_status(assignment)


def _active_travel_task_id() -> str:
    task_id = task_hub.get_active_travel_task_id()
    if task_id:
        return task_id
    return "default"


def _travel_scope_name(task_id: str | None = None) -> str:
    key = str(task_id or _active_travel_task_id()).strip()
    return f"travel_agent::{key}"


def _travel_undo_stack_key(task_id: str | None = None) -> str:
    key = str(task_id or _active_travel_task_id()).strip()
    return f"travel_agent_undo_stack::{key}"


def _clone_travel_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    timing = dict(profile.get("timing") or {})
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
        "manual_slot": profile.get("manual_slot"),
        "source": profile.get("source"),
        "confidence": profile.get("confidence"),
        "evidence": profile.get("evidence"),
        "file_sha1": profile.get("file_sha1"),
        "raw_text": profile.get("raw_text"),
        "ocr_text": profile.get("ocr_text"),
        "signal_text": profile.get("signal_text"),
        "timing": {
            "ocr_sec": float(timing.get("ocr_sec") or 0.0),
            "classify_sec": float(timing.get("classify_sec") or 0.0),
            "amount_sec": float(timing.get("amount_sec") or 0.0),
            "total_sec": float(timing.get("total_sec") or 0.0),
        },
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
    manual_slot_overrides: dict[str, str] | None = None,
    *,
    task_id: str | None = None,
) -> None:
    stack = st.session_state.setdefault(_travel_undo_stack_key(task_id), [])
    if not isinstance(stack, list):
        stack = []
    stack.append(
        {
            "assignment": _clone_travel_assignment(assignment),
            "profiles": [_clone_travel_profile(p) for p in profiles],
            "manual_overrides": dict(manual_overrides or {}),
            "manual_slot_overrides": dict(manual_slot_overrides or {}),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if len(stack) > 20:
        stack = stack[-20:]
    st.session_state[_travel_undo_stack_key(task_id)] = stack


def _travel_pop_undo_snapshot(task_id: str | None = None) -> dict[str, Any] | None:
    stack = st.session_state.get(_travel_undo_stack_key(task_id))
    if not isinstance(stack, list) or not stack:
        return None
    snapshot = stack.pop()
    st.session_state[_travel_undo_stack_key(task_id)] = stack
    return dict(snapshot) if isinstance(snapshot, dict) else None


def _travel_restore_undo_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, str]]:
    assignment = _clone_travel_assignment(dict(snapshot.get("assignment") or {}))
    profiles = [_clone_travel_profile(p) for p in list(snapshot.get("profiles") or []) if isinstance(p, dict)]
    manual_overrides = dict(snapshot.get("manual_overrides") or {})
    manual_slot_overrides = dict(snapshot.get("manual_slot_overrides") or {})
    return assignment, profiles, manual_overrides, manual_slot_overrides


def _travel_pending_action_spec_from_text(user_text: str) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    if any(token in text for token in ["应用全部修正", "应用全部建议", "覆盖当前分配结果"]):
        return {
            "action_type": "travel_apply_all",
            "summary": "批量应用当前差旅整理建议",
            "target": "当前全部待确认建议",
            "risk_level": "high",
            "payload": {"command": text},
        }
    if any(token in text for token in ["重新归并", "重新分配", "重排分组", "同一趟"]):
        return {
            "action_type": "travel_reorganize",
            "summary": "重新归并差旅材料并刷新槽位分配",
            "target": "全部已上传材料",
            "risk_level": "medium",
            "payload": {"command": text},
        }
    if any(token in text for token in ["导出报销表", "导出结果", "导出压缩包"]):
        return {
            "action_type": "travel_export",
            "summary": "确认导出当前差旅材料压缩包",
            "target": "差旅导出",
            "risk_level": "high",
            "payload": {"command": text},
        }
    return {
        "action_type": "travel_manual_confirm",
        "summary": "执行一条需要确认的差旅调整",
        "target": text[:80],
        "risk_level": "medium",
        "payload": {"command": text},
    }


def _travel_build_pending_action_from_text(user_text: str) -> dict[str, Any] | None:
    spec = _travel_pending_action_spec_from_text(user_text)
    if not spec:
        return None
    return _append_pending_action(
        _travel_scope_name(),
        action_type=str(spec.get("action_type") or ""),
        summary=str(spec.get("summary") or ""),
        target=str(spec.get("target") or ""),
        risk_level=str(spec.get("risk_level") or "medium"),
        payload=dict(spec.get("payload") or {}),
    )


def _append_travel_pending_action_from_spec(scope_name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    action_type = str(spec.get("action_type") or "").strip()
    summary = str(spec.get("summary") or "").strip()
    if not action_type or not summary:
        return None
    return _append_pending_action(
        scope_name,
        action_type=action_type,
        summary=summary,
        target=str(spec.get("target") or ""),
        risk_level=str(spec.get("risk_level") or "medium"),
        payload=dict(spec.get("payload") or {}),
    )


def _travel_execute_pending_action(
    action: dict[str, Any],
    pool_list: list[Any],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]]]:
    ok, payload, summary = _execute_agent_command(
        AgentCommand(
            command_type="travel_pending_action",
            payload={
                "action": dict(action or {}),
                **_build_travel_execution_payload(
                    pool_list=pool_list,
                    assignment=assignment,
                    profiles=profiles,
                    manual_overrides=manual_overrides,
                    manual_slot_overrides=manual_slot_overrides,
                ),
            },
            summary=str(action.get("summary") or "执行差旅待确认动作"),
            risk_level=str(action.get("risk_level") or "medium"),
            requires_confirmation=False,
            created_by="travel_workbench",
        )
    )
    if payload.get("export_confirmed"):
        st.session_state["travel_export_confirmed_from_chat"] = True
    next_assignment = dict(payload.get("assignment") or assignment)
    next_profiles = list(payload.get("profiles") or profiles)
    return ok, summary, next_assignment, next_profiles


def _travel_execute_light_edit(
    user_text: str,
    pool_list: list[Any],
    assignment: dict[str, Any],
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str],
    manual_slot_overrides: dict[str, str],
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ok, payload, summary = _execute_agent_command(
        AgentCommand(
            command_type="travel_light_edit",
            payload={
                "user_text": str(user_text or "").strip(),
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
            created_by="travel_workbench",
        )
    )
    next_assignment = dict(payload.get("assignment") or assignment)
    next_profiles = list(payload.get("profiles") or profiles)
    return ok, summary, next_assignment, next_profiles, payload


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


def _slot_target_from_doc_type(direction: str, doc_type: str) -> str | None:
    direction_text = str(direction or "").strip()
    doc_type_text = str(doc_type or "").strip()
    if doc_type_text == "transport_ticket":
        return "go_ticket" if direction_text == "go" else "return_ticket"
    if doc_type_text == "transport_payment":
        return "go_payment" if direction_text == "go" else "return_payment"
    if doc_type_text == "flight_detail":
        return "go_detail" if direction_text == "go" else "return_detail"
    if doc_type_text in {"hotel_invoice", "hotel_payment", "hotel_order"}:
        return doc_type_text
    return None


def _target_slot_from_user_text(user_text: str, file_name: str = "", current_doc_type: str = "") -> str | None:
    text = str(user_text or "").lower()
    name = str(file_name or "").lower()
    merged = f"{text} {name}"

    explicit_patterns = [
        ("去程机票明细", "go_detail"),
        ("去程明细", "go_detail"),
        ("返程机票明细", "return_detail"),
        ("返程明细", "return_detail"),
        ("去程支付记录", "go_payment"),
        ("去程支付", "go_payment"),
        ("返程支付记录", "return_payment"),
        ("返程支付", "return_payment"),
        ("去程机票发票", "go_ticket"),
        ("去程票据", "go_ticket"),
        ("返程机票发票", "return_ticket"),
        ("返程票据", "return_ticket"),
        ("酒店发票", "hotel_invoice"),
        ("酒店支付记录", "hotel_payment"),
        ("酒店支付", "hotel_payment"),
        ("酒店订单截图", "hotel_order"),
        ("酒店订单", "hotel_order"),
    ]
    for pattern, slot in explicit_patterns:
        if pattern in merged:
            return slot

    if any(token in merged for token in ["去程", "出发"]):
        inferred = _slot_target_from_doc_type("go", current_doc_type or _target_doc_type_from_user_text(user_text, file_name) or "")
        if inferred:
            return inferred
    if any(token in merged for token in ["返程", "回程", "回去"]):
        inferred = _slot_target_from_doc_type("return", current_doc_type or _target_doc_type_from_user_text(user_text, file_name) or "")
        if inferred:
            return inferred

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


def _normalize_travel_slot_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in TRAVEL_VALID_SLOTS:
        return text
    lowered = text.lower()
    alias = {
        "去程机票明细": "go_detail",
        "去程明细": "go_detail",
        "去程航班明细": "go_detail",
        "go_detail": "go_detail",
        "返程机票明细": "return_detail",
        "返程明细": "return_detail",
        "回程明细": "return_detail",
        "return_detail": "return_detail",
        "去程机票发票": "go_ticket",
        "去程票据": "go_ticket",
        "去程发票": "go_ticket",
        "go_ticket": "go_ticket",
        "返程机票发票": "return_ticket",
        "返程票据": "return_ticket",
        "返程发票": "return_ticket",
        "回程票据": "return_ticket",
        "return_ticket": "return_ticket",
        "去程支付记录": "go_payment",
        "去程支付凭证": "go_payment",
        "去程支付": "go_payment",
        "go_payment": "go_payment",
        "返程支付记录": "return_payment",
        "返程支付凭证": "return_payment",
        "返程支付": "return_payment",
        "回程支付": "return_payment",
        "return_payment": "return_payment",
        "酒店发票": "hotel_invoice",
        "住宿发票": "hotel_invoice",
        "hotel_invoice": "hotel_invoice",
        "酒店支付记录": "hotel_payment",
        "酒店支付凭证": "hotel_payment",
        "酒店支付": "hotel_payment",
        "hotel_payment": "hotel_payment",
        "酒店订单": "hotel_order",
        "酒店订单截图": "hotel_order",
        "酒店明细": "hotel_order",
        "酒店订单明细": "hotel_order",
        "住宿明细": "hotel_order",
        "hotel_order": "hotel_order",
        "未知": "unknown",
        "未识别": "unknown",
        "unknown": "unknown",
    }
    return alias.get(lowered, alias.get(text, ""))


def _parse_travel_slot_actions_with_llm(user_text: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text or not profiles:
        return []

    metric_scope = "travel"
    candidates: list[dict[str, Any]] = []
    for profile in profiles[:120]:
        name = str(profile.get("name") or "").strip()
        if not name:
            continue
        candidates.append(
            {
                "file_name": name,
                "amount": _format_amount(_safe_float(profile.get("amount"))),
                "date": str(profile.get("date") or ""),
            }
        )
    if not candidates:
        return []

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()
    valid_slots = [
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
    prompt = (
        "你是差旅报销文件槽位修改解析器。只输出JSON对象，不要解释。\n"
        "任务：从用户一句话里抽取所有“把某个文件改到某个槽位”的动作。\n"
        "只能使用 candidates 里出现的 file_name，不能编造文件名。\n"
        "candidates 只用于定位文件，不代表目标分类；target_slot 只能根据 user_message 判断。\n"
        f"target_slot 只能是：{', '.join(valid_slots)}。\n"
        "槽位含义：go_ticket=去程机票发票/票据；go_payment=去程支付记录；go_detail=去程机票明细；"
        "return_ticket=返程机票发票/票据；return_payment=返程支付记录；return_detail=返程机票明细；"
        "hotel_invoice=酒店发票；hotel_payment=酒店支付记录；hotel_order=酒店订单/酒店明细/入住离店明细。\n"
        "如果用户明确说“酒店明细/酒店订单/住宿明细/入住离店明细”，必须映射为 hotel_order，不能映射为 hotel_payment。\n"
        "只有用户明确说“酒店支付/付款/支付记录/支付凭证”时，才映射为 hotel_payment。\n"
        "如果一句话里有多个文件，要分别输出多个 action。\n"
        "示例：A.jpg是去程明细，B.jpg是返程明细 => 输出两个action，分别是go_detail和return_detail。\n"
        "示例：333.jpg是酒店明细 => target_slot=hotel_order。\n"
        "输出格式：{\"actions\":[{\"file_name\":\"...\",\"target_slot\":\"go_detail\",\"confidence\":0.9,\"reason\":\"...\"}]}\n"
        f"user_message: {text}\n"
        f"candidates: {json.dumps(candidates, ensure_ascii=False)}\n"
    )
    content = ""
    try:
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "你是稳定的JSON动作解析器。"},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(5, 24))
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "")
    except Exception:
        content = ""
    if not content:
        try:
            payload = {
                "model": model,
                "stream": False,
                "prompt": prompt,
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(5, 24))
            resp.raise_for_status()
            content = str(resp.json().get("response") or "")
        except Exception:
            _record_llm_outcome(metric_scope, False)
            return []

    parsed = _parse_json_object_loose(content)
    if not parsed:
        _record_llm_outcome(metric_scope, False)
        return []
    raw_actions = parsed.get("actions")
    if not isinstance(raw_actions, list):
        _record_llm_outcome(metric_scope, False)
        return []
    _record_llm_outcome(metric_scope, True)

    names = [str(item.get("file_name") or "") for item in candidates]

    def _resolve_candidate_name(value: Any) -> str:
        candidate_name = str(value or "").strip()
        if not candidate_name:
            return ""
        if candidate_name in names:
            return candidate_name
        lowered = candidate_name.lower()
        matches = [name for name in names if lowered and (lowered in name.lower() or name.lower() in lowered)]
        return matches[0] if len(matches) == 1 else ""

    actions: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        file_name = _resolve_candidate_name(item.get("file_name") or item.get("name"))
        target_slot = _normalize_travel_slot_value(item.get("target_slot") or item.get("slot") or item.get("target"))
        confidence = _normalize_confidence(item.get("confidence"))
        if not file_name or target_slot not in TRAVEL_VALID_SLOTS:
            continue
        if confidence is not None and confidence < 0.45:
            continue
        actions.append(
            {
                "file_name": file_name,
                "target_slot": target_slot,
                "confidence": confidence,
                "reason": str(item.get("reason") or item.get("evidence") or "").strip(),
            }
        )
    return actions


def _apply_manual_slot_from_user_text(
    user_text: str,
    profiles: list[dict[str, Any]],
) -> tuple[int, list[str], str | None]:
    text = str(user_text or "").strip()
    if not text:
        return 0, [], None

    profile_by_name = {str(profile.get("name") or ""): profile for profile in profiles if str(profile.get("name") or "")}
    llm_actions = _parse_travel_slot_actions_with_llm(text, profiles)
    changed_names: list[str] = []
    changed_slots: list[str] = []
    for action in llm_actions:
        profile = profile_by_name.get(str(action.get("file_name") or ""))
        target_slot = str(action.get("target_slot") or "")
        if not profile or target_slot not in TRAVEL_VALID_SLOTS:
            continue
        target_doc_type = str(TRAVEL_SLOT_TO_DOC_TYPE.get(target_slot) or "unknown")
        current_slot = str(profile.get("manual_slot") or profile.get("slot") or "").strip()
        current_doc_type = str(profile.get("doc_type") or "unknown").strip()
        if current_slot == target_slot and current_doc_type == target_doc_type:
            continue
        profile["manual_slot"] = target_slot
        profile["slot"] = target_slot
        profile["doc_type"] = target_doc_type
        profile["source"] = "manual_chat_slot_llm"
        changed_names.append(str(profile.get("name") or ""))
        changed_slots.append(target_slot)

    if changed_names:
        unique_slots = sorted(set(changed_slots))
        return len(changed_names), changed_names, unique_slots[0] if len(unique_slots) == 1 else "multiple_slots"

    if str(os.getenv("ALLOW_RULE_CHAT_EDIT_FALLBACK") or "").strip() != "1":
        return 0, [], None

    matched_profiles = _match_profiles_by_user_text(text, profiles)
    if not matched_profiles and len(profiles) == 1:
        matched_profiles = list(profiles)
    if not matched_profiles:
        return 0, [], None

    target_slot = _target_slot_from_user_text(
        text,
        str(matched_profiles[0].get("name") or ""),
        str(matched_profiles[0].get("doc_type") or ""),
    )
    if target_slot not in TRAVEL_VALID_SLOTS:
        return 0, [], None

    target_doc_type = str(TRAVEL_SLOT_TO_DOC_TYPE.get(target_slot) or "unknown")
    changed_names: list[str] = []
    for profile in matched_profiles:
        current_slot = str(profile.get("manual_slot") or profile.get("slot") or "").strip()
        current_doc_type = str(profile.get("doc_type") or "unknown").strip()
        if current_slot == target_slot and current_doc_type == target_doc_type:
            continue
        profile["manual_slot"] = target_slot
        profile["slot"] = target_slot
        profile["doc_type"] = target_doc_type
        profile["source"] = "manual_chat_slot_rule"
        changed_names.append(str(profile.get("name") or ""))

    return len(changed_names), changed_names, target_slot


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
        current_manual_slot = str(profile.get("manual_slot") or "").strip()
        if current_manual_slot in TRAVEL_VALID_SLOTS and str(TRAVEL_SLOT_TO_DOC_TYPE.get(current_manual_slot) or "") != target:
            profile.pop("manual_slot", None)
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
    manual_slot_overrides: dict[str, str] | None = None,
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
        if manual_slot_overrides is not None:
            _remove_manual_slot_override_for_profile(manual_slot_overrides, profile)
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
            "manual_slot",
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
    def _slot_names(slot: str) -> list[str]:
        return [str(getattr(item, "name", "") or "") for item in _as_uploaded_list(assignment.get(slot))]

    def _slot_text(slot: str) -> str:
        names = [name for name in _slot_names(slot) if name]
        if not names:
            return "无"
        return "、".join(names)

    def _amount_text(key: str) -> str:
        return _format_amount(_safe_float(assignment.get(key))) or "无"

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

    if any(key in text for key in ["对不上", "对不齐", "对应不上", "哪个和哪个"]) and status["issues"]:
        lines: list[str] = []
        if ("返程" in text or "回程" in text or not any(key in text for key in ["去程", "酒店"])) and (
            _safe_float(assignment.get("return_ticket_amount")) is not None or _safe_float(assignment.get("return_payment_amount")) is not None
        ):
            lines.append(
                "返程这组目前是："
                f"票据={_slot_text('return_ticket')}，支付={_slot_text('return_payment')}，"
                f"金额={_amount_text('return_ticket_amount')} vs {_amount_text('return_payment_amount')}。"
            )
        if ("酒店" in text or not any(key in text for key in ["去程", "返程", "回程"])) and (
            _safe_float(assignment.get("hotel_invoice_amount")) is not None or _safe_float(assignment.get("hotel_payment_amount")) is not None
        ):
            lines.append(
                "酒店这组目前是："
                f"发票={_slot_text('hotel_invoice')}，支付={_slot_text('hotel_payment')}，订单={_slot_text('hotel_order')}，"
                f"金额={_amount_text('hotel_invoice_amount')} vs {_amount_text('hotel_payment_amount')}。"
            )
        if ("去程" in text) and (
            _safe_float(assignment.get("go_ticket_amount")) is not None or _safe_float(assignment.get("go_payment_amount")) is not None
        ):
            lines.append(
                "去程这组目前是："
                f"票据={_slot_text('go_ticket')}，支付={_slot_text('go_payment')}，"
                f"金额={_amount_text('go_ticket_amount')} vs {_amount_text('go_payment_amount')}。"
            )
        if lines:
            return "\n".join(lines)

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

    if "酒店" in text and any(key in text for key in ["支付", "发票", "订单", "对应", "哪个文件"]):
        return (
            "酒店相关材料目前是：\n"
            f"- 酒店发票：{_slot_text('hotel_invoice')}\n"
            f"- 酒店支付记录：{_slot_text('hotel_payment')}\n"
            f"- 酒店订单截图：{_slot_text('hotel_order')}\n"
            f"- 金额核对：{_amount_text('hotel_invoice_amount')} vs {_amount_text('hotel_payment_amount')}"
        )

    if ("返程" in text or "回程" in text) and any(key in text for key in ["支付", "票据", "明细", "对应", "哪个文件"]):
        return (
            "返程相关材料目前是：\n"
            f"- 返程票据：{_slot_text('return_ticket')}\n"
            f"- 返程支付：{_slot_text('return_payment')}\n"
            f"- 返程明细：{_slot_text('return_detail')}\n"
            f"- 金额核对：{_amount_text('return_ticket_amount')} vs {_amount_text('return_payment_amount')}"
        )

    if "去程" in text and any(key in text for key in ["支付", "票据", "明细", "对应", "哪个文件"]):
        return (
            "去程相关材料目前是：\n"
            f"- 去程票据：{_slot_text('go_ticket')}\n"
            f"- 去程支付：{_slot_text('go_payment')}\n"
            f"- 去程明细：{_slot_text('go_detail')}\n"
            f"- 金额核对：{_amount_text('go_ticket_amount')} vs {_amount_text('go_payment_amount')}"
        )

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

    if profiles and re.search(r"\.(pdf|jpg|jpeg|png|webp)\b", lower) and any(
        key in text for key in ["去程", "返程", "明细", "票据", "发票", "支付", "订单"]
    ):
        matched = _match_profiles_by_user_text(text, profiles)
        if matched:
            profile = matched[0]
            return (
                f"{profile.get('name')} 当前是 {_doc_type_label(str(profile.get('doc_type') or 'unknown'))}"
                f" / {_slot_label(str(profile.get('manual_slot') or profile.get('slot') or 'unknown'))}。"
                "如果你想直接改，我会按你的说法立即调整。"
            )

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
    metric_scope = "travel"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _chat_model()

    system_prompt = (
        "你是差旅报销Agent。基于给定上下文回答用户问题，禁止编造未提供的信息。"
        "优先回答：缺件、金额核对问题、分配结果、下一步补件建议。"
        "可以引用制度片段或历史样例，但只能使用已提供RAG上下文。"
        "如果用户是问候或寒暄，先自然回复1-2句，再给下一步建议。"
        "不要重复欢迎语，不要复述“已进入模式”等固定句。"
        "中文回答，尽量自然；仅在清单/对账场景使用要点列表。"
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
        if content:
            _record_llm_outcome(metric_scope, True)
            return content
    except Exception:
        _record_llm_outcome(metric_scope, False)
        return None

    _record_llm_outcome(metric_scope, False)
    return None




# Public aliases used by Streamlit shell and split UI modules.
build_travel_execution_payload = _build_travel_execution_payload
safe_float = _safe_float
format_amount = _format_amount
extract_pdf_text_from_bytes = _extract_pdf_text_from_bytes
as_uploaded_list = _as_uploaded_list
files_signature = _files_signature
uploaded_file_key = _uploaded_file_key
profile_file_key = _profile_file_key
render_included_file_list = _render_included_file_list
apply_manual_overrides_to_profiles = _apply_manual_overrides_to_profiles
apply_manual_slot_overrides_to_profiles = _apply_manual_slot_overrides_to_profiles
prune_manual_overrides = _prune_manual_overrides
prune_manual_slot_overrides = _prune_manual_slot_overrides
build_travel_file_profile = _build_travel_file_profile
build_assignment_from_profiles = _build_assignment_from_profiles
slot_label = _slot_label
doc_type_label = _doc_type_label
build_travel_agent_status = _build_travel_agent_status
travel_scope_name = _travel_scope_name
travel_undo_stack_key = _travel_undo_stack_key
clone_travel_profile = _clone_travel_profile
travel_push_undo_snapshot = _travel_push_undo_snapshot
travel_pop_undo_snapshot = _travel_pop_undo_snapshot
travel_restore_undo_snapshot = _travel_restore_undo_snapshot
travel_pending_action_spec_from_text = _travel_pending_action_spec_from_text
append_travel_pending_action_from_spec = _append_travel_pending_action_from_spec
travel_execute_pending_action = _travel_execute_pending_action
merge_uploaded_lists = _merge_uploaded_lists
generate_travel_agent_reply_rule = _generate_travel_agent_reply_rule
build_travel_handoff_status_reply = _build_travel_handoff_status_reply
generate_travel_agent_reply_llm = _generate_travel_agent_reply_llm
render_travel_transport_section = _render_travel_transport_section
render_travel_hotel_section = _render_travel_hotel_section
render_travel_summary = _render_travel_summary
