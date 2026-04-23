from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TravelChatIntent(str, Enum):
    ASK_MISSING = "ask_missing"
    ASK_FILE_LIST = "ask_file_list"
    ASK_FILE_COUNT = "ask_file_count"
    ASK_MISMATCH = "ask_mismatch"
    ASK_NEXT_STEP = "ask_next_step"
    ASK_REQUIREMENT = "ask_requirement"
    CLARIFY = "clarify"


class TravelChatContext(BaseModel):
    last_intent: str | None = Field(default=None)
    last_scope: str | None = Field(default=None)
    last_target_slot: str | None = Field(default=None)
    last_answer_type: str | None = Field(default=None)


class TravelChatQuery(BaseModel):
    intent: TravelChatIntent = Field(default=TravelChatIntent.CLARIFY)
    message: str = Field(default="")
    normalized_message: str = Field(default="")
    resolved_message: str = Field(default="")
    scope: str | None = Field(default=None)
    doc_kind: str | None = Field(default=None)
    target_slots: list[str] = Field(default_factory=list)
    reason: str = Field(default="")
    score_map: dict[str, int] = Field(default_factory=dict)


_SCOPE_SLOT_GROUPS: dict[str, list[str]] = {
    "go": ["go_ticket", "go_payment", "go_detail"],
    "return": ["return_ticket", "return_payment", "return_detail"],
    "hotel": ["hotel_invoice", "hotel_payment", "hotel_order"],
}

_SLOT_LABELS: dict[str, str] = {
    "go_ticket": "去程票据",
    "go_payment": "去程支付",
    "go_detail": "去程明细",
    "return_ticket": "返程票据",
    "return_payment": "返程支付",
    "return_detail": "返程明细",
    "hotel_invoice": "酒店发票",
    "hotel_payment": "酒店支付",
    "hotel_order": "酒店订单截图",
}

FILLER_WORDS = [
    "现在",
    "目前",
    "麻烦",
    "请问",
    "帮我看下",
    "帮我看一下",
    "帮我看看",
    "帮看下",
    "帮看一下",
]
FOLLOWUP_WORDS = ["还有", "还有呢", "还有吗", "然后", "然后呢", "那呢", "还有啥", "还有什么"]
ISSUE_WORDS = ["对不上", "不一致", "不对", "异常", "问题", "差多少", "差额", "核对", "金额"]
MISSING_WORDS = ["缺", "还缺", "还差", "补", "补齐", "齐不齐", "全不全", "差什么", "差哪些"]
NEXT_STEP_WORDS = ["下一步", "怎么做", "咋办", "怎么弄", "接下来", "现在做什么", "该做什么"]
REQUIREMENT_WORDS = ["材料有哪些", "需要什么材料", "要什么材料", "准备什么材料", "报销要求", "材料清单", "清单"]
FILE_LIST_WORDS = ["哪几个文件", "哪些文件", "文件有哪些", "分别是哪些", "文件名", "哪个文件", "是哪些文件", "对应文件"]
FILE_COUNT_WORDS = ["几份", "几个", "多少份", "多少个", "几张", "数量", "有几份", "有几个", "几份材料"]


def ensure_travel_chat_context(value: Any) -> TravelChatContext:
    if isinstance(value, TravelChatContext):
        return value
    if isinstance(value, dict):
        try:
            return TravelChatContext.model_validate(value)
        except Exception:
            return TravelChatContext()
    return TravelChatContext()


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(key in text for key in keywords)


def normalize_travel_chat_message(message: str) -> str:
    text = _norm_text(message)
    for word in FILLER_WORDS:
        text = text.replace(_norm_text(word), "")
    text = re.sub(r"[，,。.!！？?；;：:~～、\-\(\)\[\]\"“”'‘’]+", "", text)
    return text


def resolve_followup_message(message: str, ctx: TravelChatContext) -> str:
    text = normalize_travel_chat_message(message)
    if not text:
        return text

    if _contains_any(text, ISSUE_WORDS):
        return text

    if text in FOLLOWUP_WORDS:
        if ctx.last_intent == TravelChatIntent.ASK_MISMATCH.value:
            return "哪些金额对不上"
        if ctx.last_intent == TravelChatIntent.ASK_MISSING.value:
            return "还缺什么"
        if ctx.last_intent == TravelChatIntent.ASK_NEXT_STEP.value:
            return "下一步做什么"
        if ctx.last_intent == TravelChatIntent.ASK_FILE_LIST.value:
            return "哪几个文件"
        if ctx.last_intent == TravelChatIntent.ASK_FILE_COUNT.value:
            return "有几份"
        return text

    if text in {"酒店", "酒店呢"} and ctx.last_intent == TravelChatIntent.ASK_MISMATCH.value:
        return "酒店哪些金额对不上"
    if text in {"返程", "返程呢", "回程", "回程呢"} and ctx.last_intent == TravelChatIntent.ASK_MISMATCH.value:
        return "返程哪些金额对不上"
    if text in {"去程", "去程呢"} and ctx.last_intent == TravelChatIntent.ASK_MISMATCH.value:
        return "去程哪些金额对不上"

    return text


def _detect_scope(text: str, ctx: TravelChatContext | None = None) -> str | None:
    if _contains_any(text, ["酒店", "住宿", "房费"]):
        return "hotel"
    if _contains_any(text, ["返程", "回程"]):
        return "return"
    if _contains_any(text, ["去程", "出发"]):
        return "go"
    if ctx and ctx.last_scope in {"hotel", "go", "return"}:
        return str(ctx.last_scope)
    return None


def _detect_doc_kind(text: str, scope: str | None) -> str | None:
    if _contains_any(text, ["订单", "订单截图"]):
        return "order"
    if _contains_any(text, ["支付", "付款", "流水"]):
        return "payment"
    if _contains_any(text, ["明细", "行程单"]):
        return "detail"
    if _contains_any(text, ["发票"]):
        return "invoice" if scope == "hotel" else "ticket"
    if _contains_any(text, ["票据", "票证", "票"]):
        return "invoice" if scope == "hotel" else "ticket"
    return None


def _resolve_slots(scope: str | None, doc_kind: str | None) -> list[str]:
    if scope == "hotel":
        if doc_kind == "invoice":
            return ["hotel_invoice"]
        if doc_kind == "payment":
            return ["hotel_payment"]
        if doc_kind == "order":
            return ["hotel_order"]
        return list(_SCOPE_SLOT_GROUPS["hotel"])

    if scope == "go":
        if doc_kind == "ticket":
            return ["go_ticket"]
        if doc_kind == "payment":
            return ["go_payment"]
        if doc_kind == "detail":
            return ["go_detail"]
        return list(_SCOPE_SLOT_GROUPS["go"])

    if scope == "return":
        if doc_kind == "ticket":
            return ["return_ticket"]
        if doc_kind == "payment":
            return ["return_payment"]
        if doc_kind == "detail":
            return ["return_detail"]
        return list(_SCOPE_SLOT_GROUPS["return"])

    return []


def _score_intents(
    text: str,
    *,
    scope: str | None,
    doc_kind: str | None,
    ctx: TravelChatContext,
) -> dict[str, int]:
    scores = {
        TravelChatIntent.ASK_MISSING.value: 0,
        TravelChatIntent.ASK_FILE_LIST.value: 0,
        TravelChatIntent.ASK_FILE_COUNT.value: 0,
        TravelChatIntent.ASK_MISMATCH.value: 0,
        TravelChatIntent.ASK_NEXT_STEP.value: 0,
        TravelChatIntent.ASK_REQUIREMENT.value: 0,
    }

    if _contains_any(text, ["对不上", "不一致", "差额", "差多少"]):
        scores[TravelChatIntent.ASK_MISMATCH.value] += 3
    if _contains_any(text, ISSUE_WORDS):
        scores[TravelChatIntent.ASK_MISMATCH.value] += 2
    if _contains_any(text, ["金额", "核对"]):
        scores[TravelChatIntent.ASK_MISMATCH.value] += 1
    if ctx.last_intent == TravelChatIntent.ASK_MISMATCH.value and _contains_any(text, ["还有", "哪里", "什么", "哪儿"]):
        scores[TravelChatIntent.ASK_MISMATCH.value] += 1

    if _contains_any(text, ["还缺", "还差", "缺什么", "缺哪些", "补什么", "齐不齐", "全不全"]):
        scores[TravelChatIntent.ASK_MISSING.value] += 3
    if _contains_any(text, MISSING_WORDS):
        scores[TravelChatIntent.ASK_MISSING.value] += 1

    if _contains_any(text, FILE_COUNT_WORDS):
        scores[TravelChatIntent.ASK_FILE_COUNT.value] += 3
    if _contains_any(text, ["多少", "几"]) and (doc_kind is not None or scope is not None):
        scores[TravelChatIntent.ASK_FILE_COUNT.value] += 1

    if _contains_any(text, FILE_LIST_WORDS):
        scores[TravelChatIntent.ASK_FILE_LIST.value] += 3
    if _contains_any(text, ["文件", "文件名"]):
        scores[TravelChatIntent.ASK_FILE_LIST.value] += 1
    if doc_kind is not None and _contains_any(text, ["哪些", "哪几个", "哪个"]):
        scores[TravelChatIntent.ASK_FILE_LIST.value] += 1

    if _contains_any(text, NEXT_STEP_WORDS):
        scores[TravelChatIntent.ASK_NEXT_STEP.value] += 3
    if _contains_any(text, ["咋办", "怎么弄", "怎么处理"]):
        scores[TravelChatIntent.ASK_NEXT_STEP.value] += 1
    if "做什么" in text and (scope is not None or _contains_any(text, ["接下来", "下一步", "现在"])):
        scores[TravelChatIntent.ASK_NEXT_STEP.value] += 2

    if _contains_any(text, REQUIREMENT_WORDS):
        scores[TravelChatIntent.ASK_REQUIREMENT.value] += 3
    if _contains_any(text, ["要求", "清单", "材料"]) and _contains_any(text, ["什么", "哪些"]):
        scores[TravelChatIntent.ASK_REQUIREMENT.value] += 1

    if scores[TravelChatIntent.ASK_FILE_COUNT.value] > 0:
        scores[TravelChatIntent.ASK_FILE_LIST.value] -= 1
    if scores[TravelChatIntent.ASK_FILE_LIST.value] > 0:
        scores[TravelChatIntent.ASK_FILE_COUNT.value] -= 1

    return scores


def parse_travel_chat_query(user_message: str, context: dict[str, Any] | None = None) -> TravelChatQuery:
    context_dict = dict(context or {})
    chat_ctx = ensure_travel_chat_context(context_dict.get("chat_context"))
    text = str(user_message or "").strip()
    normalized = normalize_travel_chat_message(text)
    resolved = resolve_followup_message(normalized, chat_ctx)

    if not normalized:
        return TravelChatQuery(
            intent=TravelChatIntent.CLARIFY,
            message=text,
            normalized_message=normalized,
            resolved_message=resolved,
            reason="empty_message",
        )

    scope = _detect_scope(resolved, chat_ctx)
    doc_kind = _detect_doc_kind(resolved, scope)
    slots = _resolve_slots(scope, doc_kind)
    scores = _score_intents(resolved, scope=scope, doc_kind=doc_kind, ctx=chat_ctx)
    best_intent = max(scores.keys(), key=lambda key: scores[key])
    best_score = int(scores.get(best_intent) or 0)

    if best_score < 2:
        return TravelChatQuery(
            intent=TravelChatIntent.CLARIFY,
            message=text,
            normalized_message=normalized,
            resolved_message=resolved,
            scope=scope,
            doc_kind=doc_kind,
            target_slots=slots,
            reason=f"low_confidence_score:{best_score}",
            score_map=dict(scores),
        )

    if best_intent == TravelChatIntent.ASK_FILE_LIST.value and not slots:
        return TravelChatQuery(
            intent=TravelChatIntent.CLARIFY,
            message=text,
            normalized_message=normalized,
            resolved_message=resolved,
            scope=scope,
            doc_kind=doc_kind,
            reason="file_list_without_target",
            score_map=dict(scores),
        )

    if best_intent == TravelChatIntent.ASK_FILE_COUNT.value and not slots:
        return TravelChatQuery(
            intent=TravelChatIntent.CLARIFY,
            message=text,
            normalized_message=normalized,
            resolved_message=resolved,
            scope=scope,
            doc_kind=doc_kind,
            reason="file_count_without_target",
            score_map=dict(scores),
        )

    return TravelChatQuery(
        intent=TravelChatIntent(best_intent),
        message=text,
        normalized_message=normalized,
        resolved_message=resolved,
        scope=scope,
        doc_kind=doc_kind,
        target_slots=slots,
        reason=f"scored_match:{best_intent}",
        score_map=dict(scores),
    )


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip()
        text = text.replace("¥", "").replace("￥", "").replace(",", "").replace("，", "")
        text = re.sub(r"[^\d.\-]", "", text)
        if text in {"", "-", ".", "-."}:
            return None
        return float(text)
    except Exception:
        return None


def _basename(value: str) -> str:
    source = str(value or "").strip()
    if not source:
        return ""
    parts = re.split(r"[\\/]+", source)
    return parts[-1] if parts else source


def _extract_file_name(item: Any) -> str:
    if isinstance(item, str):
        return _basename(item)
    if isinstance(item, dict):
        for key in ["name", "filename", "file_name", "path"]:
            value = item.get(key)
            if value:
                return _basename(str(value))
        return ""
    for attr in ["name", "filename", "file_name", "path"]:
        value = getattr(item, attr, None)
        if value:
            return _basename(str(value))
    return ""


def _extract_slot_files(assignment: dict[str, Any], slot: str) -> list[str]:
    raw = assignment.get(slot)
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        name = _extract_file_name(item)
        if name:
            names.append(name)
    return names


def _scope_from_text(text: str) -> str:
    normalized = _norm_text(text)
    if "酒店" in normalized or "住宿" in normalized:
        return "hotel"
    if "返程" in normalized or "回程" in normalized:
        return "return"
    if "去程" in normalized:
        return "go"
    return "unknown"


def _slot_refs_for_scope(assignment: dict[str, Any], scope: str) -> list[str]:
    mapping = {
        "go": ["go_ticket", "go_payment"],
        "return": ["return_ticket", "return_payment"],
        "hotel": ["hotel_invoice", "hotel_payment"],
    }
    refs: list[str] = []
    for slot in mapping.get(scope, []):
        refs.extend(_extract_slot_files(assignment, slot))
    return refs[:8]


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _build_amount_mismatches(assignment: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("go", "去程", "go_ticket_amount", "go_payment_amount"),
        ("return", "返程", "return_ticket_amount", "return_payment_amount"),
        ("hotel", "酒店", "hotel_invoice_amount", "hotel_payment_amount"),
    ]
    mismatches: list[dict[str, Any]] = []
    for scope, label, left_key, right_key in checks:
        left_value = _safe_float(assignment.get(left_key))
        right_value = _safe_float(assignment.get(right_key))
        if left_value is None or right_value is None:
            continue
        if abs(left_value - right_value) <= 0.01:
            continue
        mismatches.append(
            {
                "scope": scope,
                "label": label,
                "left_key": left_key,
                "right_key": right_key,
                "left": left_value,
                "right": right_value,
                "diff": left_value - right_value,
            }
        )
    return mismatches


def _build_issue_items_from_assignment(assignment: dict[str, Any]) -> list[dict[str, Any]]:
    label_map = {
        "go": "去程交通票据金额与支付记录金额不一致",
        "return": "返程交通票据金额与支付记录金额不一致",
        "hotel": "酒店票据金额与支付记录金额不一致",
    }
    issue_items: list[dict[str, Any]] = []
    for mismatch in _build_amount_mismatches(assignment):
        scope = str(mismatch.get("scope") or "unknown")
        invoice_amount = _safe_float(mismatch.get("left"))
        payment_amount = _safe_float(mismatch.get("right"))
        if invoice_amount is None or payment_amount is None:
            continue
        issue_items.append(
            {
                "scope": scope,
                "kind": "amount_mismatch",
                "label": label_map.get(scope, str(mismatch.get("label") or "金额不一致")),
                "invoice_amount": invoice_amount,
                "payment_amount": payment_amount,
                "file_refs": _slot_refs_for_scope(assignment, scope),
            }
        )
    return issue_items


def _build_issue_items_from_issues_text(issues: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for issue in issues:
        text = str(issue or "").strip()
        if not text:
            continue
        match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*vs\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if not match:
            continue
        invoice_amount = _safe_float(match.group(1))
        payment_amount = _safe_float(match.group(2))
        if invoice_amount is None or payment_amount is None:
            continue
        items.append(
            {
                "scope": _scope_from_text(text),
                "kind": "amount_mismatch",
                "label": text.split("：", 1)[0] if "：" in text else text,
                "invoice_amount": invoice_amount,
                "payment_amount": payment_amount,
                "file_refs": [],
            }
        )
    return items


def _issue_text_matches_scope(text: str, scope: str | None) -> bool:
    if scope not in {"go", "return", "hotel"}:
        return True
    normalized = _norm_text(text)
    if scope == "go":
        return _contains_any(normalized, ["去程", "出发"])
    if scope == "return":
        return _contains_any(normalized, ["返程", "回程"])
    if scope == "hotel":
        return _contains_any(normalized, ["酒店", "住宿", "房费"])
    return True


def _normalize_issue_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip()
        scope = str(raw.get("scope") or "").strip() or _scope_from_text(label)
        kind = str(raw.get("kind") or "amount_mismatch").strip() or "amount_mismatch"
        invoice_amount = _safe_float(raw.get("invoice_amount"))
        payment_amount = _safe_float(raw.get("payment_amount"))
        file_refs = _normalize_list(raw.get("file_refs"))
        result.append(
            {
                "scope": scope,
                "kind": kind,
                "label": label or "金额不一致",
                "invoice_amount": invoice_amount,
                "payment_amount": payment_amount,
                "file_refs": file_refs,
            }
        )
    return result


def execute_travel_chat_query(
    query: TravelChatQuery,
    assignment: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    assignment_dict = dict(assignment or {})
    status_dict = dict(status or {})
    missing = _normalize_list(status_dict.get("missing"))
    issues = _normalize_list(status_dict.get("issues"))
    issue_items = _normalize_issue_items(status_dict.get("issue_items"))
    if not issue_items:
        issue_items = _build_issue_items_from_assignment(assignment_dict)
    if not issue_items:
        issue_items = _build_issue_items_from_issues_text(issues)

    payload: dict[str, Any] = {
        "intent": str(query.intent.value),
        "query": query.model_dump(),
    }

    if query.intent == TravelChatIntent.ASK_MISSING:
        filtered_missing = list(missing)
        if query.scope == "hotel":
            filtered_missing = [item for item in missing if "酒店" in item or "住宿" in item]
        elif query.scope == "go":
            filtered_missing = [item for item in missing if "去程" in item]
        elif query.scope == "return":
            filtered_missing = [item for item in missing if "返程" in item or "回程" in item]
        payload["missing"] = filtered_missing if query.scope else missing
        return payload

    if query.intent in {TravelChatIntent.ASK_FILE_LIST, TravelChatIntent.ASK_FILE_COUNT}:
        slots = list(query.target_slots or [])
        if not slots:
            payload["intent"] = TravelChatIntent.CLARIFY.value
            payload["clarify_hint"] = "还没定位到具体材料范围，请补充“去程/返程/酒店 + 发票/支付/明细/订单”。"
            return payload
        slot_files: dict[str, list[str]] = {}
        total = 0
        for slot in slots:
            names = _extract_slot_files(assignment_dict, slot)
            slot_files[slot] = names
            total += len(names)
        payload["target_slots"] = slots
        payload["slot_files"] = slot_files
        payload["total_count"] = total
        return payload

    if query.intent == TravelChatIntent.ASK_MISMATCH:
        filtered_issue_items = list(issue_items)
        filtered_issues = list(issues)
        if query.scope in {"go", "return", "hotel"}:
            filtered_issue_items = [item for item in issue_items if str(item.get("scope") or "") == query.scope]
            filtered_issues = [item for item in issues if _issue_text_matches_scope(item, query.scope)]
        payload["issue_items"] = filtered_issue_items
        payload["issues"] = filtered_issues
        return payload

    if query.intent == TravelChatIntent.ASK_NEXT_STEP:
        steps: list[str] = []
        if missing:
            steps.append("先补齐缺件，再做金额核对。")
        if issue_items or issues:
            steps.append("核对不一致的金额项，必要时更正票据或支付归类。")
        if not steps:
            steps.append("材料和金额核对已通过，可直接导出差旅材料。")
        if query.scope == "hotel":
            steps.append("酒店场景重点确认：发票、支付记录、订单截图三项是否齐全。")
        elif query.scope == "go":
            steps.append("去程场景重点确认：票据、支付、明细三项是否对应。")
        elif query.scope == "return":
            steps.append("返程场景重点确认：票据、支付、明细三项是否对应。")
        payload["steps"] = steps
        payload["missing"] = missing
        payload["issue_items"] = list(issue_items)
        payload["issues"] = issues
        return payload

    if query.intent == TravelChatIntent.ASK_REQUIREMENT:
        if query.scope == "hotel":
            requirements = ["酒店发票", "酒店支付记录", "酒店订单截图"]
            title = "酒店材料要求"
        elif query.scope == "go":
            requirements = ["去程交通票据", "去程支付记录", "去程明细（飞机场景）"]
            title = "去程材料要求"
        elif query.scope == "return":
            requirements = ["返程交通票据", "返程支付记录", "返程明细（飞机场景）"]
            title = "返程材料要求"
        else:
            requirements = [
                "去程：交通票据 + 支付记录 + 明细（飞机）",
                "返程：交通票据 + 支付记录 + 明细（飞机）",
                "酒店：发票 + 支付记录 + 订单截图",
            ]
            title = "差旅材料要求"
        payload["title"] = title
        payload["requirements"] = requirements
        return payload

    payload["intent"] = TravelChatIntent.CLARIFY.value
    payload["clarify_hint"] = "我还不确定你想看哪类信息。可以直接问：还缺什么、酒店发票是哪几个文件、哪些金额对不上。"
    return payload


def _format_amount(value: Any) -> str:
    amount = _safe_float(value)
    if amount is None:
        return "无"
    return f"{amount:.2f}"


def _answer_type_for_intent(intent: str) -> str:
    mapping = {
        TravelChatIntent.ASK_MISSING.value: "missing_list",
        TravelChatIntent.ASK_FILE_LIST.value: "file_list",
        TravelChatIntent.ASK_FILE_COUNT.value: "file_count",
        TravelChatIntent.ASK_MISMATCH.value: "issue_list",
        TravelChatIntent.ASK_NEXT_STEP.value: "next_step",
        TravelChatIntent.ASK_REQUIREMENT.value: "requirement",
        TravelChatIntent.CLARIFY.value: "clarify",
    }
    return mapping.get(str(intent or ""), "clarify")


def update_travel_chat_context(
    ctx: TravelChatContext | dict[str, Any] | None,
    query: TravelChatQuery,
    payload: dict[str, Any],
) -> TravelChatContext:
    current = ensure_travel_chat_context(ctx)
    intent = str(query.intent.value)
    if intent == TravelChatIntent.CLARIFY.value:
        return current

    next_ctx = current.model_copy(deep=True)
    next_ctx.last_intent = intent
    next_ctx.last_answer_type = _answer_type_for_intent(intent)
    next_ctx.last_scope = str(query.scope or "") or None
    next_ctx.last_target_slot = str((query.target_slots or [None])[0] or "") or None

    if intent == TravelChatIntent.ASK_MISMATCH.value and not next_ctx.last_scope:
        issue_items = _normalize_issue_items(payload.get("issue_items"))
        unique_scopes = sorted({str(item.get("scope") or "") for item in issue_items if str(item.get("scope") or "")})
        if len(unique_scopes) == 1:
            next_ctx.last_scope = unique_scopes[0]

    return next_ctx


def render_travel_chat_answer(payload: dict[str, Any]) -> str:
    intent = str(payload.get("intent") or TravelChatIntent.CLARIFY.value)

    if intent == TravelChatIntent.ASK_MISSING.value:
        missing = _normalize_list(payload.get("missing"))
        if not missing:
            return "当前缺件已补齐。"
        return "当前还缺：\n- " + "\n- ".join(missing)

    if intent == TravelChatIntent.ASK_FILE_LIST.value:
        slot_files = dict(payload.get("slot_files") or {})
        rows: list[str] = []
        for slot in list(payload.get("target_slots") or []):
            names = [str(item).strip() for item in list(slot_files.get(slot) or []) if str(item).strip()]
            label = _SLOT_LABELS.get(slot, slot)
            rows.append(f"- {label}：{'、'.join(names) if names else '无'}")
        if not rows:
            return "当前没有可展示的文件清单。"
        return "当前文件归属如下：\n" + "\n".join(rows)

    if intent == TravelChatIntent.ASK_FILE_COUNT.value:
        slot_files = dict(payload.get("slot_files") or {})
        total_count = int(payload.get("total_count") or 0)
        detail: list[str] = []
        for slot in list(payload.get("target_slots") or []):
            names = list(slot_files.get(slot) or [])
            label = _SLOT_LABELS.get(slot, slot)
            detail.append(f"- {label}：{len(names)}份")
        if not detail:
            return "当前没有可统计的文件数量。"
        return f"当前共 {total_count} 份：\n" + "\n".join(detail)

    if intent == TravelChatIntent.ASK_MISMATCH.value:
        issue_items = _normalize_issue_items(payload.get("issue_items"))
        if issue_items:
            lines: list[str] = []
            seen: set[tuple[str, str, str]] = set()
            for item in issue_items:
                label = str(item.get("label") or "金额不一致")
                invoice_amount = _format_amount(item.get("invoice_amount"))
                payment_amount = _format_amount(item.get("payment_amount"))
                key = (label, invoice_amount, payment_amount)
                if key in seen:
                    continue
                seen.add(key)
                refs = _normalize_list(item.get("file_refs"))
                if refs:
                    lines.append(f"- {label}：{invoice_amount} vs {payment_amount}（相关文件：{'、'.join(refs[:4])}）")
                else:
                    lines.append(f"- {label}：{invoice_amount} vs {payment_amount}")
            if lines:
                return "当前金额核对问题：\n" + "\n".join(lines)

        issues = _normalize_list(payload.get("issues"))
        if issues:
            lines: list[str] = []
            seen: set[str] = set()
            for issue in issues:
                label = issue.split("：", 1)[0].strip() if "：" in issue else issue.strip()
                if label in seen:
                    continue
                seen.add(label)
                lines.append(f"- {issue}")
            return "当前金额核对问题：\n" + "\n".join(lines)

        return "当前未发现明确的金额不一致项。"

    if intent == TravelChatIntent.ASK_NEXT_STEP.value:
        steps = _normalize_list(payload.get("steps"))
        if not steps:
            return "当前可继续补充你关心的范围，例如“酒店发票是哪几个文件”。"
        numbered = [f"{idx}. {step}" for idx, step in enumerate(steps, start=1)]
        return "建议下一步：\n" + "\n".join(numbered)

    if intent == TravelChatIntent.ASK_REQUIREMENT.value:
        title = str(payload.get("title") or "差旅材料要求")
        requirements = _normalize_list(payload.get("requirements"))
        if not requirements:
            return "当前没有可展示的材料要求。"
        return f"{title}：\n- " + "\n- ".join(requirements)

    return str(payload.get("clarify_hint") or "我还不确定你具体想问哪一类。你可以直接问：还缺什么、酒店发票是哪几个文件、哪些金额对不上。")
