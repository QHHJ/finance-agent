from __future__ import annotations

import logging
import re
from typing import Any

from app.db import repo
from app.services import extractor, material_fix_agent, parser, rag_retriever, validator

from .state import FinanceGraphState

logger = logging.getLogger("finance.graph")

TRAVEL_KEYWORDS = [
    "差旅",
    "出差",
    "机票",
    "高铁",
    "火车",
    "车票",
    "航班",
    "酒店",
    "住宿",
    "行程",
]

POLICY_KEYWORDS = [
    "制度",
    "规则",
    "规定",
    "政策",
    "报销标准",
    "可以报销",
    "能报吗",
    "policy",
    "faq",
]

TRAVEL_POLICY_TRIGGER_KEYWORDS = [
    "标准",
    "超标",
    "是否可报",
    "是否可以报销",
    "报销规则",
    "补充材料",
    "不一致",
    "异常",
]

POLICY_QUESTION_MARKERS = [
    "请根据",
    "制度说明",
    "规则说明",
    "怎么报销",
    "能否报销",
    "是否可以报销",
    "材料要求",
    "policy qa",
]

MATERIAL_KEYWORDS = [
    "材料",
    "入库",
    "实验室",
    "电子元件",
    "金属制品",
    "规格",
    "型号",
    "数量",
    "单位",
]


def _string(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        text = _string(value)
        text = text.replace(",", "")
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


def _normalize_quantity(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def _to_editor_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _normalize_line_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item_name = _string(row.get("item_name"))
        spec = _string(row.get("spec"))
        quantity = _normalize_quantity(row.get("quantity"))
        unit = _string(row.get("unit"))
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


def _line_items_total(rows: list[dict[str, Any]]) -> float | None:
    values = [_safe_float(row.get("line_total_with_tax")) for row in rows]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers)


def _contains_any(text: str, keywords: list[str]) -> bool:
    source = _string(text).lower()
    if not source:
        return False
    return any(_string(keyword).lower() in source for keyword in keywords)


def _append_agent_trace(
    state: FinanceGraphState,
    *,
    agent: str,
    action: str,
    detail: str,
) -> list[dict[str, Any]]:
    trace = list(state.get("agent_trace") or [])
    trace.append(
        {
            "agent": agent,
            "action": action,
            "detail": detail,
        }
    )
    return trace


def _build_policy_context_text(hits: list[dict[str, Any]], max_len: int = 320) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        title = _string(hit.get("title")) or "policy"
        score = float(hit.get("score") or 0.0)
        content = re.sub(r"\s+", " ", _string(hit.get("content")))
        if len(content) > max_len:
            content = content[: max_len - 3] + "..."
        lines.append(f"[Policy#{idx} score={score:.3f} title={title}] {content}")
    return "\n".join(lines)


def _travel_policy_query(extracted_data: dict[str, Any], raw_text: str) -> str:
    parts = [
        _string(extracted_data.get("bill_type")),
        _string(extracted_data.get("item_content")),
        _string(extracted_data.get("seller")),
        _string(extracted_data.get("buyer")),
        _string(extracted_data.get("amount")),
        _string(raw_text)[:1000],
    ]
    return "\n".join(part for part in parts if part)


def _needs_policy_for_travel(extracted_data: dict[str, Any], raw_text: str, doc_type: str) -> tuple[bool, str]:
    merged = "\n".join(
        [
            _string(raw_text),
            _string(extracted_data.get("item_content")),
            _string(extracted_data.get("bill_type")),
            _string(extracted_data.get("seller")),
        ]
    )
    if doc_type == "unknown":
        return True, "travel_doc_type_unknown"
    if not _string(extracted_data.get("amount")):
        return True, "travel_amount_missing"
    if not _string(extracted_data.get("invoice_date")):
        return True, "travel_date_missing"
    if _contains_any(merged, TRAVEL_POLICY_TRIGGER_KEYWORDS):
        return True, "travel_policy_keyword_hit"
    return False, "travel_no_policy_needed"


def _build_suggestion_data(db, extracted_data: dict[str, Any], raw_text: str) -> dict[str, Any]:
    policies = repo.list_policy_documents(db, limit=100)
    historical_samples = repo.get_historical_samples(
        db,
        extracted_data.get("bill_type"),
        item_content=extracted_data.get("item_content"),
        limit=200,
    )
    return validator.suggest_processing(
        extracted_data=extracted_data,
        raw_text=raw_text,
        policies=policies,
        historical_samples=historical_samples,
    )


def _build_final_data(
    suggestion_data: dict[str, Any],
    extracted_data: dict[str, Any],
    *,
    task_type: str,
    route_reason: str,
    llm_error: str | None = None,
    material_fix_result: dict[str, Any] | None = None,
    travel_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "expense_category": suggestion_data.get("expense_category"),
        "required_materials": suggestion_data.get("required_materials", []),
        "risk_points": suggestion_data.get("risk_points", []),
        "policy_references": suggestion_data.get("policy_references", []),
        "similar_case_refs": suggestion_data.get("similar_case_refs", []),
        "rationale": suggestion_data.get("rationale", ""),
        "rag_trace": suggestion_data.get("rag_trace", {}),
        "extracted_fields": extracted_data,
        "task_type": task_type,
        "route_reason": route_reason,
    }
    if llm_error:
        payload["llm_error"] = llm_error
    if material_fix_result:
        payload["material_fix_result"] = material_fix_result
    if travel_context:
        payload["travel_context"] = travel_context
    return payload


def _guess_task_type(extracted_data: dict[str, Any], raw_text: str) -> tuple[str, str]:
    bill_type = _string(extracted_data.get("bill_type"))
    item_content = _string(extracted_data.get("item_content"))
    seller = _string(extracted_data.get("seller"))
    merged = "\n".join([_string(raw_text), bill_type, item_content, seller])

    has_policy = _contains_any(merged, POLICY_KEYWORDS)
    has_invoice_signal = any(
        [
            _string(extracted_data.get("invoice_number")),
            _string(extracted_data.get("amount")),
            len(_normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))) > 0,
        ]
    )

    has_travel = _contains_any(merged, TRAVEL_KEYWORDS)
    has_material = _contains_any(merged, MATERIAL_KEYWORDS)

    if has_policy and not has_invoice_signal and not has_travel and not has_material:
        return "policy", "policy intent without expense signal"
    if has_policy and not has_invoice_signal and _contains_any(merged, POLICY_QUESTION_MARKERS):
        return "policy", "policy question intent"

    if has_travel:
        return "travel", "hit travel keywords"
    if has_material:
        return "material", "hit material keywords"

    line_items = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    if len(line_items) >= 2:
        return "material", "multiple line items"

    if has_policy:
        return "policy", "hit policy keywords"

    return "generic", "fallback generic"


def _guess_travel_doc_type(extracted_data: dict[str, Any], raw_text: str) -> str:
    merged = "\n".join(
        [
            _string(raw_text),
            _string(extracted_data.get("item_content")),
            _string(extracted_data.get("bill_type")),
            _string(extracted_data.get("seller")),
        ]
    )
    if _contains_any(merged, ["酒店", "住宿", "旅店", "携程", "同程", "飞猪"]):
        return "hotel_invoice"
    if _contains_any(merged, ["机票", "航班", "高铁", "火车", "车票", "客运", "打车", "滴滴", "出租车"]):
        return "transport_ticket"
    return "unknown"


def load_task_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    if not task:
        raise ValueError(f"Task not found: {state['task_id']}")
    repo.set_task_status(db, task.id, "processing")
    return {}


def parse_pdf_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    if not task:
        raise ValueError(f"Task not found: {state['task_id']}")
    raw_text = parser.parse_file_text(task.stored_path)
    return {"raw_text": raw_text}


def extract_fields_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    pdf_path = task.stored_path if task else None
    extracted_data = extractor.extract_invoice_fields(state.get("raw_text", ""), pdf_path=pdf_path)
    return {"extracted_data": extracted_data}


def supervisor_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    raw_text = state.get("raw_text", "")
    task_type, reason = _guess_task_type(extracted_data, raw_text)

    if task_type == "travel":
        logger.info("supervisor -> travel | reason=%s", reason)
    elif task_type == "policy":
        logger.info("supervisor -> policy | reason=%s", reason)
    elif task_type == "material":
        logger.info("supervisor -> material | reason=%s", reason)
    else:
        logger.info("supervisor -> generic | reason=%s", reason)

    return {
        "task_type": task_type,
        "route_reason": reason,
        "next_action": "persist",
        "agent_trace": _append_agent_trace(
            state,
            agent="supervisor",
            action=f"route:{task_type}",
            detail=reason,
        ),
    }


def classify_task_node(state: FinanceGraphState) -> FinanceGraphState:
    # Backward-compatible alias for existing callers.
    return supervisor_node(state)


def _material_needs_repair(extracted_data: dict[str, Any]) -> tuple[bool, str]:
    rows = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    if not rows:
        return False, "no_line_items"

    suspicious_count = 0
    for row in rows:
        name = _string(row.get("item_name"))
        spec = _string(row.get("spec"))
        qty = _string(row.get("quantity"))
        line_total = _string(row.get("line_total_with_tax"))

        if not name or not qty or not line_total:
            suspicious_count += 1
        if spec and name and spec in name:
            suspicious_count += 1
        if len(name) >= 38 and not spec:
            suspicious_count += 1

    if suspicious_count > 0:
        return True, f"suspicious_rows={suspicious_count}"
    return False, "table_looks_clean"


def _material_confidence_from_repair(
    *,
    review_items: list[dict[str, Any]],
    llm_error: str | None,
    stats: dict[str, Any] | None = None,
) -> float:
    if llm_error:
        return 0.45
    stats = dict(stats or {})
    if review_items:
        confidences: list[float] = []
        for item in review_items:
            try:
                value = float(item.get("confidence"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0:
                confidences.append(value)
        if confidences:
            return max(0.35, min(0.9, sum(confidences) / len(confidences)))
        return 0.62

    auto_fixed = int(stats.get("auto_fixed_rows") or 0)
    if auto_fixed > 0:
        return 0.88
    return 0.93


def _build_material_output(
    state: FinanceGraphState,
    *,
    extracted_data: dict[str, Any],
    material_fix_result: dict[str, Any],
    review_items: list[dict[str, Any]],
    confidence: float,
    llm_error: str | None,
    repaired_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    db = state["db"]
    suggestion_data = _build_suggestion_data(db, extracted_data, state.get("raw_text", ""))
    if review_items:
        risk_points = list(suggestion_data.get("risk_points") or [])
        risk_points.append(f"detected {len(review_items)} low-confidence rows; manual review suggested")
        suggestion_data["risk_points"] = risk_points
    suggestion_data["route_reason"] = state.get("route_reason", "")
    suggestion_data["task_type"] = "material"
    suggestion_data["confidence"] = confidence

    fix_payload = dict(material_fix_result or {})
    fix_payload["confidence"] = confidence

    final_data = _build_final_data(
        suggestion_data,
        extracted_data,
        task_type="material",
        route_reason=state.get("route_reason", ""),
        llm_error=llm_error,
        material_fix_result=fix_payload,
    )
    final_data["confidence"] = confidence
    if repaired_data:
        final_data["repaired_data"] = repaired_data

    payload: dict[str, Any] = {
        "suggestion_data": suggestion_data,
        "final_data": final_data,
        "material_fix_result": fix_payload,
        "review_items": review_items,
        "confidence": confidence,
    }
    if repaired_data:
        payload["repaired_data"] = repaired_data
    if llm_error:
        payload["llm_error"] = llm_error
    return payload


def material_agent_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    rows = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    extracted_data["line_items"] = rows

    if not _string(extracted_data.get("amount")):
        total = _line_items_total(rows)
        if total is not None:
            extracted_data["amount"] = _format_amount(total)

    needs_repair, reason = _material_needs_repair(extracted_data)
    next_action = "repair" if needs_repair else "persist"
    confidence = 0.66 if needs_repair else 0.9

    logger.info("material -> %s | reason=%s", next_action, reason)
    payload: FinanceGraphState = {
        "extracted_data": extracted_data,
        "next_action": next_action,
        "material_fix_result": {
            "status": "prepared",
            "line_items": len(rows),
            "needs_repair": needs_repair,
            "decision_reason": reason,
        },
        "confidence": confidence,
        "agent_trace": _append_agent_trace(
            state,
            agent="material_agent",
            action=f"next:{next_action}",
            detail=reason,
        ),
    }

    if not needs_repair:
        payload.update(
            _build_material_output(
                state,
                extracted_data=extracted_data,
                material_fix_result=payload["material_fix_result"],
                review_items=[],
                confidence=confidence,
                llm_error=None,
                repaired_data={"line_items": list(rows)},
            )
        )
    return payload


def repair_agent_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    rows = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    base_fix_result = dict(state.get("material_fix_result", {}) or {})
    if not rows:
        confidence = 0.9
        material_fix_result = {
            **base_fix_result,
            "status": "skipped",
            "reason": "no_line_items",
        }
        payload = _build_material_output(
            state,
            extracted_data=extracted_data,
            material_fix_result=material_fix_result,
            review_items=[],
            confidence=confidence,
            llm_error=None,
            repaired_data={"line_items": []},
        )
        payload["next_action"] = "persist"
        payload["agent_trace"] = _append_agent_trace(
            state,
            agent="repair_agent",
            action="next:persist",
            detail="no_line_items",
        )
        logger.info("repair -> persist | reason=no_line_items")
        return payload

    header_context = {
        "invoice_number": extracted_data.get("invoice_number"),
        "invoice_date": extracted_data.get("invoice_date"),
        "bill_type": extracted_data.get("bill_type"),
        "item_content": extracted_data.get("item_content"),
        "seller": extracted_data.get("seller"),
        "buyer": extracted_data.get("buyer"),
        "amount": extracted_data.get("amount"),
    }
    raw_text = _string(state.get("raw_text", ""))

    llm_error: str | None = None
    try:
        result = material_fix_agent.run_llm_row_repair(
            rows,
            header_context=header_context,
            raw_text=raw_text,
        )
    except Exception as exc:
        llm_error = str(exc)
        result = {"rows": list(rows), "review_items": [], "stats": {}, "llm_error": llm_error}

    repaired_rows = _normalize_line_items(_to_editor_rows(result.get("rows"))) or list(rows)
    review_items = list(result.get("review_items") or [])
    llm_stats = dict(result.get("stats") or {})
    llm_error = llm_error or (_string(result.get("llm_error")) or None)

    output_extracted = dict(extracted_data)
    output_extracted["rule_line_items_baseline"] = _normalize_line_items(
        _to_editor_rows(extracted_data.get("rule_line_items_baseline"))
    ) or list(rows)
    output_extracted["llm_line_items_suggested"] = repaired_rows
    output_extracted["llm_agent_stats"] = llm_stats
    output_extracted["line_items"] = repaired_rows
    total = _line_items_total(repaired_rows)
    if total is not None:
        output_extracted["amount"] = _format_amount(total)
    if llm_error:
        output_extracted["llm_error"] = llm_error

    confidence = _material_confidence_from_repair(
        review_items=review_items,
        llm_error=llm_error,
        stats=llm_stats,
    )
    material_fix_result = {
        **base_fix_result,
        "status": "ok" if not llm_error else "failed",
        "stats": llm_stats,
        "review_items": len(review_items),
    }
    payload = _build_material_output(
        state,
        extracted_data=output_extracted,
        material_fix_result=material_fix_result,
        review_items=review_items,
        confidence=confidence,
        llm_error=llm_error,
        repaired_data={"line_items": repaired_rows},
    )
    payload["extracted_data"] = output_extracted
    payload["next_action"] = "persist"
    payload["agent_trace"] = _append_agent_trace(
        state,
        agent="repair_agent",
        action="next:persist",
        detail="repair_complete",
    )
    logger.info("repair -> persist | review_items=%s", len(review_items))
    return payload


def material_prepare_node(state: FinanceGraphState) -> FinanceGraphState:
    # Backward-compatible alias.
    return material_agent_node(state)


def material_repair_node(state: FinanceGraphState) -> FinanceGraphState:
    # Backward-compatible alias.
    return repair_agent_node(state)


def material_validate_node(state: FinanceGraphState) -> FinanceGraphState:
    # Backward-compatible alias. Keep old API shape for callers that still use it.
    extracted_data = dict(state.get("extracted_data", {}) or {})
    material_fix_result = dict(state.get("material_fix_result", {}) or {})
    review_items = list(state.get("review_items") or [])
    confidence = float(state.get("confidence") or 0.72)
    repaired_data = dict(state.get("repaired_data", {}) or {})
    return _build_material_output(
        state,
        extracted_data=extracted_data,
        material_fix_result=material_fix_result,
        review_items=review_items,
        confidence=confidence,
        llm_error=state.get("llm_error"),
        repaired_data=repaired_data or None,
    )


def travel_prepare_node(state: FinanceGraphState) -> FinanceGraphState:
    raw_text = _string(state.get("raw_text", ""))
    travel_context = {
        "raw_text_preview": raw_text[:600],
        "query_seed": _travel_policy_query(dict(state.get("extracted_data", {}) or {}), raw_text),
    }
    return {"travel_context": travel_context}


def travel_assign_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    travel_context = dict(state.get("travel_context", {}) or {})
    doc_type = _guess_travel_doc_type(extracted_data, state.get("raw_text", ""))
    doc_category = "transport"
    if doc_type == "hotel_invoice":
        doc_category = "hotel"
    elif doc_type == "unknown":
        doc_category = "unknown"
    elif _contains_any(_string(state.get("raw_text", "")), ["打车", "滴滴", "出租车", "网约车"]):
        doc_category = "taxi"
    elif _contains_any(_string(state.get("raw_text", "")), ["高铁", "火车", "铁路"]):
        doc_category = "rail"
    elif _contains_any(_string(state.get("raw_text", "")), ["机票", "航班", "航空"]):
        doc_category = "flight"

    trip_merge_key = "|".join(
        [
            _string(extracted_data.get("invoice_date")) or "date_na",
            _string(extracted_data.get("seller")) or "seller_na",
            _string(doc_category) or "category_na",
        ]
    )

    travel_context.update(
        {
            "doc_type_hint": doc_type,
            "doc_category": doc_category,
            "trip_merge_key": trip_merge_key,
            "amount": extracted_data.get("amount"),
            "invoice_date": extracted_data.get("invoice_date"),
            "seller": extracted_data.get("seller"),
        }
    )
    return {"travel_context": travel_context}


def travel_qa_context_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    extracted_data = dict(state.get("extracted_data", {}) or {})
    travel_context = dict(state.get("travel_context", {}) or {})
    suggestion_data = _build_suggestion_data(db, extracted_data, state.get("raw_text", ""))
    suggestion_data["travel_context"] = travel_context
    suggestion_data["route_reason"] = state.get("route_reason", "")
    suggestion_data["task_type"] = "travel"
    final_data = _build_final_data(
        suggestion_data,
        extracted_data,
        task_type="travel",
        route_reason=state.get("route_reason", ""),
        llm_error=state.get("llm_error"),
        travel_context=travel_context,
    )
    return {
        "suggestion_data": suggestion_data,
        "final_data": final_data,
    }


def travel_agent_node(state: FinanceGraphState) -> FinanceGraphState:
    logger.info("travel_agent start | task_id=%s", state.get("task_id"))

    travel_payload: FinanceGraphState = {}
    merged_state = dict(state)

    prepared = travel_prepare_node(merged_state)
    merged_state.update(prepared)
    travel_payload.update(prepared)

    assigned = travel_assign_node(merged_state)
    merged_state.update(assigned)
    travel_payload.update(assigned)

    qa_payload = travel_qa_context_node(merged_state)
    merged_state.update(qa_payload)
    travel_payload.update(qa_payload)

    extracted_data = dict(merged_state.get("extracted_data", {}) or {})
    travel_context = dict(merged_state.get("travel_context", {}) or {})
    doc_type = _string(travel_context.get("doc_type_hint")) or "unknown"
    needs_policy, reason = _needs_policy_for_travel(
        extracted_data,
        _string(merged_state.get("raw_text", "")),
        doc_type,
    )
    next_action = "policy" if needs_policy else "persist"
    travel_context["needs_policy"] = needs_policy
    travel_context["policy_decision_reason"] = reason
    travel_context["next_action"] = next_action
    travel_payload["travel_context"] = travel_context
    travel_payload["needs_policy"] = needs_policy
    travel_payload["next_action"] = next_action
    travel_payload["agent_trace"] = _append_agent_trace(
        merged_state,
        agent="travel_agent",
        action=f"next:{next_action}",
        detail=reason,
    )

    logger.info(
        "travel_agent -> %s | reason=%s | doc_type=%s",
        next_action,
        reason,
        doc_type,
    )
    return travel_payload


def policy_agent_node(state: FinanceGraphState) -> FinanceGraphState:
    task_type = _string(state.get("task_type")) or "generic"
    raw_text = _string(state.get("raw_text", ""))
    extracted_data = dict(state.get("extracted_data", {}) or {})
    suggestion_data = dict(state.get("suggestion_data", {}) or {})
    travel_context = dict(state.get("travel_context", {}) or {})

    if not suggestion_data:
        suggestion_data = _build_suggestion_data(state["db"], extracted_data, raw_text)

    if task_type == "travel":
        query = _travel_policy_query(extracted_data, raw_text)
        policy_context = rag_retriever.build_travel_policy_context(query, top_k=5)
        if not policy_context.strip():
            policy_hits = rag_retriever.retrieve_policy_hits(query, top_k=5)
            policy_context = _build_policy_context_text(policy_hits)
    else:
        query = raw_text or _travel_policy_query(extracted_data, raw_text)
        policy_hits = rag_retriever.retrieve_policy_hits(query, top_k=5)
        policy_context = _build_policy_context_text(policy_hits)

    suggestion_data["task_type"] = "travel" if task_type == "travel" else "policy"
    suggestion_data["route_reason"] = state.get("route_reason", "")
    suggestion_data["policy_context"] = policy_context
    if policy_context:
        risk_points = list(suggestion_data.get("risk_points") or [])
        risk_points.append("policy_agent attached policy context")
        suggestion_data["risk_points"] = risk_points

    if task_type == "travel":
        travel_context["policy_context"] = policy_context

    final_data = _build_final_data(
        suggestion_data,
        extracted_data,
        task_type="travel" if task_type == "travel" else "policy",
        route_reason=state.get("route_reason", ""),
        llm_error=state.get("llm_error"),
        material_fix_result=dict(state.get("material_fix_result", {}) or {}),
        travel_context=travel_context if task_type == "travel" else None,
    )
    if policy_context:
        final_data["policy_context"] = policy_context

    logger.info("policy -> persist | task_type=%s", task_type)
    return {
        "suggestion_data": suggestion_data,
        "final_data": final_data,
        "policy_context": policy_context,
        "travel_context": travel_context if task_type == "travel" else state.get("travel_context", {}),
        "next_action": "persist",
        "agent_trace": _append_agent_trace(
            state,
            agent="policy_agent",
            action="next:persist",
            detail="policy_context_attached",
        ),
    }


def generic_suggest_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    extracted_data = dict(state.get("extracted_data", {}) or {})
    suggestion_data = _build_suggestion_data(db, extracted_data, state.get("raw_text", ""))
    suggestion_data["route_reason"] = state.get("route_reason", "")
    suggestion_data["task_type"] = "generic"
    final_data = _build_final_data(
        suggestion_data,
        extracted_data,
        task_type="generic",
        route_reason=state.get("route_reason", ""),
        llm_error=state.get("llm_error"),
    )
    return {
        "suggestion_data": suggestion_data,
        "final_data": final_data,
    }


def suggest_node(state: FinanceGraphState) -> FinanceGraphState:
    # Backward-compatible alias.
    return generic_suggest_node(state)


def persist_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.save_processing_result(
        db=db,
        task_id=state["task_id"],
        raw_text=state.get("raw_text", ""),
        extracted_data=state.get("extracted_data", {}),
        suggestion_data=state.get("suggestion_data", {}),
    )

    final_data = dict(state.get("final_data", {}) or {})
    if final_data:
        merged_final = dict(task.final_data or {})
        merged_final.update(final_data)
        task.final_data = merged_final
    db.commit()
    db.refresh(task)
    return {}

