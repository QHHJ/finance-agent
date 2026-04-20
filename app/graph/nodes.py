from __future__ import annotations

import re
from typing import Any

from app.db import repo
from app.services import extractor, material_fix_agent, parser, rag_retriever, validator

from .state import FinanceGraphState

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
        text = text.replace(",", "").replace("￥", "").replace("¥", "")
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

    if _contains_any(merged, TRAVEL_KEYWORDS):
        return "travel", "命中差旅关键词"
    if _contains_any(merged, MATERIAL_KEYWORDS):
        return "material", "命中材料关键词"

    line_items = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    if len(line_items) >= 2:
        return "material", "识别到多条明细，倾向材料票据"

    return "generic", "未命中明显模式，走兜底分支"


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
    if _contains_any(merged, ["机票", "航班", "高铁", "火车", "车票", "客运"]):
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
    raw_text = parser.parse_pdf_text(task.stored_path)
    return {"raw_text": raw_text}


def extract_fields_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    task = repo.get_task(db, state["task_id"])
    pdf_path = task.stored_path if task else None
    extracted_data = extractor.extract_invoice_fields(state.get("raw_text", ""), pdf_path=pdf_path)
    return {"extracted_data": extracted_data}


def classify_task_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    raw_text = state.get("raw_text", "")
    task_type, reason = _guess_task_type(extracted_data, raw_text)
    return {
        "task_type": task_type,
        "route_reason": reason,
    }


def material_prepare_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    rows = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    extracted_data["line_items"] = rows

    if not _string(extracted_data.get("amount")):
        total = _line_items_total(rows)
        if total is not None:
            extracted_data["amount"] = _format_amount(total)

    return {
        "extracted_data": extracted_data,
        "material_fix_result": {
            "status": "prepared",
            "line_items": len(rows),
        },
    }


def material_repair_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    rows = _normalize_line_items(_to_editor_rows(extracted_data.get("line_items")))
    if not rows:
        return {
            "review_items": [],
            "material_fix_result": {"status": "skipped", "reason": "no_line_items"},
        }

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

    try:
        result = material_fix_agent.run_llm_row_repair(
            rows,
            header_context=header_context,
            raw_text=raw_text,
        )
    except Exception as exc:
        return {
            "review_items": [],
            "material_fix_result": {"status": "failed", "error": str(exc)},
            "llm_error": str(exc),
        }

    repaired_rows = _normalize_line_items(_to_editor_rows(result.get("rows")))
    review_items = list(result.get("review_items") or [])
    llm_stats = dict(result.get("stats") or {})
    llm_error = _string(result.get("llm_error")) or None

    output_extracted = dict(extracted_data)
    output_extracted["rule_line_items_baseline"] = _normalize_line_items(
        _to_editor_rows(extracted_data.get("rule_line_items_baseline"))
    ) or list(rows)
    output_extracted["llm_line_items_suggested"] = repaired_rows or list(rows)
    output_extracted["llm_agent_stats"] = llm_stats
    if repaired_rows:
        output_extracted["line_items"] = repaired_rows
        total = _line_items_total(repaired_rows)
        if total is not None:
            output_extracted["amount"] = _format_amount(total)
    if llm_error:
        output_extracted["llm_error"] = llm_error

    return {
        "extracted_data": output_extracted,
        "review_items": review_items,
        "material_fix_result": {
            "status": "ok",
            "stats": llm_stats,
            "review_items": len(review_items),
        },
        "llm_error": llm_error,
    }


def material_validate_node(state: FinanceGraphState) -> FinanceGraphState:
    db = state["db"]
    extracted_data = dict(state.get("extracted_data", {}) or {})
    suggestion_data = _build_suggestion_data(db, extracted_data, state.get("raw_text", ""))

    review_items = list(state.get("review_items") or [])
    if review_items:
        risk_points = list(suggestion_data.get("risk_points") or [])
        risk_points.append(f"检测到 {len(review_items)} 条低置信度行，建议前端人工复核。")
        suggestion_data["risk_points"] = risk_points

    suggestion_data["route_reason"] = state.get("route_reason", "")
    suggestion_data["task_type"] = "material"
    final_data = _build_final_data(
        suggestion_data,
        extracted_data,
        task_type="material",
        route_reason=state.get("route_reason", ""),
        llm_error=state.get("llm_error"),
        material_fix_result=dict(state.get("material_fix_result", {}) or {}),
    )
    return {
        "suggestion_data": suggestion_data,
        "final_data": final_data,
    }


def travel_prepare_node(state: FinanceGraphState) -> FinanceGraphState:
    raw_text = _string(state.get("raw_text", ""))
    travel_context = {
        "policy_context": rag_retriever.build_travel_policy_context(raw_text, top_k=3),
        "raw_text_preview": raw_text[:600],
    }
    return {"travel_context": travel_context}


def travel_assign_node(state: FinanceGraphState) -> FinanceGraphState:
    extracted_data = dict(state.get("extracted_data", {}) or {})
    travel_context = dict(state.get("travel_context", {}) or {})
    doc_type = _guess_travel_doc_type(extracted_data, state.get("raw_text", ""))
    travel_context.update(
        {
            "doc_type_hint": doc_type,
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
