from __future__ import annotations

import re
from typing import Any

from app.services import extractor, local_runner, material_fix_agent

from .dto import MaterialBatchProcessResult, OperationResult


LINE_ITEM_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]


def safe_float(value: Any) -> float | None:
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


def format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def to_editor_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def normalize_quantity(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def normalize_line_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item_name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        quantity = normalize_quantity(row.get("quantity"))
        unit = str(row.get("unit") or "").strip()
        line_total = format_amount(safe_float(row.get("line_total_with_tax")))
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


def line_items_total(rows: list[dict[str, str]]) -> float | None:
    values = [safe_float(row.get("line_total_with_tax")) for row in rows]
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers)


def split_name_spec(name: str, spec: str) -> tuple[str, str]:
    new_name = str(name or "").strip()
    new_spec = str(spec or "").strip()

    splitter = getattr(extractor, "_split_item_name_and_spec", None)
    refiner = getattr(extractor, "_refine_name_spec_boundary", None)
    if callable(splitter):
        try:
            new_name, new_spec = splitter(new_name, new_spec)
        except Exception:
            pass
    if callable(refiner):
        try:
            new_name, new_spec = refiner(new_name, new_spec)
        except Exception:
            pass

    if not new_spec:
        match = re.search(
            r"(M\d+(?:[Xx*]\d+(?:\.\d+)?)?|[A-Za-z][A-Za-z0-9+\-_.]{1,20}专用|[A-Za-z]{2,}\d[\w\-./]*|\d+(?:\.\d+)?(?:mm|cm|m|kg|g|V|W|A|Hz)|\d+(?:\.\d+)?\s*(?:-|~|～|x|X|[*])\s*\d+(?:\.\d+)?(?:mm|cm|m)|超薄)$",
            new_name,
        )
        if match:
            candidate = match.group(1).strip()
            head = new_name[: match.start()].strip(" -_/，,;；:：")
            if head and candidate and candidate not in {"电子元件", "金属制品"}:
                new_name, new_spec = head, candidate

    if new_spec and new_name and new_name.endswith(new_spec):
        trimmed = new_name[: -len(new_spec)].rstrip(" -_/，,;；:：")
        if trimmed:
            new_name = trimmed

    return new_name, new_spec


def auto_split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    normalized_rows = normalize_line_items(to_editor_rows(rows))
    changed = 0
    output: list[dict[str, Any]] = []
    for row in normalized_rows:
        name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        new_name, new_spec = split_name_spec(name, spec)
        if new_name != name or new_spec != spec:
            changed += 1
        output.append(
            {
                "item_name": new_name,
                "spec": new_spec,
                "quantity": str(row.get("quantity") or "").strip(),
                "unit": str(row.get("unit") or "").strip(),
                "line_total_with_tax": str(row.get("line_total_with_tax") or "").strip(),
            }
        )
    return output, changed


def extract_fields(task: Any) -> dict[str, Any]:
    extracted = dict(getattr(task, "extracted_data", {}) or {})
    rows = normalize_line_items(to_editor_rows(extracted.get("line_items")))
    auto_split_enabled = bool(extracted.get("auto_split_enabled", True))
    if rows and auto_split_enabled:
        auto_fixed_rows, _ = auto_split_rows(rows)
        if auto_fixed_rows:
            rows = auto_fixed_rows
    extracted["line_items"] = rows
    extracted["auto_split_enabled"] = auto_split_enabled

    if not str(extracted.get("item_content") or "").strip() and rows:
        names = [str(row.get("item_name") or "").strip() for row in rows if str(row.get("item_name") or "").strip()]
        extracted["item_content"] = "；".join(names[:8])

    amount_text = str(extracted.get("amount") or "").strip()
    if not amount_text:
        total = line_items_total(rows)
        if total is not None:
            extracted["amount"] = format_amount(total)

    if not isinstance(extracted.get("low_confidence_review"), list):
        extracted["low_confidence_review"] = []
    if not isinstance(extracted.get("llm_agent_stats"), dict):
        extracted["llm_agent_stats"] = {}
    extracted["rule_line_items_baseline"] = normalize_line_items(to_editor_rows(extracted.get("rule_line_items_baseline")))
    extracted["llm_line_items_suggested"] = normalize_line_items(to_editor_rows(extracted.get("llm_line_items_suggested")))
    return extracted


def build_fields_payload(fields: dict[str, Any]) -> dict[str, Any]:
    rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    amount_text = str(fields.get("amount") or "").strip()
    if not amount_text:
        total = line_items_total(rows)
        if total is not None:
            amount_text = format_amount(total)

    item_content = str(fields.get("item_content") or "").strip()
    if not item_content and rows:
        names = [str(row.get("item_name") or "").strip() for row in rows if str(row.get("item_name") or "").strip()]
        item_content = "；".join(names[:8])

    return {
        "invoice_number": str(fields.get("invoice_number") or "").strip() or None,
        "invoice_date": str(fields.get("invoice_date") or "").strip() or None,
        "amount": amount_text or None,
        "tax_amount": str(fields.get("tax_amount") or "").strip() or None,
        "seller": str(fields.get("seller") or "").strip() or None,
        "buyer": str(fields.get("buyer") or "").strip() or None,
        "bill_type": str(fields.get("bill_type") or "").strip() or None,
        "item_content": item_content or None,
        "line_items": rows,
        "low_confidence_review": list(fields.get("low_confidence_review") or []),
        "llm_agent_stats": dict(fields.get("llm_agent_stats") or {}),
        "auto_split_enabled": bool(fields.get("auto_split_enabled", True)),
        "rule_line_items_baseline": normalize_line_items(to_editor_rows(fields.get("rule_line_items_baseline"))),
        "llm_line_items_suggested": normalize_line_items(to_editor_rows(fields.get("llm_line_items_suggested"))),
    }


def apply_updates(task_id: str, fields: dict[str, Any]) -> OperationResult:
    try:
        corrections = {
            "expense_category": "材料费",
            "extracted_fields": build_fields_payload(fields),
        }
        # apply_corrections 内部包含 learn_from_material_task 回写
        local_runner.apply_corrections(task_id, corrections)
        local_runner.export_task(task_id, export_format="both")
        return OperationResult(ok=True)
    except Exception as exc:
        return OperationResult(ok=False, message=str(exc))


def run_llm_fix(task: Any, fields: dict[str, Any]) -> tuple[bool, str, Any, dict[str, Any]]:
    rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    if not rows:
        return True, "当前无明细可分析。", task, fields

    header_context = {
        "invoice_number": fields.get("invoice_number"),
        "invoice_date": fields.get("invoice_date"),
        "bill_type": fields.get("bill_type"),
        "item_content": fields.get("item_content"),
        "seller": fields.get("seller"),
        "buyer": fields.get("buyer"),
        "amount": fields.get("amount"),
    }
    raw_text = str(getattr(task, "raw_text", "") or "")

    try:
        result = material_fix_agent.run_llm_row_repair(
            rows,
            header_context=header_context,
            raw_text=raw_text,
        )
    except Exception as exc:
        return True, f"LLM修复执行失败：{exc}", task, fields

    repaired_rows = normalize_line_items(to_editor_rows(result.get("rows")))
    review_items = list(result.get("review_items") or [])
    llm_stats = dict(result.get("stats") or {})
    llm_error = str(result.get("llm_error") or "").strip()

    new_fields = dict(fields)
    new_fields["auto_split_enabled"] = True
    baseline = normalize_line_items(to_editor_rows(fields.get("rule_line_items_baseline")))
    new_fields["rule_line_items_baseline"] = baseline if baseline else list(rows)

    if repaired_rows:
        new_fields["line_items"] = repaired_rows
        new_fields["llm_line_items_suggested"] = list(repaired_rows)
    else:
        new_fields["llm_line_items_suggested"] = list(rows)

    fixed_total = line_items_total(repaired_rows)
    if fixed_total is not None:
        new_fields["amount"] = format_amount(fixed_total)
    new_fields["low_confidence_review"] = review_items
    new_fields["llm_agent_stats"] = llm_stats
    if llm_error:
        new_fields["llm_error"] = llm_error

    save_result = apply_updates(task.id, new_fields)
    if not save_result.ok:
        return True, f"LLM修复结果保存失败：{save_result.message}", task, fields

    updated_task = local_runner.get_task(task.id) or task
    updated_fields = extract_fields(updated_task)
    summary = (
        f"LLM可疑行分析完成：疑似 {llm_stats.get('suspicious_rows', 0)} 行，"
        f"自动修复 {llm_stats.get('auto_fixed_rows', 0)} 行，"
        f"待人工复核 {len(review_items)} 行。"
    )
    if llm_error:
        summary += f"（部分分块失败：{llm_error[:120]}）"
    return True, summary, updated_task, updated_fields


def build_review_compare_rows(fields: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    review_items = list(fields.get("low_confidence_review") or [])

    left_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    for item in review_items:
        if not isinstance(item, dict):
            continue
        try:
            row_no = int(item.get("row_no") or 0)
        except (TypeError, ValueError):
            continue
        if row_no <= 0:
            continue

        current = rows[row_no - 1] if row_no <= len(rows) else {}
        cur_name = str(current.get("item_name") or item.get("item_name") or "").strip()
        cur_spec = str(current.get("spec") or item.get("spec") or "").strip()
        cur_quantity = str(current.get("quantity") or "").strip()
        cur_unit = str(current.get("unit") or "").strip()
        cur_total = str(current.get("line_total_with_tax") or "").strip()

        sug_name = str(item.get("suggested_item_name") or cur_name).strip()
        sug_spec = str(item.get("suggested_spec") or cur_spec).strip()
        confidence_raw = item.get("confidence")
        try:
            confidence_text = f"{float(confidence_raw) * 100:.1f}%"
        except (TypeError, ValueError):
            confidence_text = str(confidence_raw or "")
        risk_text = "、".join(str(x) for x in (item.get("risk_types") or []) if str(x).strip())
        reason_text = str(item.get("reason") or "").strip()

        left_rows.append(
            {
                "行号": row_no,
                "项目名称": cur_name,
                "规格型号": cur_spec,
                "数量": cur_quantity,
                "单位": cur_unit,
                "每项含税总价": cur_total,
                "置信度": confidence_text,
                "风险类型": risk_text,
                "原因": reason_text,
            }
        )
        right_rows.append(
            {
                "row_no": row_no,
                "item_name": sug_name,
                "spec": sug_spec,
                "quantity": cur_quantity,
                "unit": cur_unit,
                "line_total_with_tax": cur_total,
                "confidence_text": confidence_text,
                "risk_types": risk_text,
                "reason": reason_text,
            }
        )
    return left_rows, right_rows


def apply_review_compare_edits(
    task_id: str,
    fields: dict[str, Any],
    edited_rows: list[dict[str, Any]],
) -> OperationResult:
    base_rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    if not base_rows:
        return OperationResult(ok=False, message="当前主表无可更新明细。")

    changed = 0
    for row in edited_rows:
        if not isinstance(row, dict):
            continue
        try:
            row_no = int(row.get("row_no") or 0)
        except (TypeError, ValueError):
            continue
        if row_no <= 0 or row_no > len(base_rows):
            continue

        idx = row_no - 1
        current = dict(base_rows[idx])
        next_total = format_amount(safe_float(row.get("line_total_with_tax"))) or current.get("line_total_with_tax", "")
        candidate = {
            "item_name": str(row.get("item_name") or "").strip() or current.get("item_name", ""),
            "spec": str(row.get("spec") or "").strip(),
            "quantity": normalize_quantity(row.get("quantity")),
            "unit": str(row.get("unit") or "").strip(),
            "line_total_with_tax": next_total,
        }
        if candidate != current:
            base_rows[idx] = candidate
            changed += 1

    if changed <= 0:
        return OperationResult(ok=False, message="已执行应用，但检测到右侧与主表无差异（或当前单元格未提交修改）。")

    new_fields = dict(fields)
    new_fields["auto_split_enabled"] = False
    new_fields["line_items"] = base_rows
    total = line_items_total(base_rows)
    if total is not None:
        new_fields["amount"] = format_amount(total)
    new_fields["low_confidence_review"] = []

    save_result = apply_updates(task_id, new_fields)
    if not save_result.ok:
        return OperationResult(ok=False, message=save_result.message or "保存失败。")
    return OperationResult(ok=True, message=f"已应用复核修改 {changed} 行，并写入学习案例。")


def build_rule_llm_compare_rows(fields: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule_rows = normalize_line_items(to_editor_rows(fields.get("rule_line_items_baseline")))
    llm_rows = normalize_line_items(to_editor_rows(fields.get("llm_line_items_suggested")))
    if not llm_rows:
        llm_rows = normalize_line_items(to_editor_rows(fields.get("line_items")))

    total = max(len(rule_rows), len(llm_rows))
    left_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    for idx in range(total):
        row_no = idx + 1
        rule = rule_rows[idx] if idx < len(rule_rows) else {}
        llm = llm_rows[idx] if idx < len(llm_rows) else rule
        left_rows.append(
            {
                "行号": row_no,
                "项目名称": str(rule.get("item_name") or ""),
                "规格型号": str(rule.get("spec") or ""),
                "数量": str(rule.get("quantity") or ""),
                "单位": str(rule.get("unit") or ""),
                "每项含税总价": str(rule.get("line_total_with_tax") or ""),
            }
        )
        right_rows.append(
            {
                "row_no": row_no,
                "item_name": str(llm.get("item_name") or ""),
                "spec": str(llm.get("spec") or ""),
                "quantity": str(llm.get("quantity") or ""),
                "unit": str(llm.get("unit") or ""),
                "line_total_with_tax": str(llm.get("line_total_with_tax") or ""),
            }
        )
    return left_rows, right_rows


def rule_llm_diff_count(fields: dict[str, Any]) -> int:
    rule_rows = normalize_line_items(to_editor_rows(fields.get("rule_line_items_baseline")))
    llm_rows = normalize_line_items(to_editor_rows(fields.get("llm_line_items_suggested")))
    if not llm_rows:
        llm_rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    total = max(len(rule_rows), len(llm_rows))
    changed = 0
    for idx in range(total):
        old = rule_rows[idx] if idx < len(rule_rows) else {}
        new = llm_rows[idx] if idx < len(llm_rows) else {}
        if old != new:
            changed += 1
    return changed


def apply_rule_llm_compare_edits(
    task_id: str,
    fields: dict[str, Any],
    edited_rows: list[dict[str, Any]],
) -> OperationResult:
    normalized_rows = normalize_line_items(to_editor_rows(edited_rows))
    if not normalized_rows:
        return OperationResult(ok=False, message="右侧无可应用数据。")

    current_rows = normalize_line_items(to_editor_rows(fields.get("line_items")))
    changed = 0
    total = max(len(current_rows), len(normalized_rows))
    for idx in range(total):
        old = current_rows[idx] if idx < len(current_rows) else {}
        new = normalized_rows[idx] if idx < len(normalized_rows) else {}
        if old != new:
            changed += 1
    if changed <= 0:
        return OperationResult(ok=True, message="两表当前一致，无需应用。")

    new_fields = dict(fields)
    new_fields["auto_split_enabled"] = False
    new_fields["line_items"] = normalized_rows
    new_fields["llm_line_items_suggested"] = list(normalized_rows)
    new_fields["low_confidence_review"] = []
    total_amount = line_items_total(normalized_rows)
    if total_amount is not None:
        new_fields["amount"] = format_amount(total_amount)

    save_result = apply_updates(task_id, new_fields)
    if not save_result.ok:
        return OperationResult(ok=False, message=save_result.message or "保存失败。")
    return OperationResult(ok=True, message=f"已应用规则/LLM对比结果，更新 {changed} 行，并写入学习案例。")


def process_uploaded_files(uploaded_files: list[Any]) -> MaterialBatchProcessResult:
    result = MaterialBatchProcessResult()
    for file in uploaded_files:
        task = local_runner.create_and_process_task(
            file.name,
            file.getvalue(),
            auto_process=True,
            auto_export=True,
        )
        if task is None:
            continue
        try:
            fields = extract_fields(task)
            _, _, prepared_task, _ = run_llm_fix(task, fields)
            task = prepared_task or task
        except Exception as exc:
            result.prepare_errors.append(f"{file.name}: {exc}")
        result.task_ids.append(task.id)
    return result


def reprocess_and_export(task_id: str) -> OperationResult:
    try:
        local_runner.process_task(task_id)
        local_runner.export_task(task_id, export_format="both")
        return OperationResult(ok=True)
    except Exception as exc:
        return OperationResult(ok=False, message=str(exc))
