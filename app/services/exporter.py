from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook


def _cell_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).replace(",", "").replace("¥", "").replace("￥", "").strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_material_case(extracted_data: dict[str, Any], final_data: dict[str, Any]) -> bool:
    category = str(final_data.get("expense_category") or "")
    if "材料费" in category:
        return True

    text = " ".join(
        [
            str(extracted_data.get("item_content") or ""),
            str(extracted_data.get("bill_type") or ""),
            str(extracted_data.get("seller") or ""),
        ]
    )
    hints = ["材料", "电子元件", "金属制品", "法兰", "变压器", "连接器", "实验室", "入库"]
    return any(h in text for h in hints)


def _sanitize_spec(value: Any) -> str:
    spec = str(value or "").strip()
    # OCR常把“单价”误进规格型号，纯小数在本模板下清空。
    if re.fullmatch(r"\d+\.\d+", spec):
        return ""
    return spec


def _line_total_with_tax(item: dict[str, Any]) -> float | None:
    direct = _safe_float(item.get("line_total_with_tax"))
    if direct is not None:
        return direct

    amount_no_tax = _safe_float(item.get("amount_no_tax"))
    tax_amount = _safe_float(item.get("tax_amount"))
    if amount_no_tax is not None and tax_amount is not None:
        return amount_no_tax + tax_amount
    if amount_no_tax is not None:
        return amount_no_tax
    return None


def _material_rows(extracted_data: dict[str, Any]) -> list[dict[str, str]]:
    line_items = extracted_data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []

    rows: list[dict[str, str]] = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        total = _line_total_with_tax(item)
        rows.append(
            {
                "item_name": str(item.get("item_name") or ""),
                "spec": _sanitize_spec(item.get("spec")),
                "quantity": str(item.get("quantity") or ""),
                "unit": str(item.get("unit") or ""),
                "line_total_with_tax": f"{total:.2f}" if total is not None else "",
            }
        )

    if rows:
        return rows

    fallback_total = _safe_float(extracted_data.get("amount"))
    return [
        {
            "item_name": str(extracted_data.get("item_content") or "*未识别项目*"),
            "spec": "",
            "quantity": "",
            "unit": "",
            "line_total_with_tax": f"{fallback_total:.2f}" if fallback_total is not None else "",
        }
    ]


def _build_material_sheet(workbook: Workbook, extracted_data: dict[str, Any], final_data: dict[str, Any]) -> None:
    sheet = workbook.active
    sheet.title = "材料费入库明细"

    invoice_number = extracted_data.get("invoice_number") or ""
    seller = extracted_data.get("seller") or ""

    sheet.append(["发票号码", invoice_number])
    sheet.append(["销售方名称", seller])
    sheet.append([])
    sheet.append(["项目名称(含星号)", "规格型号", "数量", "单位", "每项含税总价"])

    rows = _material_rows(extracted_data)
    row_totals: list[float] = []
    for row in rows:
        value = _safe_float(row["line_total_with_tax"])
        if value is not None:
            row_totals.append(value)
        sheet.append([row["item_name"], row["spec"], row["quantity"], row["unit"], row["line_total_with_tax"]])

    line_sum = sum(row_totals) if row_totals else None
    extracted_amount = _safe_float(extracted_data.get("amount"))

    final_total: float | None
    if extracted_amount is None:
        final_total = line_sum
    elif line_sum is None:
        final_total = extracted_amount
    elif abs(extracted_amount - line_sum) <= 0.1:
        final_total = extracted_amount
    else:
        # 当抽取总价疑似为“不含税合计”时，优先采用明细含税求和。
        final_total = line_sum

    sheet.append([])
    sheet.append(["合计总价（含税）", f"{final_total:.2f}" if final_total is not None else ""])

    detail = workbook.create_sheet("系统结果")
    detail.append(["field", "value"])
    for key, value in extracted_data.items():
        detail.append([f"extracted.{key}", _cell_value(value)])
    for key, value in final_data.items():
        detail.append([f"final.{key}", _cell_value(value)])


def _build_generic_sheet(
    workbook: Workbook,
    task_id: str,
    extracted_data: dict[str, Any],
    suggestion_data: dict[str, Any],
    final_data: dict[str, Any],
) -> None:
    sheet = workbook.active
    sheet.title = "finance_result"
    sheet.append(["field", "value"])
    sheet.append(["task_id", task_id])

    for key, value in extracted_data.items():
        sheet.append([f"extracted.{key}", _cell_value(value)])
    for key, value in suggestion_data.items():
        sheet.append([f"suggestion.{key}", _cell_value(value)])
    for key, value in final_data.items():
        sheet.append([f"final.{key}", _cell_value(value)])


def export_to_excel(
    task_id: str,
    extracted_data: dict[str, Any],
    suggestion_data: dict[str, Any],
    final_data: dict[str, Any],
    export_dir: str | Path,
) -> str:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    file_path = export_path / f"{task_id}.xlsx"

    workbook = Workbook()
    if _is_material_case(extracted_data, final_data):
        _build_material_sheet(workbook, extracted_data=extracted_data, final_data=final_data)
    else:
        _build_generic_sheet(
            workbook,
            task_id=task_id,
            extracted_data=extracted_data,
            suggestion_data=suggestion_data,
            final_data=final_data,
        )

    workbook.save(file_path)
    return str(file_path.resolve())


def export_to_text(
    task_id: str,
    extracted_data: dict[str, Any],
    suggestion_data: dict[str, Any],
    final_data: dict[str, Any],
    export_dir: str | Path,
) -> str:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    file_path = export_path / f"{task_id}.txt"

    content = (
        f"task_id: {task_id}\n"
        f"\n[extracted]\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
        f"\n[suggestion]\n{json.dumps(suggestion_data, ensure_ascii=False, indent=2)}\n"
        f"\n[final]\n{json.dumps(final_data, ensure_ascii=False, indent=2)}\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return str(file_path.resolve())
