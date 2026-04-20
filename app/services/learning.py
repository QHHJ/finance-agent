from __future__ import annotations

import hashlib
import json
from typing import Any

from app.retrieval.factory import get_retriever


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _signature(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _parse_json_or_none(text: Any) -> Any:
    if text in (None, ""):
        return None
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(str(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _normalize_row(value: Any) -> dict[str, str]:
    row = dict(value or {}) if isinstance(value, dict) else {}
    return {
        "item_name": str(row.get("item_name") or "").strip(),
        "spec": str(row.get("spec") or "").strip(),
        "quantity": str(row.get("quantity") or "").strip(),
        "unit": str(row.get("unit") or "").strip(),
        "line_total_with_tax": str(row.get("line_total_with_tax") or "").strip(),
    }


def _row_changed(before: dict[str, str], after: dict[str, str]) -> bool:
    return any(str(before.get(k) or "") != str(after.get(k) or "") for k in before.keys())


def _build_risk_tags(before: dict[str, str], after: dict[str, str]) -> list[str]:
    tags: list[str] = []
    before_name = str(before.get("item_name") or "")
    before_spec = str(before.get("spec") or "")
    after_name = str(after.get("item_name") or "")
    after_spec = str(after.get("spec") or "")

    if before_spec and before_name and before_spec in before_name and (after_name != before_name or after_spec != before_spec):
        tags.append("name_spec_overlap")
    if not before_spec and after_spec:
        tags.append("spec_filled")
    if before_name != after_name:
        tags.append("item_name_changed")
    if before_spec != after_spec:
        tags.append("spec_changed")
    return tags


def _build_material_fix_case_documents(task: Any, extracted: dict[str, Any]) -> list[dict[str, Any]]:
    task_id = _safe_text(getattr(task, "id", ""))
    bill_type = _safe_text(extracted.get("bill_type"))
    item_content = _safe_text(extracted.get("item_content"))
    seller = _safe_text(extracted.get("seller"))
    buyer = _safe_text(extracted.get("buyer"))

    docs: list[dict[str, Any]] = []
    for correction in list(getattr(task, "corrections", []) or []):
        field_name = _safe_text(getattr(correction, "field_name", ""))
        if field_name != "extracted.line_items":
            continue
        old_items = _parse_json_or_none(getattr(correction, "old_value", None))
        new_items = _parse_json_or_none(getattr(correction, "new_value", None))
        if not isinstance(old_items, list) or not isinstance(new_items, list):
            continue

        total = max(len(old_items), len(new_items))
        correction_id = int(getattr(correction, "id", 0) or 0)
        for idx in range(total):
            before = _normalize_row(old_items[idx] if idx < len(old_items) else {})
            after = _normalize_row(new_items[idx] if idx < len(new_items) else {})
            if not _row_changed(before, after):
                continue
            risk_tags = _build_risk_tags(before, after)
            row_no = idx + 1
            payload = {
                "bill_type": bill_type,
                "item_content": item_content,
                "seller": seller,
                "buyer": buyer,
                "before_row": before,
                "after_row": after,
                "risk_tags": risk_tags,
                "row_no": row_no,
            }
            payload_text = json.dumps(payload, ensure_ascii=False)
            if correction_id > 0:
                doc_key = f"material_fix_case:{task_id}:{correction_id}:{row_no}"
            else:
                doc_key = f"material_fix_case:{task_id}:{_signature(payload_text)}:{row_no}"
            docs.append(
                {
                    "doc_key": doc_key,
                    "title": f"material_fix_case_row_{row_no}",
                    "content": payload_text,
                    "metadata": {
                        "task_id": task_id,
                        "bill_type": bill_type,
                        "item_content": item_content,
                        "risk_tags": risk_tags,
                        "row_no": row_no,
                        "before_row": before,
                        "after_row": after,
                        "source": "manual_correction",
                    },
                }
            )
    return docs


def learn_from_material_task(task: Any) -> int:
    extracted = dict(getattr(task, "extracted_data", {}) or {})
    final_data = dict(getattr(task, "final_data", {}) or {})

    expense_category = _safe_text(final_data.get("expense_category"))
    if not expense_category:
        return 0

    query_text = "\n".join(
        [
            f"bill_type: {_safe_text(extracted.get('bill_type'))}",
            f"item_content: {_safe_text(extracted.get('item_content'))}",
            f"seller: {_safe_text(extracted.get('seller'))}",
            f"buyer: {_safe_text(extracted.get('buyer'))}",
            f"amount: {_safe_text(extracted.get('amount'))}",
        ]
    )
    answer_text = "\n".join(
        [
            f"expense_category: {expense_category}",
            f"required_materials: {_safe_text(final_data.get('required_materials'))}",
            f"risk_points: {_safe_text(final_data.get('risk_points'))}",
        ]
    )
    content = f"{query_text}\n{answer_text}"

    task_id = _safe_text(getattr(task, "id", ""))
    doc_key = f"material_case:{task_id}"
    documents = [
        {
            "doc_key": doc_key,
            "title": f"material_case_{task_id}",
            "content": content,
            "metadata": {
                "task_id": task_id,
                "expense_category": expense_category,
                "bill_type": _safe_text(extracted.get("bill_type")),
                "item_content": _safe_text(extracted.get("item_content")),
                "source": "manual_correction",
            },
        }
    ]
    retriever = get_retriever()
    learned = retriever.upsert_documents(source_type="material_case", source_id=task_id, documents=documents)

    fix_case_docs = _build_material_fix_case_documents(task, extracted)
    if fix_case_docs:
        learned += retriever.upsert_documents(
            source_type="material_fix_case",
            source_id=task_id,
            documents=fix_case_docs,
        )
    return learned


def learn_from_travel_profiles(
    profiles: list[dict[str, Any]],
    assignment: dict[str, Any],
    *,
    reason: str = "manual_update",
) -> int:
    if not profiles:
        return 0

    compact_rows: list[str] = []
    documents: list[dict[str, Any]] = []

    for profile in profiles:
        doc_type = str(profile.get("doc_type") or "").strip()
        slot = str(profile.get("slot") or "").strip()
        amount = _safe_text(profile.get("amount"))
        date = _safe_text(profile.get("date"))
        source = str(profile.get("source") or "").strip()
        evidence = _safe_text(profile.get("evidence"))
        file_name = _safe_text(profile.get("name"))

        compact_rows.append(f"{file_name}|{doc_type}|{slot}|{amount}|{date}")
        if not source.startswith("manual"):
            continue

        row_text = (
            f"doc_type: {doc_type}\nslot: {slot}\namount: {amount}\n"
            f"date: {date}\nevidence: {evidence}\nreason: {reason}"
        )
        row_sig = _signature(f"{file_name}|{row_text}")
        documents.append(
            {
                "doc_key": f"travel_case:file:{row_sig}",
                "source_id": row_sig,
                "title": f"travel_file_case_{file_name}",
                "content": row_text,
                "metadata": {
                    "doc_type": doc_type,
                    "slot": slot,
                    "source": source,
                    "reason": reason,
                },
            }
        )

    # Session-level case: useful for chat answering "what is missing / how allocated".
    summary_payload = "\n".join(sorted(compact_rows))
    session_sig = _signature(summary_payload)
    summary_text = "\n".join(
        [
            f"reason: {reason}",
            f"go_ticket_amount: {_safe_text(assignment.get('go_ticket_amount'))}",
            f"go_payment_amount: {_safe_text(assignment.get('go_payment_amount'))}",
            f"return_ticket_amount: {_safe_text(assignment.get('return_ticket_amount'))}",
            f"return_payment_amount: {_safe_text(assignment.get('return_payment_amount'))}",
            f"hotel_invoice_amount: {_safe_text(assignment.get('hotel_invoice_amount'))}",
            f"hotel_payment_amount: {_safe_text(assignment.get('hotel_payment_amount'))}",
            "profiles:",
            summary_payload,
        ]
    )
    documents.append(
        {
            "doc_key": f"travel_case:session:{session_sig}",
            "source_id": session_sig,
            "title": f"travel_session_case_{session_sig}",
            "content": summary_text,
            "metadata": {"reason": reason, "profile_count": len(profiles)},
        }
    )

    retriever = get_retriever()
    return retriever.upsert_documents(source_type="travel_case", source_id=session_sig, documents=documents)
