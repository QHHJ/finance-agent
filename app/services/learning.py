from __future__ import annotations

import hashlib
import json
from typing import Any

from . import rag_store


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _signature(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
    return rag_store.upsert_documents(source_type="material_case", source_id=task_id, documents=documents)


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

    return rag_store.upsert_documents(source_type="travel_case", source_id=session_sig, documents=documents)
