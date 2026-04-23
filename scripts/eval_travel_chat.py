from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.usecases import travel_chat_service as travel_chat


BASE_SAMPLES: list[dict[str, Any]] = [
    {"message": "我还缺什么", "expected_intent": "ask_missing"},
    {"message": "现在还差哪些材料", "expected_intent": "ask_missing"},
    {"message": "酒店还缺什么", "expected_intent": "ask_missing"},
    {"message": "返程材料齐不齐", "expected_intent": "ask_missing"},
    {"message": "去程全不全", "expected_intent": "ask_missing"},
    {"message": "酒店发票是哪几个文件", "expected_intent": "ask_file_list", "expected_slot": "hotel_invoice"},
    {"message": "酒店支付记录对应哪些文件", "expected_intent": "ask_file_list", "expected_slot": "hotel_payment"},
    {"message": "酒店订单截图是哪几个文件", "expected_intent": "ask_file_list", "expected_slot": "hotel_order"},
    {"message": "去程票据是哪些文件", "expected_intent": "ask_file_list", "expected_slot": "go_ticket"},
    {"message": "返程明细是哪几个文件", "expected_intent": "ask_file_list", "expected_slot": "return_detail"},
    {"message": "酒店发票有几份", "expected_intent": "ask_file_count", "expected_slot": "hotel_invoice"},
    {"message": "酒店支付有多少份", "expected_intent": "ask_file_count", "expected_slot": "hotel_payment"},
    {"message": "去程票据有几张", "expected_intent": "ask_file_count", "expected_slot": "go_ticket"},
    {"message": "返程支付记录有几个文件", "expected_intent": "ask_file_count", "expected_slot": "return_payment"},
    {"message": "返程明细数量是多少", "expected_intent": "ask_file_count", "expected_slot": "return_detail"},
    {"message": "哪些金额对不上", "expected_intent": "ask_mismatch", "check_no_duplicate_issue": True},
    {"message": "现在哪里对不上", "expected_intent": "ask_mismatch"},
    {"message": "现在还有什么对不上", "expected_intent": "ask_mismatch"},
    {"message": "酒店金额有不一致吗", "expected_intent": "ask_mismatch"},
    {"message": "返程哪个和哪个对不上", "expected_intent": "ask_mismatch"},
    {"message": "去程金额异常有哪些", "expected_intent": "ask_mismatch"},
    {"message": "我下一步该做什么", "expected_intent": "ask_next_step"},
    {"message": "接下来我怎么处理", "expected_intent": "ask_next_step"},
    {"message": "酒店下一步做什么", "expected_intent": "ask_next_step"},
    {"message": "返程现在做什么", "expected_intent": "ask_next_step"},
    {"message": "酒店材料有哪些", "expected_intent": "ask_requirement"},
    {"message": "去程需要什么材料", "expected_intent": "ask_requirement"},
    {"message": "返程报销要求是什么", "expected_intent": "ask_requirement"},
    {"message": "差旅材料清单给我", "expected_intent": "ask_requirement"},
    {"message": "哈喽", "expected_intent": "clarify"},
    {"message": "随便聊聊", "expected_intent": "clarify"},
    {"message": "这个不太对你再看一下", "expected_intent": "clarify"},
]


CONTEXT_CASES: list[dict[str, Any]] = [
    {
        "message": "还有呢",
        "expected_intent": "ask_mismatch",
        "ctx": {"last_intent": "ask_mismatch", "last_scope": None, "last_target_slot": None, "last_answer_type": "issue_list"},
    },
    {
        "message": "酒店呢",
        "expected_intent": "ask_mismatch",
        "expected_scope": "hotel",
        "ctx": {"last_intent": "ask_mismatch", "last_scope": None, "last_target_slot": None, "last_answer_type": "issue_list"},
    },
    {
        "message": "返程呢",
        "expected_intent": "ask_mismatch",
        "expected_scope": "return",
        "ctx": {"last_intent": "ask_mismatch", "last_scope": None, "last_target_slot": None, "last_answer_type": "issue_list"},
    },
]


FOLLOWUP_FLOW: list[dict[str, Any]] = [
    {"message": "哪些金额对不上", "expected_intent": "ask_mismatch"},
    {"message": "还有呢", "expected_intent": "ask_mismatch"},
    {"message": "酒店呢", "expected_intent": "ask_mismatch", "expected_scope": "hotel"},
    {"message": "返程呢", "expected_intent": "ask_mismatch", "expected_scope": "return"},
]


def _build_demo_state() -> tuple[dict[str, Any], dict[str, Any]]:
    assignment = {
        "go_ticket": [{"name": "go_ticket_A.pdf"}],
        "go_payment": [{"filename": "go_pay_A.png"}],
        "go_detail": [{"file_name": "go_detail_A.pdf"}],
        "return_ticket": [{"path": "C:/docs/return_ticket_A.pdf"}],
        "return_payment": ["return_pay_A.png"],
        "return_detail": [{"name": "return_detail_A.pdf"}],
        "hotel_invoice": [{"name": "hotel_invoice_A.pdf"}, {"filename": "hotel_invoice_B.pdf"}],
        "hotel_payment": [{"file_name": "hotel_pay_A.png"}],
        "hotel_order": [{"path": "D:/tmp/hotel_order_A.png"}],
        "go_ticket_amount": "1380.00",
        "go_payment_amount": "1379.88",
        "return_ticket_amount": "1268.00",
        "return_payment_amount": "1268.00",
        "hotel_invoice_amount": "2266.00",
        "hotel_payment_amount": "3012.00",
    }
    status = {
        "missing": ["酒店订单截图"],
        "issues": [
            "去程交通票据金额与支付记录金额不一致：1380.00 vs 1379.88",
            "酒店票据金额与支付记录金额不一致：2266.00 vs 3012.00",
        ],
        "issue_items": [
            {
                "scope": "go",
                "kind": "amount_mismatch",
                "label": "去程交通票据金额与支付记录金额不一致",
                "invoice_amount": 1380.00,
                "payment_amount": 1379.88,
                "file_refs": ["go_ticket_A.pdf", "go_pay_A.png"],
            },
            {
                "scope": "hotel",
                "kind": "amount_mismatch",
                "label": "酒店票据金额与支付记录金额不一致",
                "invoice_amount": 2266.00,
                "payment_amount": 3012.00,
                "file_refs": ["hotel_invoice_A.pdf", "hotel_pay_A.png"],
            },
        ],
        "tips": [],
        "complete": False,
    }
    return assignment, status


def _has_duplicate_issue_lines(answer: str) -> bool:
    lines = [line.strip() for line in str(answer or "").splitlines() if line.strip().startswith("- ")]
    if not lines:
        return False
    labels: list[str] = []
    for line in lines:
        body = line[2:].strip()
        label = body.split("：", 1)[0].strip() if "：" in body else body
        labels.append(label)
    return len(labels) != len(set(labels))


def _run_case(
    *,
    index: int,
    sample: dict[str, Any],
    assignment: dict[str, Any],
    status: dict[str, Any],
    chained_context: travel_chat.TravelChatContext | None = None,
) -> tuple[bool, str, travel_chat.TravelChatContext]:
    message = str(sample.get("message") or "")
    expected_intent = str(sample.get("expected_intent") or "")
    expected_slot = sample.get("expected_slot")
    expected_scope = sample.get("expected_scope")
    check_no_duplicate_issue = bool(sample.get("check_no_duplicate_issue"))

    if chained_context is not None:
        context_obj = chained_context
    else:
        context_obj = travel_chat.ensure_travel_chat_context(sample.get("ctx"))

    ok = True
    reason_parts: list[str] = []
    actual_intent = ""
    actual_slots: list[str] = []
    actual_scope = None
    answer_preview = ""
    answer_text = ""
    next_context = context_obj

    try:
        query = travel_chat.parse_travel_chat_query(
            message,
            {
                "chat_context": context_obj.model_dump(),
                "assignment": assignment,
                "status": status,
            },
        )
        actual_intent = str(query.intent.value)
        actual_slots = list(query.target_slots or [])
        actual_scope = query.scope
        payload = travel_chat.execute_travel_chat_query(query, assignment, status)
        answer = travel_chat.render_travel_chat_answer(payload)
        answer_text = str(answer or "")
        answer_preview = answer_text.replace("\n", " | ")[:120]
        next_context = travel_chat.update_travel_chat_context(context_obj, query, payload)
    except Exception as exc:  # pragma: no cover - eval script only
        ok = False
        reason_parts.append(f"runtime_error={exc}")
        return ok, f"[FAIL] #{index:02d} {message} -> {'; '.join(reason_parts)}", next_context

    if actual_intent != expected_intent:
        ok = False
        reason_parts.append(f"intent expected={expected_intent} actual={actual_intent}")

    if expected_slot and str(expected_slot) not in actual_slots:
        ok = False
        reason_parts.append(f"slot expected={expected_slot} actual={actual_slots}")

    if expected_scope and str(actual_scope or "") != str(expected_scope):
        ok = False
        reason_parts.append(f"scope expected={expected_scope} actual={actual_scope}")

    if check_no_duplicate_issue and actual_intent == travel_chat.TravelChatIntent.ASK_MISMATCH.value:
        if _has_duplicate_issue_lines(answer_text):
            ok = False
            reason_parts.append("issue_lines_duplicated")

    if ok:
        return ok, f"[PASS] #{index:02d} {message} -> intent={actual_intent} scope={actual_scope} slots={actual_slots} answer={answer_preview}", next_context
    return ok, f"[FAIL] #{index:02d} {message} -> {'; '.join(reason_parts)}", next_context


def main() -> None:
    assignment, status = _build_demo_state()
    all_cases = list(BASE_SAMPLES) + list(CONTEXT_CASES)
    total = len(all_cases) + len(FOLLOWUP_FLOW)
    passed = 0
    by_intent: dict[str, dict[str, int]] = {}

    print(f"[travel_chat_eval] total samples = {total}")
    row_idx = 0
    for sample in all_cases:
        row_idx += 1
        ok, row_text, _ = _run_case(index=row_idx, sample=sample, assignment=assignment, status=status)
        print(row_text)
        expected_intent = str(sample.get("expected_intent") or "unknown")
        bucket = by_intent.setdefault(expected_intent, {"total": 0, "pass": 0})
        bucket["total"] += 1
        if ok:
            bucket["pass"] += 1
            passed += 1

    flow_ctx = travel_chat.TravelChatContext()
    print("\n[followup_flow]")
    for sample in FOLLOWUP_FLOW:
        row_idx += 1
        ok, row_text, flow_ctx = _run_case(
            index=row_idx,
            sample=sample,
            assignment=assignment,
            status=status,
            chained_context=flow_ctx,
        )
        print(row_text + f" | next_ctx={flow_ctx.model_dump()}")
        expected_intent = str(sample.get("expected_intent") or "unknown")
        bucket = by_intent.setdefault(expected_intent, {"total": 0, "pass": 0})
        bucket["total"] += 1
        if ok:
            bucket["pass"] += 1
            passed += 1

    print("\n[summary]")
    print(f"passed={passed}/{total} ({(passed / total * 100.0):.1f}%)")
    for intent in sorted(by_intent.keys()):
        row = by_intent[intent]
        rate = (row["pass"] / row["total"] * 100.0) if row["total"] else 0.0
        print(f"- {intent}: {row['pass']}/{row['total']} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
