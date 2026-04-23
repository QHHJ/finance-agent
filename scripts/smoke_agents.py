from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents import AgentCommand, AgentTask, ReimbursementAgentOrchestrator


class FakeUpload:
    def __init__(self, name: str, content: bytes = b"") -> None:
        self.name = name
        self.size = len(content)
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _build_mock_profile(uploaded_file, index: int) -> dict:
    name = str(getattr(uploaded_file, "name", ""))
    doc_type = "unknown"
    if "ticket" in name:
        doc_type = "transport_ticket"
    elif "payment" in name:
        doc_type = "transport_payment"
    elif "detail" in name:
        doc_type = "flight_detail"
    elif "hotel_invoice" in name:
        doc_type = "hotel_invoice"
    return {
        "profile_id": f"{index}:{name}",
        "index": index,
        "file": uploaded_file,
        "name": name,
        "doc_type": doc_type,
        "amount": None,
        "date_obj": None,
        "slot": "unknown",
    }


def main() -> None:
    orchestrator = ReimbursementAgentOrchestrator()

    conversation_result = orchestrator.run_task(
        AgentTask(
            agent="conversation_agent",
            objective="plan_travel_turn",
            payload={
                "user_text": "现在还缺什么",
                "intent_parser": lambda text, context: {
                    "intent_type": "chat",
                    "is_actionable": False,
                    "risk_level": "low",
                    "needs_confirmation": False,
                    "reason": "smoke_chat",
                },
                "reply_rule": lambda user_text, assignment, status, profiles: "当前还缺返程票据。",
                "assignment": {},
                "status": {"missing": ["返程机票发票/票据"]},
                "profiles": [],
                "messages": [],
            },
        )
    )
    print(f"[conversation] ok={conversation_result.ok} summary={conversation_result.summary}")
    print(f"[conversation] reply={conversation_result.payload.get('reply', '')}")

    material_conversation_result = orchestrator.run_task(
        AgentTask(
            agent="conversation_agent",
            objective="plan_material_turn",
            payload={
                "user_text": "最后一行规格和项目名混了",
                "intent_parser": lambda text, context: {
                    "intent_type": "ambiguous",
                    "is_actionable": False,
                    "risk_level": "low",
                    "needs_confirmation": False,
                    "reason": "smoke_ambiguous",
                },
                "task": None,
                "fields": {},
                "messages": [],
                "row_count": 5,
                "quality_hint_count": 1,
                "pending_count": 0,
            },
        )
    )
    print(f"[material-conversation] ok={material_conversation_result.ok} summary={material_conversation_result.summary}")
    print(f"[material-conversation] reply={material_conversation_result.payload.get('reply', '')}")

    home_result = orchestrator.run_task(
        AgentTask(
            agent="conversation_agent",
            objective="run_home_turn",
            payload={
                "turn_processor": lambda state, user_message, uploaded_files: (
                    {"recommended_flow": "travel", "conversation_history": [{"role": "assistant", "content": "建议进入差旅"}]},
                    "建议进入差旅",
                ),
                "state": {},
                "user_message": "我要报销差旅",
                "uploaded_files": [],
            },
        )
    )
    print(f"[home] ok={home_result.ok} summary={home_result.summary}")
    print(f"[home] reply={home_result.payload.get('reply', '')}")

    travel_result = orchestrator.run_task(
        AgentTask(
            agent="travel_specialist_agent",
            objective="organize_materials",
            payload={
                "pool_files": [
                    FakeUpload("go_ticket.png"),
                    FakeUpload("go_payment.png"),
                    FakeUpload("go_detail.png"),
                    FakeUpload("hotel_invoice.png"),
                ],
                "build_profile": _build_mock_profile,
            },
        )
    )
    print(f"[travel] ok={travel_result.ok} summary={travel_result.summary}")
    print(f"[travel] missing={travel_result.payload.get('status', {}).get('missing', [])}")

    fake_task = SimpleNamespace(
        extracted_data={
            "amount": "",
            "item_content": "",
            "line_items": [
                {"item_name": "电阻 M0805", "spec": "", "quantity": "10", "unit": "个", "line_total_with_tax": "5.00"},
            ],
        }
    )
    material_result = orchestrator.run_task(
        AgentTask(
            agent="material_specialist_agent",
            objective="extract_fields",
            payload={"task": fake_task},
        )
    )
    print(f"[material] ok={material_result.ok} summary={material_result.summary}")
    if material_result.ok:
        print(f"[material] fields={material_result.payload.get('fields', {}).get('line_items', [])}")
    else:
        print("[material] specialist skipped due to unavailable optional dependency chain.")

    execution_result = orchestrator.execute_command(
        AgentCommand(
            command_type="rebuild_policy_rag_index",
            payload={"limit": 0},
            summary="rebuild policy rag index",
            created_by="smoke",
        )
    )
    print(f"[execution] ok={execution_result.ok} summary={execution_result.summary}")


if __name__ == "__main__":
    main()
