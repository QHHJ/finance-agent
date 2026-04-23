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
    print(f"[travel] missing={travel_result.payload['status']['missing']}")

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
    print(f"[material] fields={material_result.payload['fields']['line_items']}")

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
