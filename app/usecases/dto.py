from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OperationResult:
    ok: bool
    message: str = ""


@dataclass(slots=True)
class MaterialBatchProcessResult:
    task_ids: list[str] = field(default_factory=list)
    prepare_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TravelStatus:
    missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    issue_items: list[dict[str, Any]] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing": list(self.missing),
            "issues": list(self.issues),
            "issue_items": [dict(item) for item in list(self.issue_items or []) if isinstance(item, dict)],
            "tips": list(self.tips),
            "complete": bool(self.complete),
        }


@dataclass(slots=True)
class IntentParseResult:
    intent_type: str = "chat"
    is_actionable: bool = False
    risk_level: str = "low"
    needs_confirmation: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": str(self.intent_type or "chat"),
            "is_actionable": bool(self.is_actionable),
            "risk_level": str(self.risk_level or "low"),
            "needs_confirmation": bool(self.needs_confirmation),
            "reason": str(self.reason or ""),
        }


@dataclass(slots=True)
class PendingAction:
    action_id: str
    action_type: str
    summary: str
    target: str = ""
    risk_level: str = "medium"
    status: str = "pending"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": str(self.action_id or ""),
            "action_type": str(self.action_type or ""),
            "summary": str(self.summary or ""),
            "target": str(self.target or ""),
            "risk_level": str(self.risk_level or "medium"),
            "status": str(self.status or "pending"),
            "payload": dict(self.payload or {}),
            "created_at": str(self.created_at or ""),
        }


@dataclass(slots=True)
class LastAppliedAction:
    action_id: str
    action_type: str
    summary: str
    scope: str = ""
    applied_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": str(self.action_id or ""),
            "action_type": str(self.action_type or ""),
            "summary": str(self.summary or ""),
            "scope": str(self.scope or ""),
            "applied_at": str(self.applied_at or ""),
        }
