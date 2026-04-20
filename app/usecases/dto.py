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
    tips: list[str] = field(default_factory=list)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing": list(self.missing),
            "issues": list(self.issues),
            "tips": list(self.tips),
            "complete": bool(self.complete),
        }

