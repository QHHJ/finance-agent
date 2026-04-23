from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import AgentCommand, AgentEvent, AgentResult, AgentTask


class BaseAgent(ABC):
    name = "base_agent"

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        raise NotImplementedError

    def _result(
        self,
        *,
        ok: bool = True,
        summary: str = "",
        payload: dict[str, Any] | None = None,
        events: list[AgentEvent] | None = None,
        commands: list[AgentCommand] | None = None,
    ) -> AgentResult:
        return AgentResult(
            agent=self.name,
            ok=ok,
            summary=summary,
            payload=dict(payload or {}),
            events=list(events or []),
            commands=list(commands or []),
        )

    def _event(self, event_type: str, summary: str, payload: dict[str, Any] | None = None) -> AgentEvent:
        return AgentEvent(
            source=self.name,
            event_type=str(event_type or "").strip(),
            summary=str(summary or "").strip(),
            payload=dict(payload or {}),
        )
