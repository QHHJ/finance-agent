from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentTask:
    agent: str
    objective: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCommand:
    command_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    risk_level: str = "low"
    requires_confirmation: bool = False
    created_by: str = ""


@dataclass(slots=True)
class AgentEvent:
    source: str
    event_type: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    agent: str
    ok: bool = True
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    commands: list[AgentCommand] = field(default_factory=list)
