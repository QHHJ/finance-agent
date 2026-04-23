from __future__ import annotations

from typing import Any

from app.usecases import travel_agent as travel_usecase

from .base import BaseAgent
from .contracts import AgentTask


class TravelSpecialistAgent(BaseAgent):
    name = "travel_specialist_agent"

    def run(self, task: AgentTask):
        objective = str(task.objective or "").strip()
        payload = dict(task.payload or {})

        if objective == "organize_materials":
            assignment, profiles = travel_usecase.organize_materials(
                payload.get("pool_files") or [],
                build_profile=payload["build_profile"],
                manual_overrides=payload.get("manual_overrides"),
                apply_overrides=payload.get("apply_overrides"),
            )
            status = travel_usecase.build_travel_agent_status(assignment)
            return self._result(
                summary="Travel materials organized.",
                payload={"assignment": assignment, "profiles": profiles, "status": status},
                events=[self._event("travel_materials_organized", "Travel materials organized.", {"profile_count": len(profiles)})],
            )

        if objective == "build_status":
            assignment = dict(payload.get("assignment") or {})
            status = travel_usecase.build_travel_agent_status(assignment)
            return self._result(
                summary="Travel status built.",
                payload={"status": status},
                events=[self._event("travel_status_built", "Travel status built.", {"missing_count": len(status.get('missing') or [])})],
            )

        if objective == "build_policy_context":
            context = travel_usecase.build_travel_policy_context(
                raw_text=str(payload.get("raw_text") or ""),
                top_k=int(payload.get("top_k") or 3),
            )
            return self._result(
                summary="Travel policy context built.",
                payload={"context": context},
            )

        if objective == "retrieve_case_hits":
            hits = travel_usecase.retrieve_travel_case_hits(
                query=str(payload.get("query") or ""),
                top_k=int(payload.get("top_k") or 4),
                metadata_filter=payload.get("metadata_filter"),
            )
            return self._result(
                summary=f"Travel case hits retrieved: {len(hits)}",
                payload={"hits": hits},
            )

        return self._result(ok=False, summary=f"Unsupported travel objective: {objective}")
