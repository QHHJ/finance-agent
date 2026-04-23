from __future__ import annotations

from .conversation_agent import ConversationAgent
from .contracts import AgentCommand, AgentResult, AgentTask
from .execution_agent import ExecutionAgent
from .material_specialist_agent import MaterialSpecialistAgent
from .travel_specialist_agent import TravelSpecialistAgent


class ReimbursementAgentOrchestrator:
    def __init__(
        self,
        *,
        conversation_agent: ConversationAgent | None = None,
        execution_agent: ExecutionAgent | None = None,
        travel_specialist: TravelSpecialistAgent | None = None,
        material_specialist: MaterialSpecialistAgent | None = None,
    ) -> None:
        self.conversation_agent = conversation_agent or ConversationAgent()
        self.execution_agent = execution_agent or ExecutionAgent()
        self.travel_specialist = travel_specialist or TravelSpecialistAgent()
        self.material_specialist = material_specialist or MaterialSpecialistAgent()

        self._agents = {
            self.conversation_agent.name: self.conversation_agent,
            self.execution_agent.name: self.execution_agent,
            self.travel_specialist.name: self.travel_specialist,
            self.material_specialist.name: self.material_specialist,
        }

    def run_task(self, task: AgentTask) -> AgentResult:
        agent = self._agents.get(str(task.agent or "").strip())
        if agent is None:
            return AgentResult(agent=str(task.agent or "unknown"), ok=False, summary=f"Unknown agent: {task.agent}")
        return agent.run(task)

    def execute_command(self, command: AgentCommand) -> AgentResult:
        return self.execution_agent.execute(command)
