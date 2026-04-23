from .contracts import AgentCommand, AgentEvent, AgentResult, AgentTask

__all__ = [
    "AgentCommand",
    "AgentEvent",
    "AgentResult",
    "AgentTask",
    "ConversationAgent",
    "ExecutionAgent",
    "MaterialSpecialistAgent",
    "ReimbursementAgentOrchestrator",
    "TravelSpecialistAgent",
]


def __getattr__(name: str):
    if name == "ConversationAgent":
        from .conversation_agent import ConversationAgent

        return ConversationAgent
    if name == "ExecutionAgent":
        from .execution_agent import ExecutionAgent

        return ExecutionAgent
    if name == "MaterialSpecialistAgent":
        from .material_specialist_agent import MaterialSpecialistAgent

        return MaterialSpecialistAgent
    if name == "ReimbursementAgentOrchestrator":
        from .orchestrator import ReimbursementAgentOrchestrator

        return ReimbursementAgentOrchestrator
    if name == "TravelSpecialistAgent":
        from .travel_specialist_agent import TravelSpecialistAgent

        return TravelSpecialistAgent
    raise AttributeError(name)
