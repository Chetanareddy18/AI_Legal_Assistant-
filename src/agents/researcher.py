"""Researcher agent: executes each pending plan step against the tool registry."""
from src.agents.mcp_tools import default_registry
from src.agents.state import AgentState
from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def research(state: AgentState) -> AgentState:
    settings = get_settings()

    for step in state["plan"]:
        if step["status"] != "pending":
            continue
        try:
            evidence = default_registry.call("document_search", query=step["description"], top_k=settings.agent_top_k)
            step["status"] = "done"
        except Exception as exc:
            logger.warning("Research step %s failed: %s", step["id"], exc)
            evidence = []
            step["status"] = "failed"

        state["research_notes"].append({
            "step_id": step["id"],
            "query": step["description"],
            "evidence": evidence,
        })
        state["trace"].append({
            "agent": "researcher",
            "action": "retrieve_evidence",
            "detail": f"Step {step['id']} ('{step['description']}') -> {len(evidence)} passage(s) [{step['status']}]",
        })

    return state
