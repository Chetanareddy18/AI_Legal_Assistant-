"""LangGraph orchestrator wiring Planner -> Researcher -> Reviewer.

Implements the JD's "Multi-Agent Systems" and "Planner & Long-Horizon Task
Decomposition" requirements: a Planner decomposes the query into sub-tasks,
a Researcher executes them via tools, and a Reviewer/Coordinator synthesizes
a grounded answer and can trigger additional research hops (bounded by
``max_reasoning_hops``) for multi-hop reasoning.
"""
from langgraph.graph import END, StateGraph

from src.agents.planner import plan
from src.agents.researcher import research
from src.agents.reviewer import review
from src.agents.state import AgentState
from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def _widen_research(state: AgentState) -> AgentState:
    """Add one broader, whole-query research step for another hop."""
    next_id = len(state["plan"])
    state["plan"].append({"id": next_id, "description": state["query"], "status": "pending"})
    state["trace"].append({
        "agent": "planner",
        "action": "widen_search",
        "detail": f"Groundedness/coverage low, added broad re-query as step {next_id}",
    })
    return state


def _route_after_review(state: AgentState) -> str:
    return "widen" if state.get("needs_more_research") else "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan)
    graph.add_node("researcher", research)
    graph.add_node("reviewer", review)
    graph.add_node("widen", _widen_research)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reviewer")
    graph.add_conditional_edges("reviewer", _route_after_review, {"widen": "widen", "end": END})
    graph.add_edge("widen", "researcher")

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agentic_query(query: str) -> dict:
    """Entry point used by the API/UI: runs the full multi-agent pipeline."""
    settings = get_settings()
    initial_state: AgentState = {
        "query": query,
        "plan": [],
        "research_notes": [],
        "hops": 0,
        "max_hops": settings.max_reasoning_hops,
        "final_answer": "",
        "citations": [],
        "trace": [],
        "needs_more_research": False,
    }

    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)

    logger.info("Agentic run complete for query=%r in %d hop(s)", query, final_state["hops"])
    return {
        "query": query,
        "answer": final_state["final_answer"],
        "citations": final_state["citations"],
        "plan": final_state["plan"],
        "trace": final_state["trace"],
        "hops": final_state["hops"],
    }
