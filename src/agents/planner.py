"""Planner agent: decomposes a legal query into an executable, ordered plan.

Produces a JSON/DAG-style task list (sequential dependency here, easily
extended to a true DAG) so downstream agents and the UI can show explainable,
step-by-step progress instead of a single opaque LLM call.
"""
import json
import re

from src.agents.state import AgentState, PlanStep
from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_SPLIT_PATTERN = re.compile(r"\band\b|;|\n", re.IGNORECASE)

PLANNER_PROMPT = """You are a legal research planner. Break the user's question into
at most {max_steps} concrete, independently-searchable sub-questions needed to answer it
fully. Respond ONLY with a JSON array of strings, no commentary.

Question: {query}
"""


def _heuristic_plan(query: str, max_steps: int) -> list[str]:
    """Fallback planner used when no LLM is configured: split on conjunctions."""
    parts = [p.strip(" ?.") for p in _SPLIT_PATTERN.split(query) if p.strip()]
    parts = parts or [query.strip()]
    return parts[:max_steps]


def _llm_plan(query: str, max_steps: int) -> list[str]:
    from src.rag_watsonx import _get_model  # lazy import; may raise ConfigurationError

    model = _get_model()
    prompt = PLANNER_PROMPT.format(query=query, max_steps=max_steps)
    raw = model.generate_text(prompt=prompt)
    try:
        steps = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        cleaned = [str(s).strip() for s in steps if str(s).strip()]
        if cleaned:
            return cleaned[:max_steps]
    except (ValueError, json.JSONDecodeError):
        logger.warning("Planner LLM returned non-JSON output, falling back to heuristic plan")
    return _heuristic_plan(query, max_steps)


def plan(state: AgentState) -> AgentState:
    settings = get_settings()
    max_steps = settings.max_plan_steps

    try:
        steps = _llm_plan(state["query"], max_steps)
        source = "llm"
    except Exception as exc:
        logger.info("Falling back to heuristic planner: %s", exc)
        steps = _heuristic_plan(state["query"], max_steps)
        source = "heuristic"

    plan_steps: list[PlanStep] = [
        {"id": i, "description": step, "status": "pending"} for i, step in enumerate(steps)
    ]

    state["plan"] = plan_steps
    state["trace"].append({
        "agent": "planner",
        "action": "decompose_query",
        "detail": f"Generated {len(plan_steps)} step(s) via {source} planning: {steps}",
    })
    return state
