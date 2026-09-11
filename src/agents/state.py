"""Shared state schema for the multi-agent LangGraph pipeline."""
from typing import TypedDict


class PlanStep(TypedDict):
    id: int
    description: str
    status: str  # "pending" | "done" | "failed"


class ResearchNote(TypedDict):
    step_id: int
    query: str
    evidence: list[dict]  # [{text, source, score}]


class TraceEvent(TypedDict):
    agent: str
    action: str
    detail: str


class AgentState(TypedDict):
    query: str
    plan: list[PlanStep]
    research_notes: list[ResearchNote]
    hops: int
    max_hops: int
    final_answer: str
    citations: list[dict]
    trace: list[TraceEvent]
    needs_more_research: bool
