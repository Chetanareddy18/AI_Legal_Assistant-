"""Reviewer/Coordinator agent: synthesizes the final grounded answer.

Responsible for:
  - Building the final answer from accumulated research notes (via WatsonX
    when configured, otherwise an extractive fallback).
  - A lightweight groundedness check: flags answer content that has no
    lexical overlap with retrieved evidence (explainability & trust).
  - Deciding whether another research hop is warranted (adaptive,
    multi-hop reasoning) up to ``max_hops``.
"""
import re

from src.agents.state import AgentState
from src.exceptions import ConfigurationError
from src.logging_config import get_logger

logger = get_logger(__name__)

REVIEW_PROMPT = """You are a legal reviewer. Using ONLY the evidence below, write a precise,
well-cited answer to the question. If evidence is insufficient, say so explicitly.

Question: {query}

Evidence:
{evidence}

Answer:"""


def _flatten_evidence(state: AgentState) -> list[dict]:
    flat = []
    for note in state["research_notes"]:
        flat.extend(note["evidence"])
    return flat


def _extractive_answer(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return "The information is not available in the provided documents."
    top = sorted(evidence, key=lambda e: e.get("score") or 0, reverse=True)[:3]
    return " ".join(e["text"][:400] for e in top)


def _llm_answer(query: str, evidence: list[dict]) -> str:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    from src.rag_watsonx import _get_model

    model = _get_model()
    evidence_block = "\n\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(evidence)) or "None"
    prompt = REVIEW_PROMPT.format(query=query, evidence=evidence_block)
    params = {GenParams.MAX_NEW_TOKENS: 400, GenParams.TEMPERATURE: 0.2}
    return model.generate_text(prompt=prompt, params=params)


def _groundedness_ratio(answer: str, evidence: list[dict]) -> float:
    """Fraction of answer sentences with meaningful word overlap with evidence."""
    context = " ".join(e["text"] for e in evidence).lower()
    context_words = set(re.findall(r"[a-z]{4,}", context))
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 1.0

    grounded = 0
    for sentence in sentences:
        words = set(re.findall(r"[a-z]{4,}", sentence.lower()))
        if not words or (words & context_words):
            grounded += 1
    return grounded / len(sentences)


def review(state: AgentState) -> AgentState:
    evidence = _flatten_evidence(state)

    try:
        answer = _llm_answer(state["query"], evidence)
        source = "llm"
    except ConfigurationError:
        answer = _extractive_answer(state["query"], evidence)
        source = "extractive_fallback"
    except Exception as exc:
        logger.warning("Reviewer LLM generation failed, using extractive fallback: %s", exc)
        answer = _extractive_answer(state["query"], evidence)
        source = "extractive_fallback"

    groundedness = _groundedness_ratio(answer, evidence)

    state["final_answer"] = answer
    state["citations"] = [
        {"source": e.get("source", ""), "score": e.get("score"), "excerpt": e["text"][:200]}
        for e in evidence
    ]
    state["hops"] += 1
    # Ask for another hop if evidence was too thin/ungrounded and we haven't hit the cap.
    state["needs_more_research"] = (
        (groundedness < 0.5 or not evidence) and state["hops"] < state["max_hops"]
    )

    state["trace"].append({
        "agent": "reviewer",
        "action": "synthesize_answer",
        "detail": f"Answer via {source}; groundedness={groundedness:.2f}; hop={state['hops']}/{state['max_hops']}",
    })
    return state
