"""Lightweight knowledge-graph construction and querying over indexed documents.

Builds an entity-relation graph with NetworkX from document text using a
regex/heuristic NER pass by default, with an automatic upgrade to spaCy's
statistical NER if the ``en_core_web_sm`` model is installed. The graph is
persisted as JSON so it can be reloaded without re-processing documents.
"""
import json
import os
import re
from itertools import combinations

import networkx as nx

from src.logging_config import get_logger

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GRAPH_DIR = os.path.join(ROOT, "embeddings", "graph")
GRAPH_PATH = os.path.join(GRAPH_DIR, "knowledge_graph.json")

# Heuristic fallback patterns for legal-domain entities when spaCy isn't available.
_ENTITY_PATTERNS = {
    "COURT": re.compile(r"\b(Supreme Court|High Court|District Court|Tribunal)\b", re.IGNORECASE),
    "STATUTE": re.compile(r"\b(Section|Article|Act)\s+\d+[A-Za-z]*\b", re.IGNORECASE),
    "ORG": re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,3}\s(?:Ltd|Inc|Corporation|Corp|Company|Govt|Government))\b"),
    "DATE": re.compile(r"\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b", re.IGNORECASE),
}


def _extract_entities_regex(text: str) -> list[str]:
    entities = set()
    for pattern in _ENTITY_PATTERNS.values():
        entities.update(m.group(0).strip() for m in pattern.finditer(text))
    return sorted(entities)


def _extract_entities_spacy(text: str, nlp) -> list[str]:
    doc = nlp(text[:100_000])  # guard against pathologically long inputs
    return sorted({ent.text.strip() for ent in doc.ents if ent.label_ in
                   {"ORG", "LAW", "GPE", "PERSON", "DATE", "NORP"}})


def _load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_entities(text: str) -> list[str]:
    nlp = _load_spacy()
    if nlp is not None:
        return _extract_entities_spacy(text, nlp)
    return _extract_entities_regex(text)


def build_graph_from_chunks(chunks: list[str], sources: list[str]) -> nx.Graph:
    """Co-occurrence graph: entities appearing in the same chunk are linked.

    Each chunk contributes a small clique of its entities; edge weights
    accumulate co-occurrence counts, and node/edge metadata retain the
    source document for traceability (explainability requirement).
    """
    graph = nx.Graph()

    for chunk, source in zip(chunks, sources, strict=True):
        entities = extract_entities(chunk)
        for entity in entities:
            if graph.has_node(entity):
                graph.nodes[entity]["sources"].add(source)
            else:
                graph.add_node(entity, sources={source})

        for a, b in combinations(sorted(entities), 2):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)

    logger.info("Built knowledge graph with %d nodes / %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def save_graph(graph: nx.Graph, path: str = GRAPH_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(graph, edges="edges")
    # sets aren't JSON serializable
    for node in data["nodes"]:
        if "sources" in node and isinstance(node["sources"], set):
            node["sources"] = sorted(node["sources"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str = GRAPH_PATH) -> nx.Graph:
    if not os.path.exists(path):
        return nx.Graph()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def query_related_entities(entity: str, hops: int = 1, path: str = GRAPH_PATH) -> list[dict]:
    """Return entities within ``hops`` graph-distance of ``entity``, ranked by edge weight."""
    graph = load_graph(path)
    if entity not in graph:
        return []

    related = nx.single_source_shortest_path_length(graph, entity, cutoff=hops)
    results = []
    for node, distance in related.items():
        if node == entity:
            continue
        results.append({
            "entity": node,
            "distance": distance,
            "sources": sorted(graph.nodes[node].get("sources", [])),
        })
    return sorted(results, key=lambda r: r["distance"])
