from src.graph.knowledge_graph import (
    build_graph_from_chunks,
    query_related_entities,
    save_graph,
)


def test_build_graph_links_cooccurring_entities():
    chunks = [
        "The Supreme Court referred to Section 302 in its ruling.",
        "The High Court cited an unrelated Article 21 matter.",
    ]
    sources = ["a.pdf", "b.pdf"]

    graph = build_graph_from_chunks(chunks, sources)

    assert graph.has_node("Supreme Court")
    assert graph.has_edge("Supreme Court", "Section 302")


def test_query_related_entities_respects_hops(tmp_path):
    chunks = ["The Supreme Court referred to Section 302 in its ruling."]
    graph = build_graph_from_chunks(chunks, ["a.pdf"])
    path = tmp_path / "kg.json"
    save_graph(graph, path=str(path))

    related = query_related_entities("Supreme Court", hops=1, path=str(path))

    assert any(r["entity"] == "Section 302" for r in related)


def test_query_unknown_entity_returns_empty(tmp_path):
    path = tmp_path / "kg.json"
    save_graph(build_graph_from_chunks([], []), path=str(path))
    assert query_related_entities("Nonexistent", path=str(path)) == []
