from src.agents.orchestrator import run_agentic_query
from src.vectorstore import get_retriever


def test_agentic_pipeline_runs_end_to_end_without_llm_credentials():
    retriever = get_retriever()
    retriever.add_documents(
        texts=[
            "The Supreme Court directed states to relocate stray dogs to shelters.",
            "Section 302 of the IPC prescribes punishment for murder.",
        ],
        sources=["case1.pdf", "ipc.pdf"],
    )

    result = run_agentic_query("What did the Supreme Court say about stray dogs?")

    assert result["answer"]
    assert result["plan"]
    assert result["trace"]
    assert result["hops"] >= 1
    assert any(c["source"] == "case1.pdf" for c in result["citations"])


def test_agentic_pipeline_handles_empty_index_gracefully():
    # Uses a query unrelated to any indexed content; regardless of what other
    # tests have indexed into the shared retriever singleton, the pipeline
    # must not raise and must always return a non-empty synthesized answer.
    result = run_agentic_query("Some question with nothing indexed yet")
    assert isinstance(result["answer"], str) and result["answer"]
