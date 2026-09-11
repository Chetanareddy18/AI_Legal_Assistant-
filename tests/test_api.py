from fastapi.testclient import TestClient

from src.app_fastapi import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_home_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_ready_endpoint_reports_backend():
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)


def test_ask_rejects_empty_query():
    resp = client.post("/ask", json={"query": "", "top_k": 3})
    assert resp.status_code == 422


def test_agent_ask_runs_pipeline():
    from src.vectorstore import get_retriever

    get_retriever().add_documents(["A legal clause about indemnity."], ["doc.pdf"])

    resp = client.post("/agent/ask", json={"query": "What does the clause say about indemnity?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "trace" in body
