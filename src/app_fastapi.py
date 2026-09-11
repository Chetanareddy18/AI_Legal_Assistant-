"""Production-oriented FastAPI service for the AI Legal Document Assistant.

Exposes both the classic single-shot RAG endpoint (``/ask``) and the new
multi-agent, planner-driven pipeline (``/agent/ask``), plus knowledge-graph
and multi-modal search endpoints. Includes API-key auth, rate limiting, CORS,
structured error handling, and liveness/readiness probes for container
orchestration.
"""
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import get_settings
from src.exceptions import ConfigurationError, LegalAssistantError, RetrievalError
from src.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.app_name,
    description="RAG + multi-agent legal document assistant (LangChain, WatsonX/FAISS, LangGraph).",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("%s %s -> %s (%.1fms) [%s]", request.method, request.url.path, response.status_code, duration_ms, request_id)
    return response


@app.exception_handler(LegalAssistantError)
async def legal_assistant_error_handler(request: Request, exc: LegalAssistantError):
    status_code = status.HTTP_400_BAD_REQUEST if isinstance(exc, RetrievalError) else status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, ConfigurationError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content={"error": type(exc).__name__, "message": str(exc)})


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Simple static API-key guard. No-op when API_KEY isn't configured (dev mode)."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class GraphQueryRequest(BaseModel):
    entity: str = Field(..., min_length=1, max_length=200)
    hops: int = Field(default=1, ge=1, le=4)


@app.get("/")
def home():
    return {"message": f"{settings.app_name} is running", "version": app.version}


@app.get("/health")
def health():
    """Liveness probe: process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe: dependencies (vector store) are reachable."""
    try:
        from src.rag_pipeline import retriever

        retriever.get_relevant_documents("healthcheck", k=1)
        return {"status": "ready", "vector_backend": settings.vector_backend}
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(exc)})


@app.post("/ask", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def ask_question(request: Request, data: QueryRequest):
    """Single-shot grounded RAG answer (legacy endpoint)."""
    from src.rag_watsonx import generate_answer

    result = generate_answer(data.query, data.top_k)
    return {"answer": result["answer"], "passages": result["passages"]}


@app.post("/agent/ask", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def agent_ask(request: Request, data: AgentQueryRequest):
    """Multi-agent pipeline: Planner -> Researcher -> Reviewer with adaptive re-research."""
    from src.agents.orchestrator import run_agentic_query

    return run_agentic_query(data.query)


@app.post("/graph/query", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def graph_query(request: Request, data: GraphQueryRequest):
    """Knowledge-graph neighborhood lookup for a legal entity/term."""
    from src.graph.knowledge_graph import query_related_entities

    return {"entity": data.entity, "related": query_related_entities(data.entity, hops=data.hops)}


@app.post("/multimodal/search", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def multimodal_search_endpoint(request: Request, data: QueryRequest):
    """Combined text + chart/diagram image retrieval."""
    from src.multimodal import multimodal_search

    return multimodal_search(data.query, text_top_k=data.top_k)

