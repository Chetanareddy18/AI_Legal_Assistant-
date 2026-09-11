"""Grounded answer generation via IBM WatsonX foundation models."""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.exceptions import ConfigurationError, GenerationError, RetrievalError
from src.logging_config import get_logger
from src.rag_pipeline import retriever

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a legal expert AI assistant.
Use ONLY the context below.
If the answer is not in the context, say:
"The information is not available in the provided documents."
Context:
{context}
Question:
{query}

Answer:"""


@lru_cache
def _get_model():
    """Lazily construct the WatsonX model client (avoids import-time failures
    when credentials aren't configured, e.g. during tests or FAISS-only demos)."""
    settings = get_settings()
    if not (settings.watsonx_api_key and settings.watsonx_url and settings.watsonx_project_id):
        raise ConfigurationError(
            "WATSONX_API_KEY, WATSONX_URL and WATSONX_PROJECT_ID must be set to generate answers"
        )

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Model

    creds = Credentials(api_key=settings.watsonx_api_key, url=settings.watsonx_url)
    return Model(
        model_id=settings.watsonx_model_id,
        credentials=creds,
        project_id=settings.watsonx_project_id,
    )


def build_prompt(query: str, docs) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs) or "No relevant context found."
    return SYSTEM_PROMPT.format(context=context, query=query)


def generate_answer(query: str, top_k: int = 3) -> dict:
    if not query or not query.strip():
        raise RetrievalError("Query must not be empty")

    try:
        docs = retriever.get_relevant_documents(query, k=top_k)
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc

    prompt = build_prompt(query, docs)

    try:
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        model = _get_model()
        params = {GenParams.MAX_NEW_TOKENS: 350, GenParams.TEMPERATURE: 0.2}
        result = model.generate_text(prompt=prompt, params=params)
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("WatsonX generation failed")
        raise GenerationError(str(exc)) from exc

    return {
        "answer": result,
        "passages": [
            {
                "text": d.page_content,
                "source": d.metadata.get("source", ""),
                "score": d.metadata.get("score", None),
            }
            for d in docs
        ],
    }


def run_rag(question: str) -> str:
    return generate_answer(question)["answer"]


if __name__ == "__main__":
    out = run_rag("What did the court order about stray dogs?")
    print("\nANSWER:\n", out)
