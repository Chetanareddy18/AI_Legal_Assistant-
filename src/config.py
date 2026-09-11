"""Centralized application configuration.

All environment-driven settings live here so the rest of the codebase never
touches ``os.environ`` directly. Uses pydantic-settings for validation and
sensible defaults so the app degrades gracefully in local/dev environments.
"""
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (parent of src/), not the process's
# current working directory, so it loads correctly regardless of where the
# app is launched from (uvicorn, streamlit, pytest, a different cwd, etc.).
_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "AI Legal Document Assistant"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_key: str | None = Field(default=None, description="Static API key required on protected endpoints")
    allowed_origins: str = "*"
    rate_limit: str = "60/minute"

    # --- Vector store backend ---
    # "faiss" runs fully offline/local with no external service dependency.
    # "pinecone" requires PINECONE_API_KEY and is used for cloud-scale deployments.
    vector_backend: Literal["faiss", "pinecone"] = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "legal-assistant-index"

    faiss_index_dir: str = "embeddings/faiss_index"

    # --- WatsonX generation ---
    watsonx_api_key: str | None = None
    watsonx_url: str | None = None
    watsonx_project_id: str | None = None
    watsonx_model_id: str = "mistralai/mistral-medium-2505"

    # --- IBM NLU (optional, used for entity/keyword extraction) ---
    ibm_nlu_api_key: str | None = None
    ibm_nlu_url: str | None = None

    # --- Agentic system ---
    max_plan_steps: int = 6
    max_reasoning_hops: int = 3
    agent_top_k: int = 4

    # --- Multi-modal ---
    clip_model_name: str = "clip-ViT-B-32"
    enable_multimodal: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
