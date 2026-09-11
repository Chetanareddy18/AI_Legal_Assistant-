import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force a fully offline configuration for the test suite: local FAISS backend,
# isolated temp index directory, no cloud credentials, no network calls.
os.environ["VECTOR_BACKEND"] = "faiss"
os.environ["FAISS_INDEX_DIR"] = os.path.join(tempfile.mkdtemp(prefix="legal_assistant_test_"), "faiss_index")
os.environ["WATSONX_API_KEY"] = ""
os.environ["WATSONX_URL"] = ""
os.environ["WATSONX_PROJECT_ID"] = ""
os.environ["API_KEY"] = ""

from src.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
