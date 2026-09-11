"""Abstract retriever interface shared by all vector store backends."""
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """Common contract for any similarity-search backed retriever."""

    @abstractmethod
    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-k documents most relevant to ``query``."""
        raise NotImplementedError

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self.get_relevant_documents(query, k=k)

    @abstractmethod
    def add_documents(self, texts: list[str], sources: list[str]) -> int:
        """Embed and persist ``texts``. Returns number of chunks indexed."""
        raise NotImplementedError
