"""Custom exception hierarchy for predictable, typed error handling."""


class LegalAssistantError(Exception):
    """Base class for all application-specific errors."""


class RetrievalError(LegalAssistantError):
    """Raised when the vector store fails to return results."""


class GenerationError(LegalAssistantError):
    """Raised when the LLM generation step fails."""


class DocumentProcessingError(LegalAssistantError):
    """Raised when PDF/text/image extraction or chunking fails."""


class AgentExecutionError(LegalAssistantError):
    """Raised when a step in the multi-agent pipeline fails."""


class ToolExecutionError(LegalAssistantError):
    """Raised when an agent tool invocation fails."""


class ConfigurationError(LegalAssistantError):
    """Raised when required configuration/secrets are missing."""
