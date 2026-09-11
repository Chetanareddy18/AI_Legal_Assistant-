"""MCP-style tool registry.

Mirrors the Model Context Protocol's tool-calling shape (name, description,
JSON input schema, handler) so agents can discover and invoke tools in a
uniform, model-agnostic way without hardcoding integrations. Tools are
registered once and looked up by name at call time, exactly like an MCP
client would list/call tools exposed by an MCP server.
"""
import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.exceptions import ToolExecutionError
from src.logging_config import get_logger

logger = get_logger(__name__)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ToolExecutionError(f"Unknown tool: {name}")
        try:
            return self._tools[name].handler(**kwargs)
        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            raise ToolExecutionError(f"Tool '{name}' failed: {exc}") from exc


def _safe_eval_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_expr(node.left), _safe_eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_expr(node.operand))
    raise ToolExecutionError("Expression contains unsupported operations")


def _calculator_tool(expression: str) -> float:
    """Evaluate a pure arithmetic expression safely (no exec/eval on raw strings)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval_expr(tree.body)
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(f"Invalid expression: {exc}") from exc


def _document_search_tool(query: str, top_k: int = 4) -> list[dict]:
    from src.rag_pipeline import rag_search

    return rag_search(query, top_k=top_k)


def _summarizer_tool(text: str, max_sentences: int = 3) -> str:
    """Lightweight extractive summarizer (no external model dependency)."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")


def _knowledge_graph_tool(entity: str, hops: int = 1) -> list[dict]:
    from src.graph.knowledge_graph import query_related_entities

    return query_related_entities(entity, hops=hops)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(Tool(
        name="document_search",
        description="Semantic search over indexed legal documents. Returns top-k relevant passages.",
        input_schema={"query": "string", "top_k": "integer (default 4)"},
        handler=_document_search_tool,
    ))
    registry.register(Tool(
        name="calculator",
        description="Evaluate a safe arithmetic expression, e.g. for computing damages, dates, or penalties.",
        input_schema={"expression": "string, e.g. '15000 * 1.18'"},
        handler=_calculator_tool,
    ))
    registry.register(Tool(
        name="summarizer",
        description="Extractive summary of a block of text into N sentences.",
        input_schema={"text": "string", "max_sentences": "integer (default 3)"},
        handler=_summarizer_tool,
    ))
    registry.register(Tool(
        name="knowledge_graph_lookup",
        description="Look up entities related to a given legal entity/term in the document knowledge graph.",
        input_schema={"entity": "string", "hops": "integer (default 1)"},
        handler=_knowledge_graph_tool,
    ))
    return registry


default_registry = build_default_registry()
