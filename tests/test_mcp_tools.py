import pytest

from src.agents.mcp_tools import Tool, ToolRegistry, build_default_registry
from src.exceptions import ToolExecutionError


def test_calculator_tool_evaluates_safe_expression():
    registry = build_default_registry()
    assert registry.call("calculator", expression="15000 * 1.18") == pytest.approx(17700.0)


def test_calculator_tool_rejects_unsafe_expression():
    registry = build_default_registry()
    with pytest.raises(ToolExecutionError):
        registry.call("calculator", expression="__import__('os').system('echo hi')")


def test_summarizer_tool_truncates_sentences():
    registry = build_default_registry()
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    summary = registry.call("summarizer", text=text, max_sentences=2)
    assert summary.count(".") == 2


def test_unknown_tool_raises():
    registry = build_default_registry()
    with pytest.raises(ToolExecutionError):
        registry.call("does_not_exist")


def test_registry_lists_registered_tools():
    registry = ToolRegistry()
    registry.register(Tool(name="noop", description="does nothing", input_schema={}, handler=lambda: None))
    names = [t["name"] for t in registry.list_tools()]
    assert names == ["noop"]
