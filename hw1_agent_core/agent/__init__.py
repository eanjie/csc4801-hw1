"""HW1 agent package."""
from .core import Agent, MaxIterationsExceeded
from .llm_client import (
    FakeLLMClient,
    LLMClient,
    LLMResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    make_anthropic_llm_client,
)
from .tools import CalculatorTool, Tool, UppercaseTool, WordCountTool, to_anthropic_schema

__all__ = [
    "Agent",
    "MaxIterationsExceeded",
    "FakeLLMClient",
    "LLMClient",
    "LLMResponse",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "make_anthropic_llm_client",
    "Tool",
    "CalculatorTool",
    "UppercaseTool",
    "WordCountTool",
    "to_anthropic_schema",
]
