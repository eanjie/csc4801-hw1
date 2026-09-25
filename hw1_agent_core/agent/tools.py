"""
tools.py -- the Tool interface, two ready-made tools, and ONE tool for you
to implement (see the TODO on WordCountTool below).

A "tool" is anything your agent can hand arguments to and get a string
result back from. The LLM never runs code directly -- it asks your agent
to run a named tool with a JSON object of arguments (a `ToolUseBlock`),
and your agent is responsible for finding the tool by name and calling it.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result (or raise on failure)."""
        ...


def to_anthropic_schema(tool: Tool) -> dict[str, Any]:
    """Convert a Tool into the dict shape the `tools=[...]` API param expects."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


@dataclasses.dataclass
class CalculatorTool:
    """Evaluates a basic arithmetic expression, e.g. "2 + 2 * 3"."""

    name: str = "calculator"
    description: str = "Evaluate a basic arithmetic expression like '2 + 2 * 3'."
    input_schema: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
    )

    def execute(self, expression: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError(f"Illegal characters in expression: {expression!r}")
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 (sandboxed above)


@dataclasses.dataclass
class UppercaseTool:
    """Converts a piece of text to upper case."""

    name: str = "uppercase"
    description: str = "Convert a piece of text to upper case."
    input_schema: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    def execute(self, text: str) -> str:
        return text.upper()


# ---------------------------------------------------------------------------
# TODO(You): implement WordCountTool.
#
# It must count the number of whitespace-separated words in `text` and
# return the count *as a string* (all tools in this course return strings,
# matching how ToolResultBlock.content is always a string).
#
# Examples:
#   WordCountTool().execute(text="hello world")      -> "2"
#   WordCountTool().execute(text="  a   b  c ")        -> "3"
#   WordCountTool().execute(text="")                    -> "0"
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class WordCountTool:
    """Counts the number of words in a piece of text."""

    name: str = "word_count"
    description: str = "Count the number of whitespace-separated words in a piece of text."
    input_schema: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    def execute(self, text: str) -> str:
        return str(len(text.split()))
