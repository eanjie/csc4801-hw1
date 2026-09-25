"""
llm_client.py -- shared LLM client interface for all CSC4801 agent assignments.

DO NOT MODIFY THIS FILE. It is identical across HW1-HW4 and the autograder
depends on its exact behavior.

This module defines a minimal, Anthropic-Messages-API-shaped interface so
that the `Agent` you write works unmodified whether it is driven by:

  * `FakeLLMClient` -- a fully scripted, deterministic client used by every
    public and hidden test in this course. No network access or API key is
    ever required to run the tests.

  * `AnthropicLLMClient` -- a thin wrapper around the real `anthropic`
    Python SDK, provided so you can also run your agent against a real
    model for your own exploration/demo. The autograder never calls this.

Understanding the three block types below is the whole trick to this
assignment:

  TextBlock       -- the model "talking" (reasoning or a final answer)
  ToolUseBlock    -- the model asking YOU to run a tool, with arguments
  ToolResultBlock -- YOU reporting a tool's output back to the model

A single LLM turn returns a list of blocks (usually some mix of TextBlock
and ToolUseBlock). Your agent turns ToolUseBlocks into ToolResultBlocks by
actually executing the tool, then sends those results back as the next
message so the model can continue.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclasses.dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = dataclasses.field(default_factory=dict)
    type: str = "tool_use"


@dataclasses.dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclasses.dataclass
class LLMResponse:
    content: list[ContentBlock]
    # One of: "end_turn" (model is done), "tool_use" (model wants tool
    # results before continuing), "max_tokens" (response got cut off).
    stop_reason: str


class LLMClient:
    """Abstract interface every agent in this course is built against."""

    def create_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """
    A scripted, deterministic stand-in for a real LLM.

    Example
    -------
        llm = FakeLLMClient([
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="calculator", input={"expression": "2+2"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="The answer is 4.")], stop_reason="end_turn"),
        ])

    Each call to `create_message` pops the next scripted response off the
    front of the queue. `self.calls` records every `(system, messages,
    tools)` triple your agent called it with, in order, so tests can assert
    that your agent built the right context at each step.
    """

    def __init__(self, script: list[LLMResponse]):
        self._script: list[LLMResponse] = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create_message(self, system, messages, tools=None) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                # store a deep-enough copy so later mutation of `messages`
                # by the agent doesn't retroactively change what we recorded
                "messages": [dict(m) for m in messages],
                "tools": tools,
            }
        )
        if not self._script:
            raise AssertionError(
                "FakeLLMClient script exhausted: your agent called the LLM "
                "more times than the test expected. This usually means your "
                "loop isn't stopping when stop_reason == 'end_turn'."
            )
        return self._script.pop(0)


def make_anthropic_llm_client(
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """
    Real LLM client backed by the `anthropic` package. Imported lazily so
    that the test suite (which only ever uses FakeLLMClient) never requires
    the `anthropic` package or an API key to be installed/set.

    `base_url` lets you point the client at any Anthropic-Messages-API
    compatible endpoint instead of api.anthropic.com -- e.g. Xiaomi's MiMo
    or another compatible provider/proxy.
    """
    import anthropic  # local import: only needed for real (non-test) usage

    class _AnthropicLLMClient(LLMClient):
        def __init__(self):
            self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
            self._model = model

        def create_message(self, system, messages, tools=None) -> LLMResponse:
            # The agent stores TextBlock / ToolUseBlock / ToolResultBlock
            # dataclass objects in its history; the SDK only serializes
            # plain dicts, so convert every block on the way out.
            wire_messages = [
                {
                    "role": m["role"],
                    "content": [
                        dataclasses.asdict(b) if dataclasses.is_dataclass(b) else b
                        for b in m["content"]
                    ],
                }
                for m in messages
            ]
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=wire_messages,
                tools=tools or [],
            )
            content: list[ContentBlock] = []
            for block in resp.content:
                if block.type == "text":
                    content.append(TextBlock(text=block.text))
                elif block.type == "tool_use":
                    content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
            return LLMResponse(content=content, stop_reason=resp.stop_reason)

    return _AnthropicLLMClient()
