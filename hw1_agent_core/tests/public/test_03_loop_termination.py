"""Public test 3: the agent must not loop forever."""
import pytest

from agent.core import Agent, MaxIterationsExceeded
from agent.llm_client import FakeLLMClient, LLMResponse, ToolUseBlock
from agent.tools import CalculatorTool


def test_loop_terminates_after_max_iterations():
    # The fake LLM ALWAYS asks for another tool call and never stops.
    always_tool_use = [
        LLMResponse(
            content=[ToolUseBlock(id=f"t{i}", name="calculator", input={"expression": "1+1"})],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    llm = FakeLLMClient(always_tool_use)
    agent = Agent(llm_client=llm, tools={"calculator": CalculatorTool()}, max_iterations=5)

    with pytest.raises(MaxIterationsExceeded):
        agent.run("Keep going forever.")

    # must not have called the LLM more times than the budget allows
    assert llm.call_count == 5
