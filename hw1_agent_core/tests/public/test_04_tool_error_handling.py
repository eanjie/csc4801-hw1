"""Public test 4: tool failures must be reported back to the model, not crash the agent."""
from agent.core import Agent
from agent.llm_client import FakeLLMClient, LLMResponse, TextBlock, ToolUseBlock
from agent.tools import CalculatorTool


def test_tool_error_is_reported_not_raised():
    llm = FakeLLMClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="calculator", input={"expression": "2+"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="Sorry, that expression was invalid.")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(llm_client=llm, tools={"calculator": CalculatorTool()})

    result = agent.run("What is 2+?")  # malformed expression -> CalculatorTool raises

    assert result == "Sorry, that expression was invalid."
    tool_result_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "user"][1:]
    assert tool_result_messages[0]["content"][0].is_error is True


def test_unknown_tool_name_is_reported_not_raised():
    llm = FakeLLMClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="does_not_exist", input={})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="I could not find that tool.")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(llm_client=llm, tools={"calculator": CalculatorTool()})

    result = agent.run("Use a tool that doesn't exist.")

    assert result == "I could not find that tool."
    tool_result_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "user"][1:]
    assert tool_result_messages[0]["content"][0].is_error is True
