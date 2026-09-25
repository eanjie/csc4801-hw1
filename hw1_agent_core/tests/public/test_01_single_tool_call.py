"""Public test 1: a single tool call, then a final text answer."""
from agent.core import Agent
from agent.llm_client import FakeLLMClient, LLMResponse, TextBlock, ToolUseBlock
from agent.tools import CalculatorTool


def test_single_tool_call_then_final_answer():
    llm = FakeLLMClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="calculator", input={"expression": "2+2"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="The answer is 4.")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(llm_client=llm, tools={"calculator": CalculatorTool()})

    result = agent.run("What is 2+2?")

    assert result == "The answer is 4."
    assert llm.call_count == 2

    # the second call to the LLM must include the tool's result in the
    # conversation history it was given
    second_call_messages = llm.calls[1]["messages"]
    tool_result_messages = [m for m in second_call_messages if m["role"] == "user"][1:]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0]["content"][0].content == "4"
    assert tool_result_messages[0]["content"][0].is_error is False
