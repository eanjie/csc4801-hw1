"""Public test 2: two sequential tool calls (different tools) before a final answer."""
from agent.core import Agent
from agent.llm_client import FakeLLMClient, LLMResponse, TextBlock, ToolUseBlock
from agent.tools import CalculatorTool, UppercaseTool


def test_multi_turn_multiple_distinct_tools():
    llm = FakeLLMClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="calculator", input={"expression": "3*3"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content=[ToolUseBlock(id="t2", name="uppercase", input={"text": "done"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="Result: DONE")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(
        llm_client=llm,
        tools={"calculator": CalculatorTool(), "uppercase": UppercaseTool()},
    )

    result = agent.run("Compute 3*3 then shout 'done'.")

    assert result == "Result: DONE"
    assert llm.call_count == 3
    # conversation history must have grown: user prompt, assistant, tool
    # results, assistant, tool results (5 messages by the last call)
    assert len(llm.calls[-1]["messages"]) == 5
