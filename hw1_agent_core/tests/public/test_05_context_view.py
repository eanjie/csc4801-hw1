"""Public test 5: context management in prepare_context. The stored history
keeps every tool output in full, but the view sent to the LLM replaces all
but the most recent `keep_recent_tool_results` tool results with
CLEARED_TOOL_RESULT (lecture 03, Tool Count Budget)."""
from agent.core import CLEARED_TOOL_RESULT, Agent
from agent.llm_client import FakeLLMClient, LLMResponse, TextBlock, ToolUseBlock
from agent.tools import CalculatorTool


def test_old_tool_results_are_cleared_in_the_llm_view():
    llm = FakeLLMClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="calculator", input={"expression": "1+1"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content=[ToolUseBlock(id="t2", name="calculator", input={"expression": "2+2"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="2 and 4")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(llm_client=llm, tools={"calculator": CalculatorTool()}, keep_recent_tool_results=1)

    assert agent.run("Add 1+1, then 2+2.") == "2 and 4"

    # call 2: t1's result is the only one and the most recent -> sent in full
    results_call2 = [m for m in llm.calls[1]["messages"] if m["role"] == "user"][1:]
    assert results_call2[0]["content"][0].content == "2"

    # call 3: t2's result is now the most recent one and is kept; t1's is
    # cleared but keeps its tool_use_id so the pairing stays valid
    results_call3 = [m for m in llm.calls[2]["messages"] if m["role"] == "user"][1:]
    assert len(results_call3) == 2
    old, new = results_call3[0]["content"][0], results_call3[1]["content"][0]
    assert old.tool_use_id == "t1"
    assert old.content == CLEARED_TOOL_RESULT
    assert new.tool_use_id == "t2"
    assert new.content == "4"

    # the stored transcript is untouched: both results still hold real output
    stored = [m for m in agent.messages if m["role"] == "user"][1:]
    assert [m["content"][0].content for m in stored] == ["2", "4"]
