"""
core.py -- THE MAIN ASSIGNMENT. Implement the agent loop described in
lecture 02 (Principles of Agents):

    Receive Prompt -> Initialize -> [Prepare Context -> Call LLM ->
    Process Response -> Call Tools -> Receive Tool Results -> Check Loop]
    -> End

Every method of `Agent` below is one step of that pipeline, and `run()` is
the loop that ties the steps together. Each docstring states the method's
CONTRACT -- what it is given, what it must return, and which invariants it
must maintain. It does not tell you how to write the code; that is the
assignment.

How the hidden tests use these contracts: they drive `run()` with a
scripted `FakeLLMClient`, they call the step methods directly by name,
they substitute individual step methods and check that the rest of the
loop reacts, and they inspect both `self.messages` and the exact arguments
your agent passed to the LLM client (`FakeLLMClient.calls`). A loop that
returns the right final string but violates a contract along the way will
lose points.

The message/block formats you must produce are the Anthropic Messages API
shapes summarized in README section 2 and in `agent/llm_client.py`; the
client interface is `LLMClient.create_message`. You should not need to
change any other file (tools.py has one small TODO of its own; everything
else is off-limits).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm_client import LLMClient, LLMResponse, TextBlock, ToolResultBlock, ToolUseBlock
from .tools import Tool, to_anthropic_schema


class MaxIterationsExceeded(Exception):
    """Raised by run() when the agent loop exhausts its iteration budget."""


# What prepare_context() sends in place of a tool result that has aged out
# of the keep_recent_tool_results budget (lecture 03, Tool Count
# budget). Tests import this constant; do not change its value.
CLEARED_TOOL_RESULT = "[Old tool result content cleared]"


@dataclass
class Agent:
    llm_client: LLMClient
    tools: dict[str, Tool]
    system_prompt: str = "You are a helpful coding assistant."
    max_iterations: int = 10
    # Context management (Step 3): how many of the most recent tool results
    # are sent to the LLM in full. Older ones are cleared from the view --
    # see prepare_context().
    keep_recent_tool_results: int = 5

    # The running conversation. Populated by initialize(); every step that
    # records something records it here, in API message shape.
    messages: list[dict[str, Any]] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    # Part 1: Input Processing
    # ------------------------------------------------------------------
    def receive_prompt(self, user_prompt: str) -> dict[str, Any]:
        """
        Step 1 (Receive Prompt): convert the raw user string into ONE
        message in API message shape: a user-role message whose content is
        a list holding a single text block that carries `user_prompt`
        verbatim (see README section 2 for the text-block shape).

        Return that message. This step only builds the message; it must
        not modify self.messages.
        """
        # TODO: implement
        raise NotImplementedError

    def initialize(self, user_prompt: str) -> None:
        """
        Step 2 (Initialize): start a fresh conversation for `user_prompt`.
        After this call, self.messages must contain exactly one message --
        the one produced by receive_prompt -- and nothing left over from any
        earlier run() on the same Agent instance.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Part 2: Main Loop
    # ------------------------------------------------------------------
    def prepare_context(self) -> list[dict[str, Any]]:
        """
        Step 3 (Prepare Context): build and return the list of messages
        that will be sent to the LLM on this turn. This is where context
        management (lecture 03) happens. call_llm() must obtain its
        messages by calling this method rather than reading self.messages
        itself -- the hidden tests substitute prepare_context() and check
        that what it returns is exactly what reaches the LLM client.

        This assignment implements one of the lecture's strategies, a
        tool count budget:

          - Count the ToolResultBlocks in the conversation from the most
            recent one backwards. The newest self.keep_recent_tool_results
            of them are sent unchanged.
          - Every older ToolResultBlock is replaced, in the returned list,
            by a new ToolResultBlock with the same tool_use_id, the same
            is_error, and content equal to CLEARED_TOOL_RESULT.
          - Everything else is sent exactly as stored: the user prompt,
            every assistant message (text and tool_use blocks, including
            their inputs), the order of the messages and the number of
            messages. Clearing changes the content of some blocks, never
            the shape of the conversation, so every tool_use is still
            matched by a tool_result carrying its id.

        The returned list is a VIEW for this one call. Wherever it differs
        from the stored history it must be built from new message dicts
        and new ToolResultBlocks; self.messages, and every block object
        inside it, must be left untouched -- the stored transcript always
        keeps the full tool output.
        """
        # TODO: implement
        raise NotImplementedError

    def call_llm(self) -> LLMResponse:
        """
        Step 4 (Call LLM): make exactly one call to
        self.llm_client.create_message and return its LLMResponse.

        The call must carry:
          - the system prompt, through the client's `system` parameter --
            never as a message inside the conversation;
          - the messages returned by prepare_context();
          - the API schema of every tool currently registered in
            self.tools (see `to_anthropic_schema` in tools.py), in
            registration order; an agent with no tools sends an empty list.

        Must not modify self.messages.
        """
        # TODO: implement
        raise NotImplementedError

    def process_response(self, response: LLMResponse) -> tuple[list[TextBlock], list[ToolUseBlock]]:
        """
        Step 5 (Process Response): separate response.content into the
        TextBlocks and the ToolUseBlocks it contains. Each of the two lists
        must preserve the blocks' original relative order.

        A response may contain any number of either kind (including zero),
        in any interleaving -- e.g. the model may "think out loud" in text
        and request a tool in the same turn -- and every block must land in
        exactly one of the two lists.

        Return (text_blocks, tool_use_blocks).
        """
        # TODO: implement
        raise NotImplementedError

    def call_tools(self, tool_use_blocks: list[ToolUseBlock]) -> list[ToolResultBlock]:
        """
        Steps 6+7 (Call Tools / Receive Tool Results): execute every
        requested tool call and report the outcome of each one.

        For each ToolUseBlock, the tool to run is the entry of self.tools
        whose key equals block.name, and its arguments are block.input (a
        JSON object mapping the tool's parameter names to values). Each
        ToolUseBlock produces exactly one ToolResultBlock whose tool_use_id
        is that block's id:

          - success: content is the string the tool returned,
            is_error is False;
          - failure -- the name is not registered, or executing the tool
            raises any Exception (illegal values, missing or unexpected
            arguments, ...): content is a short human-readable error
            message, is_error is True.

        A failure must never propagate out of this method: the model, not
        the harness, decides what to do about it (see DESIGN.md Q1).

        Return the ToolResultBlocks in the same order as the input blocks.
        """
        # TODO: implement
        raise NotImplementedError

    def append_assistant_turn(self, response: LLMResponse) -> None:
        """
        Record the model's turn: append ONE assistant-role message to
        self.messages whose content is the model's content list exactly as
        returned -- same block objects, same order, nothing added or
        dropped. The history must be a faithful transcript: the tool_use
        ids stored here are what the following tool results are matched
        against.
        """
        # TODO: implement
        raise NotImplementedError

    def append_tool_results(self, tool_results: list[ToolResultBlock]) -> None:
        """
        Record the tool results: append ONE user-role message to
        self.messages whose content is the list of ToolResultBlocks (see
        README section 2 for why the role is "user").

        Do not append anything when tool_results is empty -- an empty
        message is not a valid turn.
        """
        # TODO: implement
        raise NotImplementedError

    def check_loop(self, response: LLMResponse, iteration: int) -> bool:
        """
        Step 8 (Check Loop): return True if the loop should run another
        iteration, False if it should stop.

        The model is asking for tool results back -- and the loop must
        continue -- only when response.stop_reason is "tool_use". Every
        other stop reason (e.g. "end_turn", "max_tokens") ends the loop.

        `iteration` is the 0-based count of turns completed so far,
        provided in case you want it for logging; the iteration budget
        itself is enforced by run(), not here.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Part 3: Output
    # ------------------------------------------------------------------
    def extract_final_answer(self, text_blocks: list[TextBlock]) -> str:
        """
        Step 9 (End): build the string run() returns from the text blocks
        of the final turn: the `.text` of every block, in order, separated
        by a single newline. Return "" when there are no text blocks.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Orchestration -- the loop itself.
    # ------------------------------------------------------------------
    def run(self, user_prompt: str) -> str:
        """
        Run the complete agent loop for one user prompt and return the
        agent's final answer.

        Build this method ONLY out of the step methods above -- the hidden
        tests substitute individual steps (e.g. prepare_context) and check
        that run() reflects the substitution. Contract:

          - Start a fresh conversation for `user_prompt` (Step 2).
          - Perform at most self.max_iterations iterations. One iteration
            is one LLM call plus whatever tool execution that call
            requests. The LLM must never be called more than
            self.max_iterations times.
          - Every response is recorded in self.messages -- including the
            final one -- and every batch of tool results is recorded before
            the next LLM call, so self.messages is always a complete
            transcript of the conversation.
          - Stop as soon as a response says the model is done (Step 8) and
            return the final answer built from THAT response's text blocks
            (Step 9). Text from earlier turns is reasoning, not the answer.
          - If self.max_iterations iterations complete without the model
            finishing, raise MaxIterationsExceeded.
        """
        # TODO: implement
        raise NotImplementedError
