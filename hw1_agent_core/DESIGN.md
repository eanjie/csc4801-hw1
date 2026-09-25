# HW1 Design Writeup

## 1. Why must `call_tools` catch exceptions instead of letting them propagate?

If a tool raises an exception and it escapes `call_tools`, the entire agent loop crashes and the conversation stops abruptly. The model never receives a structured report of what went wrong, so it cannot recover, retry with corrected arguments, or explain the failure to the user.

By catching exceptions and returning a `ToolResultBlock` with `is_error=True`, the harness keeps the conversation alive. The error becomes just another piece of context the model reads on the next turn, exactly like a successful tool result. The transcript stays complete: every `tool_use` still gets a matching `tool_result`, preserving the message shape the API expects.

## 2. Why are tool results sent as a `"user"`-role message rather than a new message type?

The Anthropic Messages API only defines two conversational roles in the message list: `"user"` and `"assistant"`. There is no separate `"tool"` role in this API shape. Tool results are therefore injected as a user turn — the harness is effectively saying "here is the information you asked for."

From the model's perspective, tool output looks like user-provided context appended after its own assistant turn. That mirrors the turn-taking pattern the model was trained on: the assistant speaks, then the user replies with new information, then the assistant continues. Using a dedicated role would require API support the harness does not have; piggybacking on `"user"` keeps the wire format simple and compatible with real LLM clients.

## 3. Concurrent vs. sequential tool execution

**Safe to run concurrently:** Two independent read-only lookups in the same turn, e.g. `word_count(text="hello world")` and `uppercase(text="hello")`. Neither tool reads or writes shared mutable state, and neither's output is an input to the other. The harness could run them in parallel and still assemble results in the original request order.

**Unsafe to run concurrently:** A write followed by a read on the same resource, e.g. a hypothetical `write_file(path="/tmp/x", content="A")` immediately followed by `read_file(path="/tmp/x")`. If executed in parallel, the read might complete before the write finishes and return stale or empty content.

**What the harness would need to tell them apart:** Tool metadata declaring side effects (read-only vs. write), resource identifiers each call touches (file paths, database keys), and dependency edges when one call's arguments reference another's expected output. Without that, the safe default is sequential execution in request order, which is what this assignment implements.

## 4. What the count budget does NOT fix, and a strategy to add next

The tool-count budget only replaces the *content* of old tool results in the view sent to the LLM. It does not fix:

1. **Growing assistant turns** — every model response (including reasoning text and full `tool_use` inputs) is still sent verbatim on every subsequent call, so long conversations with many turns keep adding tokens.
2. **The user prompt and system prompt** — these are resent in full every iteration and never shrink.
3. **Message count** — cleared results still occupy a slot in the conversation; only their text is shortened, so a very long transcript with many tool rounds still has structural overhead.

**Strategy to add next: summarization / compaction** (lecture 03). This belongs in **Step 3 (Prepare Context)**, before `call_llm`. When the estimated token count of `prepare_context()`'s output exceeds a threshold, the harness would call a separate summarization pass (or apply a deterministic rule) to replace a block of older messages with a single assistant or user summary message. It needs a token estimator, a compaction policy (what to summarize and what to keep verbatim), and optionally a second LLM call dedicated to summarization — none of which the current harness provides.
