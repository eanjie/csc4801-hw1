# HW1: Build a Coding Agent From Scratch

**Course:** CSC4801 AI-assisted Software Engineering

**Builds on:** Lectures 01 (LLMs), 02 (Principles of Agents), 03(Advanced Agents)

## 1. Overview

Lecture 02 described the agent loop as a numbered pipeline:

```
Part 1: Input Processing                       |  Part 2: Main Loop                                 |  Part 3: Output
----------------------------------------------------------------------------------------------------------------------
0 Start -> 1 Receive Prompt -> 2 Initialize -> |  [3 Prepare Context -> 4 Call LLM                  |  ->  9 End
                                               |   5 Process Response -> 6 Call Tools ->            |
                                               |   7 Receive Tool Results -> 8 Check Loop] (repeat) |
```

In this assignment you will implement exactly this loop, from scratch, with
no agent framework (no LangChain, no AutoGPT, nothing but the Python
standard library). By the end you will have a small but fully working
ReAct-style tool-using agent.

You are **not** calling a real LLM in this assignment (that is non-deterministic, which makes it impossible to grade fairly). Instead
you drive your agent with `FakeLLMClient`, a scripted stand-in defined in
`agent/llm_client.py`, that returns pre-programmed responses. Every public
and hidden test constructs a `FakeLLMClient` with a script and checks that
your agent behaves correctly against it. The exact same `Agent` code would
work unmodified against a real model -- see `make_anthropic_llm_client` at
the bottom of `agent/llm_client.py` if you want to try that yourself (fully
optional, ungraded).

## 2. What you must implement

All your work is in two files:

| File             | What's TODO                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent/core.py`  | The `Agent` class: the 10 step methods (`receive_prompt`, `initialize`, `prepare_context`, `call_llm`, `process_response`, `call_tools`, `append_assistant_turn`, `append_tool_results`, `check_loop`, `extract_final_answer`) **and `run()`, the loop that ties them together**. Each docstring states the method's contract -- what it receives, what it must return, which invariants it must keep. The docstrings do not tell you how to write the code; that is the assignment. |
| `agent/tools.py` | One tool, `WordCountTool.execute`, is left for you to implement as a warm-up in tool authoring.                                                                                                                                                                                                                                                                                                                                                                                      |

Do **not** modify `agent/llm_client.py` -- the autograder depends on its
exact behavior, and it is identical across every assignment in this course.

### Key data types (`agent/llm_client.py`)

- `TextBlock(text)` -- the model "talking" (reasoning or a final answer).
- `ToolUseBlock(id, name, input)` -- the model requesting you run a tool.
- `ToolResultBlock(tool_use_id, content, is_error)` -- your report of what
  happened when you ran a tool.
- `LLMResponse(content, stop_reason)` -- what `create_message()` returns.
  `stop_reason` is `"tool_use"` (the model wants tool results before
  continuing), `"end_turn"` (the model is done), or `"max_tokens"` (the
  response was cut off by the output limit).

### The Anthropic Messages API message shape

Messages are dicts: `{"role": "user"|"assistant", "content": [...]}`,
where `content` is a list of blocks. The three kinds of block you will
put there:

- the user's prompt is a text block in dict form,
  `{"type": "text", "text": "..."}`;
- the model's turn is recorded with the exact block objects the model
  returned (`TextBlock` / `ToolUseBlock`);
- tool results are sent back as a **user**-role message (not assistant,
  not system) whose content is the list of `ToolResultBlock`s.

The system prompt is not a message at all -- it travels through the
separate `system` parameter of `create_message`.

### Context management in `prepare_context`

`self.messages` grows every iteration and is re-sent to the LLM on every
call. Lecture 03 covered several strategies a harness uses to keep that in
check; this assignment implements one of them, the **Tool Count Budget**:

- `Agent` has a field `keep_recent_tool_results` (default 5).
- `prepare_context()` builds the list of messages for the current call.
  The newest `keep_recent_tool_results` `ToolResultBlock`s in the
  conversation are sent unchanged; every older one is replaced by a new
  `ToolResultBlock` with the same `tool_use_id` and `is_error` and with
  `content` equal to the constant `CLEARED_TOOL_RESULT`
  (`"[Old tool result content cleared]"`, defined in `core.py`).
- Nothing else changes: the user prompt, the assistant turns (including
  `tool_use` inputs), the message order and the message count are exactly
  as stored, so every `tool_use` is still matched by a `tool_result`.
- The returned list is a view for this one call. `self.messages` -- the
  stored transcript -- keeps every tool output in full.

Public test 5 shows the rule in action.

## 3. How to run the tests

```bash
cd hw1_agent_core
pip install -r requirements.txt
pytest tests/public -v
```

We give you **5 public tests** (`tests/public/test_01..05_*.py`) covering:

1. a single tool call followed by a final answer,
2. multiple sequential tool calls with different tools,
3. loop termination (`MaxIterationsExceeded`) when the model never stops,
4. tool errors (bad input, unknown tool name) being reported back to the
   model instead of crashing your agent,
5. context management: old tool results are cleared from the view sent
   to the LLM while the stored history keeps them (`prepare_context`).

At grading time we run these 5 **plus 10 hidden tests** against your
`agent/` package, unmodified. The hidden tests use the same machinery as
the public ones -- a scripted `FakeLLMClient`, calls to `run()`, direct
calls to the individual step methods, and inspection of `llm.calls` and
`agent.messages` -- but they cover contract details and edge cases that
the public tests do not touch. Passing all 5 public tests is necessary
but not sufficient: read every docstring in `core.py` carefully, since
the hidden tests check the exact contract described there, not just
"does `run()` return the right string." Reading the public tests to see
how a test drives the agent is a good idea; writing tests of your own for
cases the public tests leave out is a better one.

## 4. Grading (100 points)

| Component                                     | Points |
| --------------------------------------------- | ------ |
| 10 hidden tests (7 pts each)                  | 70     |
| All 5 public tests passing (2 pts each)       | 10     |
| Short design writeup (`DESIGN.md`, see below) | 20     |

### `DESIGN.md` (create this file yourself, 2 pages max)

Answer briefly:

1. Why must `call_tools` catch exceptions instead of letting them propagate?
   What would happen to the conversation if it didn't?
2. Why are tool results sent as a `"user"`-role message rather than a new
   message type? What does this imply about how the model "sees" tool
   output?
3. A single assistant turn may contain several `ToolUseBlock`s. Your
   `call_tools` runs them one after another, in order. Give one concrete
   case where running them concurrently would be safe, one case where it
   would not, and what information the harness would need in order to
   tell the two apart.
4. Your `prepare_context` clears old tool results, but `self.messages`
   still grows every iteration and is still re-sent in full on every
   call. Name at least two things the count budget does NOT fix (things
   that keep growing, or that can still exceed the context window on a
   single call). Then pick one other strategy from lecture 03 you would
   add next: say which step of the loop it belongs to, how it works, and
   what it needs that the current harness does not have.

## 5. Constraints

- No third-party agent frameworks. Standard library + `pytest` only.
- Do not edit `agent/llm_client.py`, `tests/public/*`, or the signature/name
  of any method already defined in `core.py`/`tools.py` -- the hidden tests
  call these methods directly by name.
- `run()` must be built out of the step methods (the hidden tests replace
  individual step methods and check that `run()` uses the replacement); do
  not re-implement a step's logic inline in `run()`.
- Do not add global mutable state; all state must live on `Agent` instances
  (the hidden tests instantiate multiple agents in the same process).

## 6. Use of AI assistants

This course is about AI-assisted software engineering, and that includes
your own workflow: you MAY use an AI assistant (MiMo, ChatGPT, Claude,
Copilot, ...) for any part of this assignment, including `core.py` and
`tools.py`. One condition applies: **the writeup is yours.** `DESIGN.md`
must be written in your own words. Discussing with an assistant to check
your understanding is fine; pasted model output in the writeup receives
no credit.

You are responsible for every line you submit: "the model wrote it" is
not a defense for a wrong answer or for code you cannot explain. Fair
warning: HW2 builds directly on this loop, and the DESIGN.md questions
are graded on understanding -- outsourcing the code without
understanding it will cost you later in the course.

Sharing code between students remains prohibited. You may
discuss the _design_ of the agent loop with classmates.
