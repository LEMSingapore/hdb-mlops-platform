# ADR 0004: LangGraph orchestration replaces the direct Anthropic tool-use loop

**Status:** Accepted
**Date:** 2026-06-12

## Context

The Phase 1.6 chat UI ran on a single Anthropic tool-use loop in `chat_agent.py`: one `messages.create` call with three tools attached, a `while` loop that dispatched whatever tool the model asked for, fed the result back, and repeated until the model stopped calling tools and produced prose. It worked, and for a three-tool flow it was about as small as orchestration gets. But the model owned the control flow. Whether the postal lookup ran, whether a prediction happened before an explanation, whether all five fields were actually present before predicting — all of that lived inside the model's reasoning, visible only as a sequence of tool calls after the fact. When something went wrong, the failure was a turn in a transcript, not a step I could point at.

Phase 1.6b had already moved the actual capabilities — predict, explain, postal lookup — behind the MCP server, so the tool layer was settled. What was left was the orchestration: the deterministic shape of a turn. I wanted that shape to be code I could read, test branch by branch, and attach per-step error handling to, rather than an emergent property of an LLM's tool-calling. That is what Phase 1.6c set out to build.

## Decision

The chat turn is orchestrated by an explicit LangGraph state machine over a single `GraphState`, with the LLM demoted from controller to two bounded steps inside it. The graph is `parse -> postal_lookup -> validate -> (predict -> explain) -> narrate`, with a conditional edge after `validate` routing everything that is not `ready_to_predict` — missing fields, out of scope, or an upstream error — straight to `narrate`. Each node owns one step, returns a dict of the fields it changed, and carries its own error handling; a tool fault sets `status = "error"` on the state and the graph routes around the prediction nodes rather than raising.

The two LLM-backed nodes call the Anthropic SDK directly — `parse` makes one Haiku call that returns strict JSON for slot-filling or an out-of-scope verdict, and `narrate` makes one Haiku call to phrase the terminal state. They do **not** go through `langchain_anthropic`. The orchestration layer (LangGraph) and the tool layer (the Phase 1.6b MCP server, called in-process) are kept as separate concerns: the graph decides *what happens when*, the MCP tools *do the work*, and neither leaks into the other.

The graph is invoked stateless per turn. Each user message constructs a fresh `GraphState(user_message=...)` and runs it to completion; the conversation history lives in Streamlit's session state for display only, and the `parse` node sees just the latest message.

## Consequences

The control flow is now a diagram I can point at, and every branch off it is a test. `test_graph_branches.py` drives the five conditional paths — missing fields, out of scope, a predict-tool fault, a postal lookup miss, and a postal lookup that resolves the town and carries the prediction — by stubbing the LLM nodes with deterministic doubles and handing the MCP nodes fake clients, so the assertions turn on routing, not on phrasing or live services. The old loop could only be tested by mocking the model's tool-call sequence, which tested the mock as much as the code. Error handling is localised: a failure in the postal lookup or the prediction tool is caught in that node, recorded on the state, and narrated gracefully, instead of surfacing as an exception or a confused model turn.

Calling the Anthropic SDK directly inside the nodes, rather than through `langchain_anthropic`, is a deliberate trade-off. It keeps one Anthropic client pattern across the whole codebase — the same `Anthropic(api_key=...)` singleton and `cache_control` configuration the legacy agent used — and avoids pulling LangChain's message-and-model abstractions in just to make two calls. The cost is that the nodes are not portable to a different LLM provider without editing them; for a single-provider portfolio project that is the right side of the trade. LangGraph earns its place here as the orchestration graph; LangChain's model wrappers would have been weight without benefit.

The stateless-per-turn invocation is the main limitation I am accepting. A follow-up like "what about a 5-room instead" starts a fresh graph run that re-parses only the new message, so the chat cannot yet refine a previous answer conversationally. Multi-turn awareness — threading prior turns into the parse node, MessagesPlaceholder-style — is a deliberate deferral, not an oversight; the per-turn graph is simpler to reason about and test, and the UI still shows the full conversation. Adding turn memory later is a change to one node's input, not to the graph's shape.

One implementation quirk is worth recording because it surprised me and will surprise the next reader. `graph.ainvoke(state)` does not return a `GraphState` — it returns a plain `dict` of the merged channel values. Reading a field off the result therefore needs `GraphState(**result)` to reconstruct the typed object first. The Streamlit integration and the tests both do this, and the `GraphState` docstring notes it. It is a LangGraph convention (nodes return dict updates, the runtime merges them into channels), not a bug, but it reads as one until you know.

The legacy `chat_agent.py` stays in the tree for now — Streamlit simply stops calling it. Removing it is a separate cleanup once the graph has proven itself in use, kept out of this change so the orchestration swap is reviewable on its own.

## Alternatives considered

**Keep the direct Anthropic tool-use loop.** It was already working and is genuinely compact for three tools. Rejected because the model owned the control flow: validation, ordering, and the predict-then-explain sequence were all emergent from the model's tool-calling rather than explicit, testable code. The branch coverage and per-step error handling I wanted are exactly what an explicit graph gives and an LLM-driven loop cannot.

**Use `langchain_anthropic` with LangGraph's prebuilt agent helpers.** This is the idiomatic LangGraph-plus-LangChain path and would have handed me a ReAct-style agent almost for free. Rejected because it reintroduces the same model-owns-the-flow problem the prebuilt agent is built around, and because it pulls LangChain's model and message abstractions across the whole chat path to wrap two Haiku calls I can make directly. I wanted LangGraph for its graph, not LangChain for its agent.

**A pipeline orchestrator wrapping the work (ZenML, Prefect, or similar).** Heavier machinery aimed at scheduled, multi-step data and training pipelines, not a sub-second interactive chat turn. It is also on the project's explicit out-of-scope list. Rejected as a category error: the problem is request-time orchestration of a handful of in-process steps, which a compiled state graph models exactly, with none of the scheduler, persistence, or DAG-runner weight those tools carry.
