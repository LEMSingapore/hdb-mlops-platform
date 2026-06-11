"""LangGraph orchestration for the HDB chat app.

The graph replaces the chat agent's tool-use loop with an explicit state machine:
``parse -> postal_lookup -> validate -> (predict -> explain) -> narrate``. Each
node owns one step with its own error handling, and the tool layer underneath is
the Phase 1.6b MCP server, invoked in-process via :mod:`ui.chat_app.graph.mcp_client`.
"""

from ui.chat_app.graph.graph_builder import build_graph
from ui.chat_app.graph.state import GraphState

__all__ = ["GraphState", "build_graph"]
