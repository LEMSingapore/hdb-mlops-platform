"""Assemble the chat orchestration graph.

Wires the six nodes into a :class:`~langgraph.graph.StateGraph` over
:class:`~ui.chat_app.graph.state.GraphState`:

    parse -> postal_lookup -> validate --(ready)--> predict -> explain -> narrate -> END
                                       \\--(else)----------------------> narrate -> END

After ``validate``, a conditional edge routes ``ready_to_predict`` states through
prediction and explanation, and every other terminal status
(``needs_follow_up``, ``out_of_scope``, ``error``) straight to ``narrate``.
Checkpointing is left at LangGraph's in-memory default — no persistence yet.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ui.chat_app.graph.nodes import explain, lookup, narrate, parse, predict, validate
from ui.chat_app.graph.state import GraphState


def _after_validate(state: GraphState) -> str:
    """Route to prediction when ready, otherwise to narration."""
    if state.status == "ready_to_predict":
        return "predict"
    return "narrate"


def build_graph() -> CompiledStateGraph:
    """Build and compile the chat orchestration graph."""
    graph = StateGraph(GraphState)
    graph.add_node("parse", parse.run)
    graph.add_node("postal_lookup", lookup.run)
    graph.add_node("validate", validate.run)
    graph.add_node("predict", predict.run)
    graph.add_node("explain", explain.run)
    graph.add_node("narrate", narrate.run)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "postal_lookup")
    graph.add_edge("postal_lookup", "validate")
    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"predict": "predict", "narrate": "narrate"},
    )
    graph.add_edge("predict", "explain")
    graph.add_edge("explain", "narrate")
    graph.add_edge("narrate", END)

    return graph.compile()
