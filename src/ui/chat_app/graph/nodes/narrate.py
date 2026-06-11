"""narrate node — turn the final state into user-facing text.

Day 1 stub: deterministic string formatting per terminal status. Day 2 replaces
the ``ready_to_predict`` branch with a real Anthropic narration call that frames
the price and contributors conversationally; the other branches stay simple.
"""

from ui.chat_app.graph.state import GraphState

_DEFAULT_DECLINE = (
    "I can only estimate Singapore HDB resale flat prices. Please ask about an HDB flat."
)


async def run(state: GraphState) -> dict:
    """Write ``response_text`` from the terminal status (stub).

    - ``ready_to_predict`` with a price: state the prediction, model version, and
      top three contributors.
    - ``needs_follow_up``: ask for the missing fields.
    - ``out_of_scope`` or ``error``: return the recorded error or a decline message.
    """
    if state.status == "ready_to_predict" and state.predicted_price is not None:
        top_three = ", ".join(c["feature"] for c in state.top_contributors[:3])
        text = (
            f"Predicted price: S${state.predicted_price:.0f}, "
            f"model v{state.model_version}, "
            f"top contributors: {top_three}"
        )
        return {"response_text": text}

    if state.status == "needs_follow_up":
        return {"response_text": f"Please provide: {', '.join(state.missing_fields)}"}

    return {"response_text": state.error or _DEFAULT_DECLINE}
