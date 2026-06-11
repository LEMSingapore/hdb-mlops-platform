"""Shared field resolution for the prediction and explanation nodes.

Both nodes need the same five model inputs, and both prefer a town the parse
node extracted directly over one the postal lookup resolved. Centralising that
here keeps the two nodes in lockstep — they can never disagree on which town to
predict for.
"""

from typing import Any

from ui.chat_app.graph.state import GraphState


def model_fields(state: GraphState) -> dict[str, Any]:
    """Return the five model-input fields for an MCP prediction or explanation call.

    Prefers ``state.town`` over ``state.resolved_town`` so an explicit town the
    user gave wins over one inferred from a postal code. Call only after
    validation has set ``status == "ready_to_predict"``, which guarantees every
    value here is present.
    """
    return {
        "town": state.town or state.resolved_town,
        "flat_type": state.flat_type,
        "floor_area_sqm": state.floor_area_sqm,
        "lease_commence_date": state.lease_commence_date,
        "month": state.month,
    }
