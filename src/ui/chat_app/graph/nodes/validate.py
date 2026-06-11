"""validate node — check the five model fields are present.

Pure Python, no tool or LLM call. A town from either ``town`` or
``resolved_town`` satisfies the town requirement. The node sets ``status`` to
``ready_to_predict`` when all five fields are present and ``needs_follow_up``
otherwise, listing the absent fields in ``missing_fields`` so the narrate node
can ask for them.
"""

from ui.chat_app.graph.state import GraphState

REQUIRED_FIELDS = ("town", "flat_type", "floor_area_sqm", "lease_commence_date", "month")


async def run(state: GraphState) -> dict:
    """Set ``status`` and ``missing_fields`` from which model inputs are present.

    Only runs when ``status == "pending"``. A terminal status set upstream — an
    ``error`` from a tool fault or an ``out_of_scope`` verdict from the parse
    node — passes through unchanged so a validation verdict cannot mask it.
    """
    if state.status != "pending":
        return {}

    present = {
        "town": state.town or state.resolved_town,
        "flat_type": state.flat_type,
        "floor_area_sqm": state.floor_area_sqm,
        "lease_commence_date": state.lease_commence_date,
        "month": state.month,
    }
    missing = [field for field in REQUIRED_FIELDS if present[field] is None]

    if missing:
        return {"missing_fields": missing, "status": "needs_follow_up"}
    return {"missing_fields": [], "status": "ready_to_predict"}
