"""parse node — extract model inputs from the user's message.

Day 1 stub: returns a fixed extraction for the canonical test input ("3 room
flat in Tampines, postal 528003") regardless of the actual message, so the rest
of the graph has deterministic, fully-populated inputs to run against. Day 2
replaces the body with a real Anthropic call that does slot-filling from free
text.
"""

from ui.chat_app.graph.state import GraphState


async def run(state: GraphState) -> dict:
    """Return the canonical extracted fields (stub).

    The values match the project's standing test input: a Tampines 4-room flat
    with a postal code. Postal 528003 is deliberately one that is absent from the
    lookup table, so ``postal_lookup`` exercises its miss path while ``town``
    being present keeps the happy path intact.
    """
    return {
        "town": "TAMPINES",
        "flat_type": "4 ROOM",
        "floor_area_sqm": 95.0,
        "lease_commence_date": 1985,
        "month": "2024-06",
        "postal_code": 528003,
    }
