"""State schema for the chat orchestration graph.

The graph threads a single :class:`GraphState` through every node. Nodes never
mutate the state in place; each returns a dict of changed fields and LangGraph
merges it into a fresh state between steps. ``model_copy(update=...)`` produces
the same merged result outside the graph, which is what the node unit tests use.

Field groups follow the flow: the user message comes in, ``parse`` extracts the
model inputs, ``postal_lookup`` resolves a town from a postal code, ``validate``
sets ``status`` and ``missing_fields``, ``predict`` and ``explain`` fill the
prediction and SHAP outputs, and ``narrate`` writes ``response_text``.
"""

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal[
    "pending",
    "ready_to_predict",
    "needs_follow_up",
    "out_of_scope",
    "error",
]


class GraphState(BaseModel):
    """Single state object threaded through the orchestration graph.

    Optional fields default to ``None`` until the node responsible for them runs,
    so any node can inspect what earlier nodes did or did not produce. List fields
    default to empty via ``default_factory`` to avoid the shared-mutable-default
    trap.
    """

    # Input.
    user_message: str

    # Extracted by the parse node (None until parsed).
    town: str | None = None
    flat_type: str | None = None
    floor_area_sqm: float | None = None
    lease_commence_date: int | None = None
    month: str | None = None
    postal_code: int | None = None

    # Set by the parse node to explain an out_of_scope or error verdict, so the
    # narrate node can decline with a reason rather than a generic message.
    parse_reasoning: str | None = None

    # Output of the postal lookup (None unless a postal code was given and resolved).
    resolved_town: str | None = None

    # Validation result.
    missing_fields: list[str] = Field(default_factory=list)
    status: Status = "pending"

    # Prediction outputs.
    predicted_price: float | None = None
    model_version: int | None = None

    # Explanation outputs.
    top_contributors: list[dict] = Field(default_factory=list)
    base_value: float | None = None

    # Final response to the user.
    response_text: str = ""

    # Error tracking (None unless a node failed).
    error: str | None = None
