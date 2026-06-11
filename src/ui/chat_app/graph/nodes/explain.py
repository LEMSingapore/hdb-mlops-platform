"""explain node — call the MCP SHAP explanation tool.

Runs only after a successful prediction. It calls the MCP ``explain_prediction``
tool and records the top contributors (already aggregated to one entry per
source feature with human-readable labels) and the model's base value. A tool
fault sets ``status = "error"``.
"""

import logging

from ui.chat_app.graph.mcp_client import get_client
from ui.chat_app.graph.nodes._features import model_fields
from ui.chat_app.graph.state import GraphState

logger = logging.getLogger(__name__)


async def run(state: GraphState) -> dict:
    """Record SHAP top contributors and base value for the prediction.

    No-ops unless ``status == "ready_to_predict"`` and a price has been set.
    Records an error update if the explanation tool call fails.
    """
    if state.status != "ready_to_predict" or state.predicted_price is None:
        return {}

    client = get_client()
    try:
        result = await client.explain_prediction(**model_fields(state))
    except Exception as exc:  # any tool fault becomes graph-visible state
        logger.warning("explain failed: %s: %s", type(exc).__name__, exc)
        return {
            "error": f"Explanation failed: {type(exc).__name__}: {exc}",
            "status": "error",
        }

    return {
        "top_contributors": result["top_contributors"],
        "base_value": result["base_value"],
    }
