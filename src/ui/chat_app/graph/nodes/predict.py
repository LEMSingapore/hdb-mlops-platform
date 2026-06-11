"""predict node — call the MCP prediction tool.

Runs only when validation marked the state ``ready_to_predict``. It resolves the
town (explicit over looked-up), calls the MCP ``predict_price`` tool, and records
the price and the serving model's version. A tool fault sets ``status = "error"``
so the graph routes straight to narrate.
"""

import logging

from ui.chat_app.graph.mcp_client import get_client
from ui.chat_app.graph.nodes._features import model_fields
from ui.chat_app.graph.state import GraphState

logger = logging.getLogger(__name__)


async def run(state: GraphState) -> dict:
    """Predict the resale price and record it with the model version.

    No-ops unless ``status == "ready_to_predict"``. Records an error update if
    the prediction tool call fails.
    """
    if state.status != "ready_to_predict":
        return {}

    client = get_client()
    try:
        result = await client.predict_price(**model_fields(state))
    except Exception as exc:  # any tool fault becomes graph-visible state
        logger.warning("predict failed: %s: %s", type(exc).__name__, exc)
        return {
            "error": f"Prediction failed: {type(exc).__name__}: {exc}",
            "status": "error",
        }

    return {
        "predicted_price": result["predicted_resale_price"],
        "model_version": result["model_version"],
    }
