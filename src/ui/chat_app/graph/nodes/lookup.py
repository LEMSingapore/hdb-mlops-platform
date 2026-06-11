"""postal_lookup node — resolve a postal code to a town via the MCP tool.

When the parse node extracted a postal code, this node calls the MCP
``lookup_postal_code`` tool and records the resolved town on the state. A postal
code absent from the lookup table is not an error — the wrapper returns ``None``
and the node passes the state through unchanged, leaving any town the user gave
intact. Only a genuine tool fault sets ``status = "error"``.
"""

import logging

from ui.chat_app.graph.mcp_client import get_client
from ui.chat_app.graph.state import GraphState

logger = logging.getLogger(__name__)


async def run(state: GraphState) -> dict:
    """Resolve ``state.postal_code`` to ``resolved_town`` when possible.

    No-ops (returns an empty update) when there is no postal code, when the code
    is not in the lookup table, or when the resolved address has no HDB town.
    Records an error update if the tool call itself fails.
    """
    if state.postal_code is None:
        return {}

    client = get_client()
    try:
        result = await client.lookup_postal_code(state.postal_code)
    except Exception as exc:  # any tool fault becomes graph-visible state
        logger.warning("postal_lookup failed: %s: %s", type(exc).__name__, exc)
        return {
            "error": f"Postal lookup failed: {type(exc).__name__}: {exc}",
            "status": "error",
        }

    if result is None or result.get("town") is None:
        return {}
    return {"resolved_town": result["town"]}
