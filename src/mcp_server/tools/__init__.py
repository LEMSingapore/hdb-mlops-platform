"""Tool implementations registered with the FastMCP server.

Each tool is a plain typed function returning a TypedDict. The server module
registers them with the FastMCP instance; keeping registration out of these
modules lets them be imported and unit-tested without constructing the server.
"""

from mcp_server.tools.lookup import lookup_postal_code
from mcp_server.tools.model_info import get_model_info
from mcp_server.tools.predict import explain_prediction, predict_price
from mcp_server.tools.similar import find_similar_transactions

__all__ = [
    "explain_prediction",
    "find_similar_transactions",
    "get_model_info",
    "lookup_postal_code",
    "predict_price",
]
