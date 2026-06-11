"""In-process MCP client used by the orchestration graph's nodes.

Nodes reach the Phase 1.6b tools through this wrapper rather than importing the
tool functions directly. Routing through FastMCP's in-process ``Client`` keeps
the graph honest — it exercises the same tool-call path an external MCP client
(Claude Desktop, Cursor) uses, including schema validation and the
``structured_content`` envelope — without spawning a subprocess or opening a
socket. The MCP server loads the ``@champion`` model once and reuses it, so the
cost here is the tool call, not a model reload.

The wrapper is a lazily-constructed module-level singleton, mirroring the
``_client`` pattern in :mod:`ui.chat_app.chat_agent`.
"""

import logging
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server.server import mcp

logger = logging.getLogger(__name__)


class MCPToolClient:
    """Async wrapper over FastMCP's in-process ``Client``.

    Each method opens a short-lived ``Client(mcp)`` context, calls one tool, and
    returns its structured payload. ``get_model_info`` and
    ``find_similar_transactions`` are intentionally absent — the current chat
    flow does not call them.
    """

    async def lookup_postal_code(self, postal_code: int) -> dict[str, Any] | None:
        """Resolve a postal code to an address, or ``None`` if it is not in the table.

        The MCP tool raises :class:`ToolError` for a postal code absent from the
        lookup table. For the chat flow that is an expected miss — the user may
        have typed an unknown or non-HDB code — so it maps to ``None`` rather
        than surfacing as a failure. Genuine tool faults raise other exception
        types and propagate to the calling node, which records them on the state.
        """
        try:
            async with Client(mcp) as client:
                result = await client.call_tool("lookup_postal_code", {"postal_code": postal_code})
        except ToolError:
            logger.info("Postal code %s not found in lookup table", postal_code)
            return None
        return result.structured_content

    async def predict_price(
        self,
        *,
        town: str,
        flat_type: str,
        floor_area_sqm: float,
        lease_commence_date: int,
        month: str,
    ) -> dict[str, Any]:
        """Predict a single flat's resale price.

        Returns the tool's structured payload: ``predicted_resale_price``,
        ``model_version``, and ``model_alias``.
        """
        async with Client(mcp) as client:
            result = await client.call_tool(
                "predict_price",
                {
                    "town": town,
                    "flat_type": flat_type,
                    "floor_area_sqm": floor_area_sqm,
                    "lease_commence_date": lease_commence_date,
                    "month": month,
                },
            )
        return _require_structured(result.structured_content, "predict_price")

    async def explain_prediction(
        self,
        *,
        town: str,
        flat_type: str,
        floor_area_sqm: float,
        lease_commence_date: int,
        month: str,
    ) -> dict[str, Any]:
        """Explain a single prediction with SHAP feature contributions.

        Returns the tool's structured payload, including ``top_contributors``
        (aggregated, human-readable labels) and ``base_value``.
        """
        async with Client(mcp) as client:
            result = await client.call_tool(
                "explain_prediction",
                {
                    "town": town,
                    "flat_type": flat_type,
                    "floor_area_sqm": floor_area_sqm,
                    "lease_commence_date": lease_commence_date,
                    "month": month,
                },
            )
        return _require_structured(result.structured_content, "explain_prediction")


def _require_structured(payload: dict[str, Any] | None, tool: str) -> dict[str, Any]:
    """Assert a tool returned a structured payload, narrowing the type for callers.

    The HDB tools all return TypedDicts, so ``structured_content`` is always a
    dict in practice. A ``None`` here means the tool contract changed — fail loud
    rather than hand a node a silent ``None``.
    """
    if payload is None:
        raise RuntimeError(f"MCP tool {tool!r} returned no structured content")
    return payload


_client: MCPToolClient | None = None


def get_client() -> MCPToolClient:
    """Return the module-level :class:`MCPToolClient`, constructing it on first use."""
    global _client
    if _client is None:
        _client = MCPToolClient()
    return _client
