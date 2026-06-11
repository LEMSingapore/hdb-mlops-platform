"""FastMCP server instance and tool registration.

Exposes five tools over the Model Context Protocol so any MCP client (Claude
Desktop, Cursor, a future LangGraph agent) can use the same prediction and
explanation surface the Streamlit chat app uses, with no per-client code.
"""

from fastmcp import FastMCP

from mcp_server.tools import (
    explain_prediction,
    find_similar_transactions,
    get_model_info,
    lookup_postal_code,
    predict_price,
)

mcp: FastMCP = FastMCP(
    name="hdb-mlops",
    instructions=(
        "HDB resale price prediction tools. Use lookup_postal_code first when "
        "given a postal code; then call predict_price or explain_prediction with "
        "the resolved town. find_similar_transactions surfaces comparable "
        "historical sales. get_model_info reports the serving model's version "
        "and metrics."
    ),
)

mcp.tool(predict_price)
mcp.tool(explain_prediction)
mcp.tool(lookup_postal_code)
mcp.tool(get_model_info)
mcp.tool(find_similar_transactions)
