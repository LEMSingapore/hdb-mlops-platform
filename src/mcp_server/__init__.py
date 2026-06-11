"""MCP server exposing the HDB platform's capabilities to LLM clients.

The server calls the platform's underlying modules directly — the in-process
model loader, the postal lookup, the MLflow registry, and the SQLite data
layer — rather than HTTP-wrapping the FastAPI service. See
docs/adr/0003-mcp-direct-imports-not-http.md for the rationale.
"""
