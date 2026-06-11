"""Entry point: ``python -m mcp_server`` runs the server over stdio transport."""

from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
