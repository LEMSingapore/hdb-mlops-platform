"""Graph nodes, one module per step.

Each module exposes a single ``run`` coroutine taking the current
:class:`~ui.chat_app.graph.state.GraphState` and returning a dict of the fields
it changed. ``parse`` and ``narrate`` make Anthropic Claude Haiku calls;
``postal_lookup``, ``predict``, and ``explain`` call the MCP tools; ``validate``
is pure Python.
"""
