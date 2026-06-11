"""Graph nodes, one module per step.

Each module exposes a single ``run`` coroutine taking the current
:class:`~ui.chat_app.graph.state.GraphState` and returning a dict of the fields
it changed. ``parse`` and ``narrate`` are Day 1 stubs returning canonical data;
``postal_lookup``, ``validate``, ``predict``, and ``explain`` are the real
implementations.
"""
