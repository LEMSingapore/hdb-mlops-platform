# ADR 0003: MCP server calls platform modules directly, not over HTTP

**Status:** Accepted
**Date:** 2026-06-11

## Context

Phase 1.6b exposes the platform's capabilities — prediction, SHAP explanation, postal lookup, model provenance, comparable-transaction search — as Model Context Protocol tools so any MCP client (Claude Desktop, Cursor, the Phase 1.6c LangGraph agent) can call them without bespoke per-client glue. The platform already runs a FastAPI service that serves `/predict` and `/explain` over HTTP. That raised an obvious question: should the MCP server be a thin HTTP proxy that forwards each tool call to FastAPI, or should it import and call the platform's Python modules directly?

The HTTP-proxy route is the conventional "wrap the existing API" answer. It keeps a single code path to the model and a single place where inference happens. But it forces every MCP tool call through a localhost round-trip, makes the MCP server useless unless FastAPI is already running, and reloads or re-implements the SHAP and SQLite logic that only exists below the HTTP boundary anyway — `find_similar_transactions` and `get_model_info` have no FastAPI endpoints to proxy to.

## Decision

The MCP server calls the platform's underlying modules directly — `ModelLoader`, `lookup.postal`, the MLflow registry client, and the `data.queries` SQLite layer — in a single process. It does not HTTP-wrap FastAPI.

`predict_price` and `explain_prediction` share a lazily-initialised `ModelLoader` singleton scoped to the MCP server process: the `@champion` model loads on first tool invocation and stays resident in memory. FastAPI keeps running as a separate process for HTTP clients (the form app, direct `curl`). The two serve different consumer types from the same registry — they can run side by side or independently.

## Consequences

The MCP server has no localhost dependency: it works whether or not FastAPI is up, which matters for the stdio-transport Claude Desktop integration where there is no expectation of a running web service. Tool calls skip serialisation and the network entirely — a prediction is an in-memory `model.predict()` call, not a JSON round-trip. The five tools draw on whichever layer fits: the model loader for inference, MLflow for provenance, SQLite for comparables. There is no duplicate inference path to keep in sync, because inference lives in one place — the loaded sklearn pipeline — and both FastAPI and MCP call into it.

One consequence specific to `explain_prediction` is worth recording, because it shaped the tool's contract. The model one-hot encodes `town` and `flat_type`, so SHAP returns a contribution for every dummy — including the dummies that are 0 for the row being explained. Ranking the raw per-dummy contributions by absolute value surfaces inactive categories: for a 4-room flat the top contributors can include a "3-room" entry, whose SHAP value is the counterfactual "if this flat were a 3-room" effect, not what being a 4-room contributes. That reads as a bug to a human and an LLM client flagged it during smoke-testing. So `explain_prediction` aggregates each column's dummies into a single signed sum attributed to the row's active category before selecting the top contributors — the human-facing `top_contributors` view shows one entry per categorical. The raw per-dummy decomposition stays in `feature_contributions` for programmatic consumers that want the full SHAP breakdown. FastAPI's `/explain` is deliberately left alone: it surfaces the per-dummy view to its HTTP consumers, and the two interfaces serving different contracts is consistent with the side-by-side design above.

The trade-off I am accepting is tight coupling to the platform's in-process Python. The MCP server must run in the same environment as the rest of the codebase, with the same dependencies, and it loads its own copy of the model rather than sharing FastAPI's resident instance — so running both means two models in memory. For a single-node portfolio deployment that is a non-issue; at multi-replica scale it would argue for a shared model server, which is a Phase 6 concern, not this one. Streamable HTTP transport, which would let the MCP server run remotely, is deferred to Phase 6 — stdio covers the local Claude Desktop case this phase targets.

## Alternatives considered

**HTTP-proxy the FastAPI service.** Each MCP tool would forward to a FastAPI endpoint. Rejected because it adds a localhost dependency and network round-trip for no benefit when both processes share a machine, and because two of the five tools (`find_similar_transactions`, `get_model_info`) have no endpoint to proxy — I would have had to build the HTTP surface first, then wrap it, to expose logic the MCP server can already reach directly.

**A shared model server both FastAPI and MCP call.** Factor model loading into a standalone process that both the HTTP service and the MCP server query. This is the right answer at scale — it removes the double-load — but it introduces a third process, an IPC boundary, and lifecycle coordination that a solo single-node deployment does not need. It belongs in the Phase 6 deployment story alongside the MLflow SPOF discussion, not in the local-dev tool surface.
