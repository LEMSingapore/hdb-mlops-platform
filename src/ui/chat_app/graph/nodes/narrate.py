"""narrate node — turn the final state into user-facing text via Claude Haiku.

The node picks one of three branches off ``state.status`` and makes a single
Anthropic call to phrase the reply. Each branch supplies a different system
prompt and context block; the call itself is shared:

- ``ready_to_predict`` — narrate the predicted price and its top contributors;
- ``needs_follow_up`` — ask one concise question for the missing fields;
- ``out_of_scope`` / ``error`` — decline or apologise without leaking internals.

The Anthropic client is a module-level singleton mirroring the pattern in
:mod:`ui.chat_app.chat_agent`.
"""

import logging
from typing import Any

from anthropic import Anthropic

from ui.chat_app.chat_agent import MODEL
from ui.chat_app.config import ChatConfig
from ui.chat_app.graph.state import GraphState

logger = logging.getLogger(__name__)

_config = ChatConfig()
_client = Anthropic(api_key=_config.anthropic_api_key or None)

# Used only if the LLM call fails, so the user still gets a coherent reply.
_FALLBACK_DECLINE = (
    "I can only estimate Singapore HDB resale flat prices. Please ask about an HDB flat."
)
_FALLBACK_ERROR = (
    "Sorry, something went wrong while estimating that price. "
    "Please try again, or contact support if it persists."
)

# Human-friendly field labels for the follow-up question.
_FIELD_LABELS = {
    "town": "the HDB town",
    "flat_type": "the flat type (e.g. 4 ROOM)",
    "floor_area_sqm": "the floor area in square metres",
    "lease_commence_date": "the year the lease started",
    "month": "the transaction month",
}

_PREDICT_SYSTEM = (
    "You are a friendly assistant that presents a Singapore HDB resale price "
    "estimate. You are given the prediction and the top model contributors as "
    "context — do not invent or recompute any numbers. Reply in British English, "
    "no emojis. Lead with the headline price, then a short 'What's driving this "
    "price' list of the contributors (note whether each pushes the price up or "
    "down), then a one-line caveat that the figure is an estimate. Format prices "
    "with a leading S$ and comma thousands separators, e.g. S$586,888."
)
_FOLLOW_UP_SYSTEM = (
    "You are a friendly assistant collecting the details needed to estimate a "
    "Singapore HDB resale price. The user's message was missing some fields. Ask "
    "one short, natural follow-up question that requests all of the missing "
    "fields together — do not ask several separate questions. Reply in British "
    "English, no emojis."
)
_DECLINE_SYSTEM = (
    "You are a friendly assistant that only estimates Singapore HDB resale flat "
    "prices. Politely decline the user's request in one or two sentences, using "
    "the given reason if helpful, and steer them towards asking about an HDB "
    "resale flat. Reply in British English, no emojis."
)
_ERROR_SYSTEM = (
    "You are a friendly assistant for Singapore HDB resale price estimates. "
    "Something went wrong on our side. Briefly apologise in one or two sentences "
    "and suggest the user try again or contact support. Do not mention error "
    "codes, stack traces, or internal details. Reply in British English, no "
    "emojis."
)


def _predict_context(state: GraphState) -> str:
    price = f"S${state.predicted_price:,.0f}" if state.predicted_price is not None else "unknown"
    lines = [
        f"Predicted resale price: {price}",
        f"Model version: {state.model_version}",
        "Top contributors (signed Singapore-dollar impact on the price):",
    ]
    for contributor in state.top_contributors[:3]:
        lines.append(f"- {contributor['feature']}: {contributor['contribution']:+,.0f}")
    return "\n".join(lines)


def _follow_up_context(state: GraphState) -> str:
    labels = [_FIELD_LABELS.get(field, field) for field in state.missing_fields]
    return "The user still needs to provide: " + ", ".join(labels)


def _decline_context(state: GraphState) -> str:
    if state.parse_reasoning:
        return f"Reason the request is out of scope: {state.parse_reasoning}"
    return "The request is not about a Singapore HDB resale flat."


def _branch(state: GraphState) -> tuple[str, str, str]:
    """Return (system_prompt, context, fallback_text) for the current status."""
    if state.status == "ready_to_predict" and state.predicted_price is not None:
        return _PREDICT_SYSTEM, _predict_context(state), _FALLBACK_ERROR
    if state.status == "needs_follow_up":
        return _FOLLOW_UP_SYSTEM, _follow_up_context(state), _FALLBACK_DECLINE
    if state.status == "out_of_scope":
        return _DECLINE_SYSTEM, _decline_context(state), _FALLBACK_DECLINE
    return _ERROR_SYSTEM, "An internal step failed.", _FALLBACK_ERROR


async def run(state: GraphState) -> dict[str, Any]:
    """Write ``response_text`` by narrating the terminal state with Claude Haiku.

    Falls back to a fixed message for the branch if the LLM call fails, so the
    user always receives a coherent reply rather than an exception.
    """
    system, context, fallback = _branch(state)

    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": context}],
        )
    except Exception as exc:  # the reply must not depend on the LLM being reachable
        logger.warning("narrate call failed: %s: %s", type(exc).__name__, exc)
        return {"response_text": fallback}

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {"response_text": text or fallback}
