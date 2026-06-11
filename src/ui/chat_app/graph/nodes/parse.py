"""parse node — extract model inputs from the user's message via Claude Haiku.

A single Anthropic call does the slot-filling. The model returns a strict JSON
object: a ``status`` (``extracted`` or ``out_of_scope``), a brief ``reasoning``
string, and the six fields the downstream graph needs. Non-HDB property questions
(condos, landed, commercial) come back ``out_of_scope`` so the graph declines
without touching the model. A response that does not parse as the expected JSON
sets ``status = "error"`` rather than guessing at fields.

The Anthropic client is a module-level singleton mirroring the pattern in
:mod:`ui.chat_app.chat_agent`, from which the model id and the town/flat-type
vocabularies are imported to keep a single source of truth.
"""

import json
import logging
from datetime import date
from typing import Any

from anthropic import Anthropic

from ui.chat_app.chat_agent import FLAT_TYPES, MODEL, TOWNS
from ui.chat_app.config import ChatConfig
from ui.chat_app.graph.state import GraphState

logger = logging.getLogger(__name__)

_config = ChatConfig()
# Pass None when the field is empty so the SDK reads ANTHROPIC_API_KEY from the
# environment rather than treating an empty string as a set (but invalid) key.
_client = Anthropic(api_key=_config.anthropic_api_key or None)

_SYSTEM_PROMPT_TEMPLATE = (
    "You extract structured fields from a user's plain-English question about "
    "Singapore HDB resale flat prices. You never answer the question or estimate "
    "a price yourself — you only fill a field set.\n\n"
    "Today's date is {today}. The current month is {current_month}.\n\n"
    "Return a single JSON object and nothing else, with exactly these keys:\n"
    '  "status": "extracted" or "out_of_scope"\n'
    '  "reasoning": a brief reason for the status\n'
    '  "fields": an object with these keys, each null if the user did not give it:\n'
    '    "town": one of the HDB towns below, uppercased, or null\n'
    '    "flat_type": one of the flat types below, uppercased, or null\n'
    '    "floor_area_sqm": a number in square metres, or null\n'
    '    "lease_commence_date": the four-digit year the lease began, or null\n'
    '    "month": the transaction month as "YYYY-MM", or null\n'
    '    "postal_code": a six-digit Singapore postal code as an integer, or null\n\n'
    "Rules:\n"
    "- Set status to out_of_scope for any question about non-HDB property — "
    "condominiums, landed houses, commercial units, or property outside "
    "Singapore. Leave all fields null in that case.\n"
    "- Leave a field null if the user did not provide it. Do not invent values.\n"
    "- The one exception is month: if the user gives no transaction month, set it "
    "to the current month ({current_month}).\n"
    "- Normalise town and flat_type to the exact uppercase vocabulary values "
    'below (for example "tampines" becomes "TAMPINES", "4-room" becomes '
    '"4 ROOM").\n'
    "- A postal code helps resolve the town later; it is not itself a model "
    "input, so record it in postal_code and still leave town null unless the "
    "user named the town.\n\n"
    "Valid towns: {towns}\n"
    "Valid flat types: {flat_types}"
)

_EXPECTED_FIELD_KEYS = {
    "town",
    "flat_type",
    "floor_area_sqm",
    "lease_commence_date",
    "month",
    "postal_code",
}


def _system_prompt() -> str:
    today = date.today()
    return _SYSTEM_PROMPT_TEMPLATE.format(
        today=today.strftime("%d %B %Y"),
        current_month=today.strftime("%Y-%m"),
        towns=", ".join(TOWNS),
        flat_types=", ".join(FLAT_TYPES),
    )


def _error(reason: str) -> dict[str, Any]:
    """Build the state update for a parse failure, logging the reason for debugging."""
    logger.warning("parse failed: %s", reason)
    return {"status": "error", "parse_reasoning": reason}


async def run(state: GraphState) -> dict[str, Any]:
    """Extract the model fields from ``state.user_message`` via a Claude Haiku call.

    Returns one of three shapes:

    - extracted: the six fields plus ``status = "pending"`` so the graph
      continues to ``postal_lookup``;
    - out_of_scope: ``status = "out_of_scope"`` and a ``parse_reasoning`` the
      narrate node uses to decline;
    - error: ``status = "error"`` when the model's reply does not parse as the
      expected JSON.
    """
    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": state.user_message},
                # Prefill the opening brace so the model continues with JSON only.
                {"role": "assistant", "content": "{"},
            ],
        )
    except Exception as exc:  # network or API fault becomes an error verdict
        return _error(f"Anthropic call failed: {type(exc).__name__}: {exc}")

    raw = "{" + "".join(b.text for b in resp.content if b.type == "text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error(f"model returned non-JSON output: {exc}: {raw!r}")

    status = payload.get("status")
    reasoning = payload.get("reasoning")

    if status == "out_of_scope":
        return {"status": "out_of_scope", "parse_reasoning": reasoning}

    if status != "extracted":
        return _error(f"unexpected status {status!r} in {payload!r}")

    fields = payload.get("fields")
    if not isinstance(fields, dict) or not set(fields) >= _EXPECTED_FIELD_KEYS:
        return _error(f"missing or malformed fields object in {payload!r}")

    town = fields["town"]
    flat_type = fields["flat_type"]
    return {
        "town": town.upper() if isinstance(town, str) else None,
        "flat_type": flat_type.upper() if isinstance(flat_type, str) else None,
        "floor_area_sqm": fields["floor_area_sqm"],
        "lease_commence_date": fields["lease_commence_date"],
        "month": fields["month"],
        "postal_code": fields["postal_code"],
        "status": "pending",
    }
