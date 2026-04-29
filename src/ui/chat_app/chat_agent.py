"""Claude tool-use loop for extracting HDB flat features from natural language
and calling the FastAPI prediction service.

The LLM never runs inference itself — it handles slot-filling, follow-up
questions, and response framing. All prediction and lookup work is delegated
to the FastAPI service via :mod:`ui.chat_app.predictor`.
"""

import json
import logging
from datetime import date
from typing import Any

from anthropic import Anthropic

from ui.chat_app.config import ChatConfig
from ui.chat_app.predictor import (
    APIConnectionError,
    ServerError,
    ServiceUnavailableError,
    ValidationError,
    explain_price,
    lookup_postal,
    predict_price,
)

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


_config = ChatConfig()
# Pass None when the field is empty so the SDK reads ANTHROPIC_API_KEY from the environment
# directly rather than treating an empty string as a set (but invalid) key.
_client = Anthropic(api_key=_config.anthropic_api_key or None)

TOWNS: list[str] = [
    "ANG MO KIO",
    "BEDOK",
    "BISHAN",
    "BUKIT BATOK",
    "BUKIT MERAH",
    "BUKIT PANJANG",
    "BUKIT TIMAH",
    "CENTRAL AREA",
    "CHOA CHU KANG",
    "CLEMENTI",
    "GEYLANG",
    "HOUGANG",
    "JURONG EAST",
    "JURONG WEST",
    "KALLANG/WHAMPOA",
    "MARINE PARADE",
    "PASIR RIS",
    "PUNGGOL",
    "QUEENSTOWN",
    "SEMBAWANG",
    "SENGKANG",
    "SERANGOON",
    "TAMPINES",
    "TOA PAYOH",
    "WOODLANDS",
    "YISHUN",
]

FLAT_TYPES: list[str] = [
    "1 ROOM",
    "2 ROOM",
    "3 ROOM",
    "4 ROOM",
    "5 ROOM",
    "EXECUTIVE",
    "MULTI-GENERATION",
]

TOOLS: list[Any] = [
    {
        "name": "lookup_postal_code",
        "description": (
            "Resolve a Singapore postal code to its address, including block, "
            "street, and HDB town. Call this when the user provides a postal code "
            "and the town is not already known. Postal code is NOT a model input — "
            "it is a helper that lets you skip asking the user for their town."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "postal_code": {
                    "type": "integer",
                    "description": "6-digit Singapore postal code.",
                },
            },
            "required": ["postal_code"],
        },
    },
    {
        "name": "predict_hdb_price",
        "description": (
            "Predict the resale price of a Singapore HDB flat. Only call this "
            "once all five model fields are known with confidence. If any field "
            "is missing or ambiguous, ask the user a concise follow-up question "
            "instead of calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "town": {
                    "type": "string",
                    "enum": TOWNS,
                    "description": "HDB planning town (uppercase).",
                },
                "flat_type": {
                    "type": "string",
                    "enum": FLAT_TYPES,
                    "description": "HDB flat type.",
                },
                "floor_area_sqm": {
                    "type": "number",
                    "description": "Floor area in square metres.",
                },
                "lease_commence_date": {
                    "type": "integer",
                    "description": "Year the lease began, e.g. 1990.",
                },
                "month": {
                    "type": "string",
                    "description": (
                        "Transaction month in YYYY-MM format. "
                        "If the user does not specify a month, use the current month."
                    ),
                },
            },
            "required": [
                "town",
                "flat_type",
                "floor_area_sqm",
                "lease_commence_date",
                "month",
            ],
        },
    },
    {
        "name": "explain_hdb_price",
        "description": (
            "Explain a predicted HDB resale price by returning the top SHAP "
            "contributors. Call this after predict_hdb_price when you want to "
            "narrate why the price is what it is. Uses the same five model fields."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "town": {
                    "type": "string",
                    "enum": TOWNS,
                    "description": "HDB planning town (uppercase).",
                },
                "flat_type": {
                    "type": "string",
                    "enum": FLAT_TYPES,
                    "description": "HDB flat type.",
                },
                "floor_area_sqm": {
                    "type": "number",
                    "description": "Floor area in square metres.",
                },
                "lease_commence_date": {
                    "type": "integer",
                    "description": "Year the lease began.",
                },
                "month": {
                    "type": "string",
                    "description": "Transaction month in YYYY-MM format.",
                },
            },
            "required": [
                "town",
                "flat_type",
                "floor_area_sqm",
                "lease_commence_date",
                "month",
            ],
        },
    },
]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are an assistant that estimates Singapore HDB resale prices using a "
    "GradientBoostingRegressor model served via a FastAPI backend.\n\n"
    "Today's date is {today}. The current month is {current_month}.\n\n"
    "Tools available:\n"
    "1. lookup_postal_code — resolves a postal code to town, block, and street. "
    "Use this when the user provides a postal code and the town is not yet known. "
    "Postal code is NOT a model input — it only helps you find the town.\n"
    "2. predict_hdb_price — predicts the resale price from five fields: "
    "town, flat_type, floor_area_sqm, lease_commence_date, month (YYYY-MM).\n"
    "3. explain_hdb_price — same five inputs; returns the top SHAP contributors. "
    "Call this after predicting to narrate why the price is what it is.\n\n"
    "Rules:\n"
    "- The five model fields are: town, flat_type, floor_area_sqm, "
    "lease_commence_date, month.\n"
    "- If the user provides a postal code and town is not yet known, call "
    "lookup_postal_code first. If it returns a null town, ask the user directly.\n"
    "- If any required field is missing or ambiguous, ask ONE concise follow-up "
    "question. Do not call predict_hdb_price until all five fields are confirmed.\n"
    "- If month is not specified, use the current month ({current_month}).\n"
    "- Normalise town and flat_type to the allowed enum values (uppercase).\n"
    "- After predicting, call explain_hdb_price and narrate the top contributors "
    "in plain English — for example: 'The price is driven up by floor area "
    "(+S$45k) and town (+S$12k), and pulled down by older lease (-S$8k).'\n"
    "- Present the predicted price in Singapore dollars and briefly restate the "
    "flat the estimate is for. Remind the user the figure is an estimate.\n"
    "- Politely decline requests about non-HDB properties (condos, landed, "
    "commercial) — this model only covers HDB resale flats."
)


def _system_prompt() -> str:
    today = date.today()
    return _SYSTEM_PROMPT_TEMPLATE.format(
        today=today.strftime("%d %B %Y"),
        current_month=today.strftime("%Y-%m"),
    )


def _dispatch_tool(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool call and return (result_text, is_error)."""
    try:
        if tool_name == "lookup_postal_code":
            addr = lookup_postal(**tool_input)
            if addr is None:
                return (
                    json.dumps({"found": False}),
                    False,
                )
            return (
                json.dumps(
                    {
                        "found": True,
                        "town": addr.town,
                        "block": addr.block,
                        "street_full": addr.street_full,
                    }
                ),
                False,
            )

        if tool_name == "predict_hdb_price":
            result = predict_price(**tool_input)
            return (
                json.dumps(
                    {
                        "predicted_resale_price": result["predicted_resale_price"],
                        "model_version": result["model_version"],
                        "model_alias": result["model_alias"],
                    }
                ),
                False,
            )

        if tool_name == "explain_hdb_price":
            result = explain_price(**tool_input)
            return json.dumps(result), False

        return f"Unknown tool: {tool_name}", True

    except (ServiceUnavailableError, ValidationError, ServerError, APIConnectionError) as exc:
        logger.warning("Tool %s failed: %s: %s", tool_name, type(exc).__name__, exc)
        return (
            (
                f"error: {type(exc).__name__}: {exc}. "
                "Report this exact error message to the user verbatim — do NOT "
                "say the system is temporarily unavailable or ask them to retry."
            ),
            True,
        )


def chat_turn(history: list) -> tuple[list, str]:
    """Run one user turn through the tool-use loop.

    Args:
        history: Running Anthropic-format messages list. Mutated in place with
            the assistant reply and any tool results.

    Returns:
        Tuple of (updated history, assistant text for display).
    """
    system = _system_prompt()

    while True:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=history,
        )

        history.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "tool_use":
            tool_use = next(b for b in resp.content if b.type == "tool_use")
            result_text, is_error = _dispatch_tool(tool_use.name, tool_use.input)

            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result_text,
                            "is_error": is_error,
                        }
                    ],
                }
            )
            continue

        text = "".join(b.text for b in resp.content if b.type == "text")
        return history, text
