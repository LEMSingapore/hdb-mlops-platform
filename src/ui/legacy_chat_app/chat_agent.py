# mypy: ignore-errors
# ruff: noqa
"""Claude tool-use loop for extracting HDB flat features from natural language
and calling the local predictor. The LLM never sees the model — it only handles
slot-filling, follow-up questions, and response framing."""

from anthropic import Anthropic

from predictor import FLAT_TYPES, TOWNS, predict_price

MODEL = "claude-haiku-4-5-20251001"

_client = Anthropic()

TOOLS = [
    {
        "name": "predict_hdb_price",
        "description": (
            "Predict the resale price of a Singapore HDB flat. Only call this "
            "once all five fields are known with confidence. If any field is "
            "missing or ambiguous, ask the user a concise follow-up question "
            "instead of calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "floor_area": {
                    "type": "number",
                    "description": "Floor area in square meters (sqm).",
                },
                "lease_commence_date": {
                    "type": "integer",
                    "description": "Year the lease began, between 1960 and the current year.",
                },
                "postal_code": {
                    "type": "integer",
                    "description": "6-digit Singapore postal code.",
                },
                "town": {
                    "type": "string",
                    "enum": TOWNS,
                    "description": "HDB town name (uppercase).",
                },
                "flat_type": {
                    "type": "string",
                    "enum": FLAT_TYPES,
                    "description": "HDB flat type.",
                },
            },
            "required": [
                "floor_area",
                "lease_commence_date",
                "postal_code",
                "town",
                "flat_type",
            ],
        },
    }
]

SYSTEM_PROMPT = (
    "You are an assistant that estimates Singapore HDB resale prices. "
    "Users describe a flat in plain English; your job is to extract the five "
    "required fields (floor_area in sqm, lease_commence_date as a year, "
    "postal_code, town, flat_type) and call the predict_hdb_price tool.\n\n"
    "Rules:\n"
    "- If any field is missing or ambiguous, ask ONE concise follow-up question "
    "and do not call the tool yet.\n"
    "- Normalize town and flat_type to the allowed enum values (uppercase).\n"
    "- Politely decline requests about non-HDB properties (condos, landed, "
    "commercial) — this model only covers HDB resale flats.\n"
    "- When you get a tool result, present the price in Singapore dollars and "
    "briefly restate the flat the estimate is for. Remind the user the figure "
    "is an estimate."
)


def chat_turn(history: list) -> tuple[list, str]:
    """Run one user turn. `history` is the running Anthropic messages list and
    is mutated in place. Returns (history, assistant_text_for_display)."""
    while True:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=history,
        )

        history.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "tool_use":
            tool_use = next(b for b in resp.content if b.type == "tool_use")
            try:
                price = predict_price(**tool_use.input)
                result_text = f"predicted_price_sgd={price}"
                is_error = False
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[chat_agent] predict_price failed:\n{tb}", flush=True)
                result_text = (
                    f"error: {type(e).__name__}: {e}. "
                    "Report this exact error message to the user verbatim "
                    "so they can debug — do NOT say the system is temporarily "
                    "unavailable or ask them to retry."
                )
                is_error = True

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
