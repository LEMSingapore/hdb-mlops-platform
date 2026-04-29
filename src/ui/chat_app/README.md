# Chat App

Natural-language HDB resale price predictor. Users describe a flat in plain English;
Claude Haiku extracts the required fields via tool use and calls the FastAPI prediction
service for `/predict` and `/explain`.

## Requirements

`ANTHROPIC_API_KEY` must be set in the environment before starting the app:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run src/ui/chat_app/streamlit_app.py
```

The FastAPI service must also be running (default: `http://127.0.0.1:8000`).
Override the URL via the `API_BASE_URL` environment variable.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required) | Anthropic API key for Claude Haiku |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI prediction service |
| `REQUEST_TIMEOUT_SECONDS` | `10` | HTTP timeout for calls to the prediction service |

## Example queries

- "3-room in Tampines, 90 sqm, lease started 1992"
- "4-room Bishan flat, 105 sqm, started in 1988, postal 570100"
- "Executive in Woodlands, 147 sqm, lease 1995, selling in March 2025"
