"""Configuration for the chat-based Streamlit front end."""

from pydantic import Field
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    """Settings for the chat app.

    All fields are overridable via environment variables of the same name
    (case-insensitive). ANTHROPIC_API_KEY must be set in the environment —
    the chat agent uses the Anthropic API directly.
    """

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: int = 10

    model_config = {"populate_by_name": True}
