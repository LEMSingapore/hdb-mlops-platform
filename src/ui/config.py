"""UI configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class UIConfig(BaseSettings):
    """Configuration for the Streamlit front end.

    All fields are overridable via environment variables of the same name
    (case-insensitive). Set API_BASE_URL in deployment to point at the
    FastAPI service running in the same stack.
    """

    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: int = 10
