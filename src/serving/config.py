"""Serving configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    """Configuration for the FastAPI serving application.

    All fields are overridable via environment variables of the same name
    (case-insensitive). In deployment, set MLFLOW_TRACKING_URI to point at
    the shared tracking server running alongside the serving pod.
    """

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "hdb-predictor"
    model_alias: str = "champion"
    reload_interval_seconds: int = 60
