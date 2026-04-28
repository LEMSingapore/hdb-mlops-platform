"""Training configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Configuration for the training pipeline.

    All fields are overridable via environment variables of the same name
    (case-insensitive). For example, set MLFLOW_TRACKING_URI to point at a
    remote tracking server without touching code.
    """

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "hdb-resale-price"
    model_registry_name: str = "hdb-predictor"
