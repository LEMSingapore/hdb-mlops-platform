"""Model provenance tool.

Queries the MLflow registry for the version currently behind the ``@champion``
alias and returns its metrics, hyperparameters, and registration time. Reads
the registry directly rather than going through the in-memory loader so the
reported metadata reflects the registry's current state.
"""

from datetime import UTC, datetime
from typing import TypedDict

from mlflow import MlflowClient

from mcp_server.config import MCPConfig
from serving.schemas import HDBFeatureInput

_FEATURES: list[str] = list(HDBFeatureInput.model_fields.keys())


class ModelInfo(TypedDict):
    """Metadata for the model version currently serving as champion."""

    version: int
    alias: str
    run_id: str
    metrics: dict[str, float]
    params: dict[str, str]
    registered_at: str
    features: list[str]


def get_model_info() -> ModelInfo:
    """Return metadata for the model version behind the @champion alias.

    Returns:
        The champion version's number, run ID, training metrics (train/test
        RMSE, MAE, R2), hyperparameters, ISO-8601 registration timestamp, and
        the five input feature names the model expects.
    """
    config = MCPConfig()
    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(config.model_name, config.model_alias)
    run = client.get_run(mv.run_id)

    registered_at = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=UTC).isoformat()

    return ModelInfo(
        version=int(mv.version),
        alias=config.model_alias,
        run_id=mv.run_id,
        metrics={k: float(v) for k, v in run.data.metrics.items()},
        params=dict(run.data.params),
        registered_at=registered_at,
        features=_FEATURES,
    )
