"""Happy-path test for get_model_info.

Registers a fixture model version with metrics and params, points the
@champion alias at it, and verifies the tool reads provenance from the
registry. The MLFLOW_TRACKING_URI env var steers MCPConfig at call time.
"""

import mlflow
import mlflow.sklearn
import pytest
from fastmcp import Client
from mlflow import MlflowClient

from mcp_server.server import mcp
from tests.conftest import build_fixture_pipeline, make_synthetic_features

_PARAMS = {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 3}
_METRICS = {
    "train_rmse": 1000.0,
    "test_rmse": 1200.0,
    "train_mae": 800.0,
    "test_mae": 900.0,
    "train_r2": 0.95,
    "test_r2": 0.90,
}


def _register_champion_with_metadata(tracking_uri: str) -> int:
    """Register one fixture version with params + metrics and alias it @champion."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-model-info")
    X, y = make_synthetic_features(n=80)
    pipeline = build_fixture_pipeline()
    pipeline.fit(X, y)
    signature = mlflow.models.infer_signature(X, pipeline.predict(X))
    with mlflow.start_run():
        mlflow.log_params(_PARAMS)
        mlflow.log_metrics(_METRICS)
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name="hdb-predictor",
        )
    client = MlflowClient()
    client.set_registered_model_alias("hdb-predictor", "champion", info.registered_model_version)
    return int(info.registered_model_version)


@pytest.fixture
def champion_registry(isolated_mlflow_uri, monkeypatch):
    """Registry with a champion version; MCPConfig reads it via the env var."""
    version = _register_champion_with_metadata(isolated_mlflow_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    return version


class TestGetModelInfo:
    async def test_happy_path_returns_champion_metadata(self, champion_registry) -> None:
        version = champion_registry
        async with Client(mcp) as client:
            result = await client.call_tool("get_model_info", {})
        data = result.structured_content
        assert data["version"] == version
        assert data["alias"] == "champion"
        assert data["run_id"]
        assert data["registered_at"]  # ISO-8601 string

    async def test_metrics_and_params_round_trip(self, champion_registry) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("get_model_info", {})
        data = result.structured_content
        assert data["metrics"]["test_rmse"] == 1200.0
        assert data["metrics"]["test_r2"] == 0.90
        # MLflow stores params as strings.
        assert data["params"]["n_estimators"] == "5"

    async def test_features_are_the_five_field_schema(self, champion_registry) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("get_model_info", {})
        assert result.structured_content["features"] == [
            "town",
            "flat_type",
            "floor_area_sqm",
            "lease_commence_date",
            "month",
        ]
