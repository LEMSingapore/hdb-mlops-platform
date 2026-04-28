"""Integration tests for the training pipeline.

The end-to-end test uses n_estimators=20 and synthetic CSV files to stay
well under 30 seconds while exercising the full train() path: data loading,
feature engineering, model fitting, MLflow logging, registration, and alias
promotion.
"""

import argparse

import mlflow
import mlflow.exceptions
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from training.config import TrainingConfig
from training.train import (
    _RAW_FILES,
    build_pipeline,
    compute_metrics,
    load_and_prepare_data,
    train,
)

_TOWNS = ["TAMPINES", "ANG MO KIO", "BEDOK", "JURONG WEST", "WOODLANDS"]
_FLAT_TYPES = ["3 ROOM", "4 ROOM", "5 ROOM"]
_FLAT_MODELS = ["Model A", "New Generation", "Improved"]
_STOREY_RANGES = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12"]


def _write_synthetic_csv(path, n_rows: int, rng: np.random.Generator) -> None:
    """Write one synthetic raw HDB CSV file at the given path."""
    data = {
        "town": [_TOWNS[rng.integers(0, len(_TOWNS))] for _ in range(n_rows)],
        "flat_type": [_FLAT_TYPES[rng.integers(0, len(_FLAT_TYPES))] for _ in range(n_rows)],
        "flat_model": [_FLAT_MODELS[rng.integers(0, len(_FLAT_MODELS))] for _ in range(n_rows)],
        "storey_range": [
            _STOREY_RANGES[rng.integers(0, len(_STOREY_RANGES))] for _ in range(n_rows)
        ],
        "floor_area_sqm": rng.integers(65, 130, n_rows).astype(float).tolist(),
        "lease_commence_date": rng.integers(1980, 2010, n_rows).tolist(),
        "month": [
            f"{rng.integers(2015, 2024)}-{int(rng.integers(1, 13)):02d}" for _ in range(n_rows)
        ],
        "resale_price": rng.integers(350_000, 700_000, n_rows).tolist(),
    }
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Write all five expected raw CSV files with synthetic HDB data."""
    rng = np.random.default_rng(0)
    for filename in _RAW_FILES:
        _write_synthetic_csv(tmp_path / filename, n_rows=60, rng=rng)
    return tmp_path


@pytest.fixture
def train_mlflow_uri(tmp_path):
    """Isolated SQLite tracking URI for training tests."""
    db = tmp_path / "train_mlflow.db"
    return f"sqlite:///{db}"


@pytest.fixture
def train_config(train_mlflow_uri):
    return TrainingConfig(
        mlflow_tracking_uri=train_mlflow_uri,
        mlflow_experiment_name="test-hdb-train",
        model_registry_name="hdb-predictor",
    )


@pytest.fixture
def minimal_train_args(synthetic_data_dir):
    return argparse.Namespace(
        data_dir=str(synthetic_data_dir),
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        max_features=0.5,
        test_size=0.2,
        promote="champion",
    )


class TestComputeMetrics:
    def test_returns_rmse_mae_r2_keys(self):
        y_true = pd.Series([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    def test_perfect_predictions_give_r2_of_one(self):
        y = pd.Series([100.0, 200.0, 300.0])
        metrics = compute_metrics(y, y.to_numpy())
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)

    def test_rmse_is_positive(self):
        y_true = pd.Series([100.0, 200.0])
        y_pred = np.array([150.0, 250.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] > 0


class TestBuildPipeline:
    def test_pipeline_fits_and_predicts(self):
        from tests.conftest import make_synthetic_features

        pipeline = build_pipeline(
            n_estimators=5,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            max_features=0.5,
        )
        X, y = make_synthetic_features(n=60)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert preds.shape == (60,)
        assert all(p > 0 for p in preds)


class TestLoadAndPrepareData:
    def test_returns_correct_feature_columns(self, synthetic_data_dir):
        X, y = load_and_prepare_data(synthetic_data_dir)
        expected_cols = {
            "floor_area_sqm",
            "lease_commence_date",
            "float_time_series",
            "town",
            "storey_range",
            "flat_info",
        }
        assert set(X.columns) == expected_cols

    def test_flat_info_combines_type_and_model(self, synthetic_data_dir):
        X, _ = load_and_prepare_data(synthetic_data_dir)
        # flat_info should be "FLAT_TYPE FLAT_MODEL" with consistent casing
        assert X["flat_info"].str.contains(" ").all()

    def test_float_time_series_is_numeric(self, synthetic_data_dir):
        X, _ = load_and_prepare_data(synthetic_data_dir)
        assert pd.api.types.is_float_dtype(X["float_time_series"])
        # Values should be in the range covered by the synthetic months (2015-2023)
        assert X["float_time_series"].between(2015.0, 2024.0).all()

    def test_rows_from_all_files_are_concatenated(self, synthetic_data_dir):
        X, _ = load_and_prepare_data(synthetic_data_dir)
        # 5 files * 60 rows each
        assert len(X) == 300


class TestTrainEndToEnd:
    def test_train_registers_model_and_logs_metrics(
        self, minimal_train_args, train_config, train_mlflow_uri
    ):
        """Reduced-size end-to-end run; should complete well under 30 seconds."""
        run_id = train(minimal_train_args, train_config)

        assert isinstance(run_id, str)
        assert len(run_id) > 0

        mlflow.set_tracking_uri(train_mlflow_uri)
        client = MlflowClient()

        # Model is registered under the expected name
        versions = client.search_model_versions("name='hdb-predictor'")
        assert len(versions) >= 1

        # Run has the expected metrics
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert metrics["test_rmse"] > 0
        assert metrics["test_mae"] > 0

    def test_train_logs_train_metrics_alongside_test_metrics(
        self, minimal_train_args, train_config, train_mlflow_uri
    ):
        run_id = train(minimal_train_args, train_config)
        mlflow.set_tracking_uri(train_mlflow_uri)
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        assert "train_rmse" in metrics
        assert "train_mae" in metrics
        assert "train_r2" in metrics

    def test_train_attaches_signature_to_logged_model(
        self, minimal_train_args, train_config, train_mlflow_uri
    ):
        run_id = train(minimal_train_args, train_config)
        mlflow.set_tracking_uri(train_mlflow_uri)
        model_info = mlflow.models.get_model_info(f"runs:/{run_id}/model")
        assert model_info.signature is not None
        assert model_info.signature.inputs is not None

    def test_train_with_promote_sets_champion_alias(
        self, minimal_train_args, train_config, train_mlflow_uri
    ):
        train(minimal_train_args, train_config)
        mlflow.set_tracking_uri(train_mlflow_uri)
        client = MlflowClient()
        mv = client.get_model_version_by_alias("hdb-predictor", "champion")
        assert mv is not None

    def test_train_without_promote_leaves_no_alias(
        self, synthetic_data_dir, train_config, train_mlflow_uri
    ):
        args_no_promote = argparse.Namespace(
            data_dir=str(synthetic_data_dir),
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            max_features=0.5,
            test_size=0.2,
            promote=None,
        )
        train(args_no_promote, train_config)
        mlflow.set_tracking_uri(train_mlflow_uri)
        client = MlflowClient()
        with pytest.raises(mlflow.exceptions.MlflowException):
            client.get_model_version_by_alias("hdb-predictor", "champion")
