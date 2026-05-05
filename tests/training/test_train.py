"""Integration tests for the training pipeline.

The end-to-end tests use n_estimators=20 and the tiny_sqlite_db fixture (20
hand-crafted rows) to stay well under 30 seconds while exercising the full
train() path: data loading from SQLite, preprocessing, model fitting, MLflow
logging, registration, and automatic @champion alias promotion.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.exceptions
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from training.config import TrainingConfig
from training.train import (
    FEATURES,
    build_pipeline,
    compute_metrics,
    train,
)


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
def minimal_train_args():
    return argparse.Namespace(
        quick=False,
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        max_features=0.5,
        test_size=0.2,
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

    def test_pipeline_accepts_raw_month_string(self):
        """The pipeline must accept raw 'YYYY-MM' input — no external preprocessing."""
        from tests.conftest import make_synthetic_features

        pipeline = build_pipeline(
            n_estimators=5,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            max_features=0.5,
        )
        X, y = make_synthetic_features(n=40)
        pipeline.fit(X, y)
        single_row = pd.DataFrame(
            [
                {
                    "town": "TAMPINES",
                    "flat_type": "4 ROOM",
                    "floor_area_sqm": 93.0,
                    "lease_commence_date": 1990,
                    "month": "2024-06",
                }
            ]
        )
        preds = pipeline.predict(single_row)
        assert preds.shape == (1,)
        assert preds[0] > 0


class TestTrainEndToEnd:
    def test_train_reads_from_sqlite_and_registers_model(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        """Reduced-size end-to-end run; should complete well under 30 seconds."""
        run_id = train(minimal_train_args, train_config, db_path=tiny_sqlite_db)

        assert isinstance(run_id, str)
        assert len(run_id) > 0

        mlflow.set_tracking_uri(train_mlflow_uri)
        client = MlflowClient()
        versions = client.search_model_versions("name='hdb-predictor'")
        assert len(versions) >= 1

    def test_train_logs_test_and_train_metrics(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        run_id = train(minimal_train_args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        for key in ("test_rmse", "test_mae", "test_r2", "train_rmse", "train_mae", "train_r2"):
            assert key in metrics, f"Missing metric: {key}"
        assert metrics["test_rmse"] > 0

    def test_train_attaches_signature_to_logged_model(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        run_id = train(minimal_train_args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        model_info = mlflow.models.get_model_info(f"runs:/{run_id}/model")
        assert model_info.signature is not None
        assert model_info.signature.inputs is not None

    def test_train_auto_sets_champion_alias(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        """@champion is set automatically on every successful run."""
        train(minimal_train_args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        client = MlflowClient()
        mv = client.get_model_version_by_alias("hdb-predictor", "champion")
        assert mv is not None

    def test_quick_flag_uses_twenty_estimators(
        self,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        args = argparse.Namespace(
            quick=True,
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            max_features=0.5,
            test_size=0.2,
        )
        run_id = train(args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        run = mlflow.get_run(run_id)
        assert run.data.params["n_estimators"] == "20"

    def test_logged_model_accepts_raw_input_without_external_preprocessing(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        """The logged model artifact must accept the 5 raw fields directly."""
        run_id = train(minimal_train_args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        raw_input = pd.DataFrame(
            [
                {
                    "town": "TAMPINES",
                    "flat_type": "4 ROOM",
                    "floor_area_sqm": 93.0,
                    "lease_commence_date": 1990,
                    "month": "2024-06",
                }
            ]
        )
        preds = model.predict(raw_input)
        assert preds.shape == (1,)
        assert preds[0] > 0

    def test_train_uses_only_feature_columns_from_sqlite(
        self,
        minimal_train_args: argparse.Namespace,
        train_config: TrainingConfig,
        train_mlflow_uri: str,
        tiny_sqlite_db: Path,
    ) -> None:
        """Training must use only FEATURES — extra SQLite columns (id, block, etc.) are ignored."""
        run_id = train(minimal_train_args, train_config, db_path=tiny_sqlite_db)
        mlflow.set_tracking_uri(train_mlflow_uri)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        # A DataFrame with only the 5 expected features must predict without error.
        df = pd.DataFrame(
            [
                {
                    "town": "ANG MO KIO",
                    "flat_type": "3 ROOM",
                    "floor_area_sqm": 67.0,
                    "lease_commence_date": 1983,
                    "month": "2022-03",
                }
            ]
        )
        assert set(df.columns) == set(FEATURES)
        preds = model.predict(df)
        assert preds.shape == (1,)
