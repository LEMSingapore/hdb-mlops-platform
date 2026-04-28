"""Shared fixtures and helpers for the test suite."""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from serving.model_loader import ExplainerBundle

_NUMERIC_FEATURES = ["floor_area_sqm", "lease_commence_date", "float_time_series"]
_CATEGORICAL_FEATURES = ["town", "storey_range", "flat_info"]


def make_synthetic_features(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic HDB feature rows for training fixture models."""
    rng = np.random.default_rng(42)
    towns = ["TAMPINES", "ANG MO KIO", "BEDOK"]
    flat_infos = ["4 ROOM Model A", "3 ROOM New Generation", "5 ROOM Improved"]
    storey_ranges = ["04 TO 06", "07 TO 09", "10 TO 12"]
    rows = []
    for _ in range(n):
        rows.append(
            {
                "floor_area_sqm": float(rng.integers(65, 130)),
                "lease_commence_date": int(rng.integers(1980, 2010)),
                "float_time_series": 2018.0 + float(rng.integers(0, 6)),
                "town": towns[int(rng.integers(0, len(towns)))],
                "storey_range": storey_ranges[int(rng.integers(0, len(storey_ranges)))],
                "flat_info": flat_infos[int(rng.integers(0, len(flat_infos)))],
            }
        )
    X = pd.DataFrame(rows)
    y = pd.Series(
        rng.integers(350_000, 650_000, n, dtype=np.int64).astype(float),
        name="resale_price",
    )
    return X, y


def build_fixture_pipeline() -> Pipeline:
    """Return a minimal sklearn pipeline suitable for test fixtures."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", _NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(n_estimators=5, random_state=7)),
        ]
    )


def register_model_version(
    tracking_uri: str,
    model_name: str = "hdb-predictor",
    experiment_name: str = "test-hdb",
) -> str:
    """Train and register one model version. Returns the version string."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    X, y = make_synthetic_features(n=80)
    pipeline = build_fixture_pipeline()
    pipeline.fit(X, y)
    signature = mlflow.models.infer_signature(X, pipeline.predict(X))
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )
    return info.registered_model_version


def build_fixture_explainer_bundle(pipeline: Pipeline) -> ExplainerBundle:
    """Build an ExplainerBundle from a fitted fixture pipeline.

    Used in app tests that need a real SHAP explainer without an MLflow roundtrip.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(regressor)
    return ExplainerBundle(
        explainer=explainer,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


class StubModelLoader:
    """Test double for ModelLoader. No MLflow calls; state is set at construction."""

    def __init__(
        self,
        *,
        model=None,
        version=None,
        run_id: str | None = None,
        explainer: ExplainerBundle | None = None,
    ) -> None:
        self._model = model
        self._version = version
        self._run_id = run_id
        self._explainer = explainer

    def load_initial(self) -> None:
        pass

    def start_background_polling(self) -> None:
        pass

    def get_model(self):
        if self._model is None:
            raise RuntimeError("Model not yet loaded; startup may have failed.")
        return self._model

    def get_version(self):
        return self._version

    def get_run_id(self) -> str | None:
        return self._run_id

    def get_explainer(self) -> ExplainerBundle | None:
        return self._explainer


@pytest.fixture
def isolated_mlflow_uri(tmp_path):
    """Per-test SQLite tracking URI — prevents state leaking between tests."""
    db = tmp_path / "mlflow_test.db"
    return f"sqlite:///{db}"
