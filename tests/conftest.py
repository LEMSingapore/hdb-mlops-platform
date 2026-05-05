"""Shared fixtures and helpers for the test suite."""

import sqlite3
from pathlib import Path

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
from training.train import MonthToFloatTransformer

_NUMERIC_FEATURES = ["floor_area_sqm", "lease_commence_date"]
_CATEGORICAL_FEATURES = ["town", "flat_type"]


def make_synthetic_features(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic HDB feature rows for training fixture models."""
    rng = np.random.default_rng(42)
    towns = ["TAMPINES", "ANG MO KIO", "BEDOK"]
    flat_types = ["4 ROOM", "3 ROOM", "5 ROOM"]
    months = ["2018-01", "2019-06", "2020-12", "2021-03", "2022-09"]
    rows = []
    for _ in range(n):
        rows.append(
            {
                "floor_area_sqm": float(rng.integers(65, 130)),
                "lease_commence_date": int(rng.integers(1980, 2010)),
                "town": towns[int(rng.integers(0, len(towns)))],
                "flat_type": flat_types[int(rng.integers(0, len(flat_types)))],
                "month": months[int(rng.integers(0, len(months)))],
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
            ("month", MonthToFloatTransformer(), ["month"]),
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
            name="model",
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


_TINY_DB_DDL = """\
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    town TEXT NOT NULL,
    flat_type TEXT NOT NULL,
    block TEXT NOT NULL,
    street_name TEXT NOT NULL,
    storey_range TEXT NOT NULL,
    floor_area_sqm REAL NOT NULL,
    flat_model TEXT NOT NULL,
    lease_commence_date INTEGER NOT NULL,
    resale_price REAL NOT NULL
);\
"""

# 20 hand-crafted rows: 2 towns, 3 flat types.
# TAMPINES 4 ROOM — 8 rows (exact-match pool for find_similar tests)
# TAMPINES 5 ROOM — 2 rows (insufficient for k=5; triggers town-only fallback)
# ANG MO KIO 4 ROOM — 5 rows
# ANG MO KIO 3 ROOM — 5 rows
# Columns: month, town, flat_type, block, street_name, storey_range,
#          floor_area_sqm, flat_model, lease_commence_date, resale_price
_TINY_ROWS: list[tuple] = [
    (
        "2022-01",
        "TAMPINES",
        "4 ROOM",
        "101",
        "TAMPINES AVE 1",
        "04 TO 06",
        90.0,
        "Model A",
        1990,
        450_000.0,
    ),
    (
        "2022-02",
        "TAMPINES",
        "4 ROOM",
        "102",
        "TAMPINES AVE 1",
        "07 TO 09",
        92.0,
        "Model A",
        1991,
        460_000.0,
    ),
    (
        "2022-03",
        "TAMPINES",
        "4 ROOM",
        "103",
        "TAMPINES AVE 2",
        "04 TO 06",
        95.0,
        "Model A",
        1989,
        470_000.0,
    ),
    (
        "2022-04",
        "TAMPINES",
        "4 ROOM",
        "104",
        "TAMPINES AVE 2",
        "10 TO 12",
        88.0,
        "Model A",
        1992,
        440_000.0,
    ),
    (
        "2022-05",
        "TAMPINES",
        "4 ROOM",
        "105",
        "TAMPINES AVE 3",
        "04 TO 06",
        93.0,
        "Model A",
        1990,
        465_000.0,
    ),
    (
        "2022-06",
        "TAMPINES",
        "4 ROOM",
        "106",
        "TAMPINES AVE 3",
        "07 TO 09",
        91.0,
        "Model A",
        1988,
        455_000.0,
    ),
    (
        "2022-07",
        "TAMPINES",
        "4 ROOM",
        "107",
        "TAMPINES ST 21",
        "04 TO 06",
        94.0,
        "Model B",
        1993,
        475_000.0,
    ),
    (
        "2022-08",
        "TAMPINES",
        "4 ROOM",
        "108",
        "TAMPINES ST 22",
        "01 TO 03",
        89.0,
        "Model B",
        1991,
        445_000.0,
    ),
    (
        "2022-09",
        "TAMPINES",
        "5 ROOM",
        "110",
        "TAMPINES AVE 4",
        "07 TO 09",
        110.0,
        "Improved",
        1990,
        520_000.0,
    ),
    (
        "2022-10",
        "TAMPINES",
        "5 ROOM",
        "111",
        "TAMPINES AVE 4",
        "10 TO 12",
        115.0,
        "Improved",
        1992,
        540_000.0,
    ),
    (
        "2022-01",
        "ANG MO KIO",
        "4 ROOM",
        "201",
        "AMK AVE 1",
        "04 TO 06",
        88.0,
        "Model A",
        1985,
        420_000.0,
    ),
    (
        "2022-02",
        "ANG MO KIO",
        "4 ROOM",
        "202",
        "AMK AVE 2",
        "07 TO 09",
        90.0,
        "Model A",
        1986,
        430_000.0,
    ),
    (
        "2022-03",
        "ANG MO KIO",
        "4 ROOM",
        "203",
        "AMK AVE 3",
        "04 TO 06",
        92.0,
        "Model A",
        1984,
        440_000.0,
    ),
    (
        "2022-04",
        "ANG MO KIO",
        "4 ROOM",
        "204",
        "AMK AVE 4",
        "10 TO 12",
        86.0,
        "Model B",
        1987,
        415_000.0,
    ),
    (
        "2022-05",
        "ANG MO KIO",
        "4 ROOM",
        "205",
        "AMK AVE 5",
        "04 TO 06",
        91.0,
        "Model B",
        1985,
        425_000.0,
    ),
    (
        "2022-01",
        "ANG MO KIO",
        "3 ROOM",
        "210",
        "AMK AVE 1",
        "04 TO 06",
        65.0,
        "Model A",
        1982,
        300_000.0,
    ),
    (
        "2022-02",
        "ANG MO KIO",
        "3 ROOM",
        "211",
        "AMK AVE 2",
        "07 TO 09",
        67.0,
        "Model A",
        1983,
        310_000.0,
    ),
    (
        "2022-03",
        "ANG MO KIO",
        "3 ROOM",
        "212",
        "AMK AVE 3",
        "04 TO 06",
        68.0,
        "Model A",
        1981,
        305_000.0,
    ),
    (
        "2022-04",
        "ANG MO KIO",
        "3 ROOM",
        "213",
        "AMK AVE 4",
        "10 TO 12",
        66.0,
        "Model B",
        1984,
        315_000.0,
    ),
    (
        "2022-05",
        "ANG MO KIO",
        "3 ROOM",
        "214",
        "AMK AVE 5",
        "04 TO 06",
        70.0,
        "Model B",
        1982,
        320_000.0,
    ),
]


@pytest.fixture
def tiny_sqlite_db(tmp_path) -> Path:
    """Build a minimal SQLite DB with 20 hand-crafted rows for data layer tests.

    Covers 2 towns (TAMPINES, ANG MO KIO) and 3 flat types (3 ROOM, 4 ROOM,
    5 ROOM). The distribution is intentionally uneven to exercise the
    find_similar fallback path (TAMPINES 5 ROOM has only 2 rows).

    Shared between tests/data/ and tests/training/.
    """
    db_path = tmp_path / "test_hdb.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_TINY_DB_DDL)
        conn.executemany(
            "INSERT INTO transactions "
            "(month, town, flat_type, block, street_name, storey_range, "
            "floor_area_sqm, flat_model, lease_commence_date, resale_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _TINY_ROWS,
        )
        conn.commit()
    finally:
        conn.close()
    return Path(db_path)
