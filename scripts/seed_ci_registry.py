"""Seed a synthetic ``@champion`` into a fresh MLflow registry for CI.

CI has no training data — ``data/`` is gitignored and the DVC remote needs MinIO
credentials the runner does not hold — and the real champion artifact is 8.4 MB,
too large to commit past the ``check-added-large-files`` hook. So the compose
smoke test in ``.github/workflows/ci.yml`` cannot mount the production registry.

This script builds a tiny GradientBoostingRegressor on synthetic HDB rows, logs
it through local-process MLflow into ``./mlflow.db`` + ``./mlruns``, and points
the ``@champion`` alias at the new version. The model reuses the production
``build_pipeline`` from ``training.train``, so it carries the same ``preprocessor``
and ``regressor`` named steps the SHAP loader expects and accepts the same
feature columns the ``/predict`` request sends. The smoke test asserts response
*shape*, not a specific price, so a synthetic champion is sufficient signal.

The artifacts land under ``./mlruns`` exactly as local-process MLflow lays them
out. Run ``scripts/migrate_artifact_paths_to_proxy.py`` afterwards to rewrite the
stored paths to ``mlflow-artifacts:`` proxy URIs so the containerised tracking
server can serve them — the same two-step flow that produced the real champion in
Phase 2 Session B. See docs/adr/0007-ci-workflow-and-registry-strategy.md.

Usage:

    python scripts/seed_ci_registry.py
    python scripts/migrate_artifact_paths_to_proxy.py
"""

import logging
import sys
from pathlib import Path

# The package uses a src layout; tests rely on pythonpath=["src"]. A standalone
# script gets no such hook, so put src on the import path explicitly. This keeps
# the seed working whether or not the project is installed editable.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from training.train import build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

_MODEL_NAME = "hdb-predictor"
_ALIAS = "champion"
_EXPERIMENT_NAME = "ci-smoke"
_DB_PATH = _REPO_ROOT / "mlflow.db"
_TRACKING_URI = f"sqlite:///{_DB_PATH}"

# Match the category vocabularies the smoke-test request uses (TAMPINES, 4 ROOM)
# so the one-hot encoder produces a non-zero encoding rather than falling back to
# the handle_unknown='ignore' all-zero path.
_TOWNS = ["TAMPINES", "ANG MO KIO", "BEDOK"]
_FLAT_TYPES = ["4 ROOM", "3 ROOM", "5 ROOM"]
_MONTHS = ["2018-01", "2019-06", "2020-12", "2021-03", "2022-09"]


def _synthetic_training_frame(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic HDB feature rows and a correlated target.

    The target is a linear function of floor area and lease year plus noise, so
    the boosted trees learn a non-trivial mapping and SHAP contributions are
    meaningful rather than degenerate.
    """
    rng = np.random.default_rng(7)
    floor_area = rng.integers(65, 130, n).astype(float)
    lease = rng.integers(1980, 2015, n)
    town = rng.choice(_TOWNS, n)
    flat_type = rng.choice(_FLAT_TYPES, n)
    month = rng.choice(_MONTHS, n)
    price = 180_000.0 + 3_200.0 * floor_area + 1_500.0 * (lease - 1980) + rng.normal(0, 15_000, n)
    features = pd.DataFrame(
        {
            "town": town,
            "flat_type": flat_type,
            "floor_area_sqm": floor_area,
            "lease_commence_date": lease,
            "month": month,
        }
    )
    return features, pd.Series(price, name="resale_price")


def seed() -> int:
    """Train, log, and promote a synthetic champion. Returns the version number."""
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(_EXPERIMENT_NAME)

    features, target = _synthetic_training_frame()
    # Small n_estimators keeps the artifact tiny and the seed sub-second while
    # still exercising the real pipeline structure and SHAP TreeExplainer path.
    pipeline = build_pipeline(
        n_estimators=15,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=2,
        max_features=1.0,
    )
    pipeline.fit(features, target)
    signature = mlflow.models.infer_signature(features, pipeline.predict(features))

    with mlflow.start_run(run_name="ci-seed"):
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name=_MODEL_NAME,
        )

    version = int(info.registered_model_version)
    client = MlflowClient()
    client.set_registered_model_alias(_MODEL_NAME, _ALIAS, version)
    logger.info("Seeded %s v%d and set @%s", _MODEL_NAME, version, _ALIAS)
    return version


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed()


if __name__ == "__main__":
    main()
