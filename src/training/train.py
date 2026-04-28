"""Training script for HDB resale price predictor.

Trains a GradientBoostingRegressor inside a sklearn Pipeline that bundles
the ColumnTransformer preprocessor with the model. Logs params, metrics,
the model artifact, its signature, and an input example to MLflow, then
registers the model in the MLflow Model Registry.

Promotion to aliases (@champion, @challenger) is intentional and explicit:
pass --promote champion (or --promote challenger) to set the alias after a
successful run. Never use transition_model_version_stage().

Usage:
    python -m training.train --promote champion
    python -m training.train --n-estimators 500 --learning-rate 0.05
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from training.config import TrainingConfig

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = ["floor_area_sqm", "lease_commence_date", "float_time_series"]
CATEGORICAL_FEATURES: list[str] = ["town", "storey_range", "flat_info"]
TARGET: str = "resale_price"

# Ordered to match legacy dataset filenames exactly.
_RAW_FILES: list[str] = [
    "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
    "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
    "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
    "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
    "Resale flat prices based on registration date from Jan-2017 onwards.csv",
]

# Aliases that a caller may promote a trained model to.
VALID_ALIASES: frozenset[str] = frozenset({"champion", "challenger", "shadow"})


def load_and_prepare_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load all raw CSV files and return feature DataFrame and target Series.

    Normalises casing differences across datasets (the 1990-1999 file uses
    uppercase for flat_model; later files use title case). Derives two
    engineered features: flat_info (flat_type + flat_model) and
    float_time_series (month as a fractional year for temporal ordering).

    Args:
        data_dir: Directory containing the five raw CSV files.

    Returns:
        X: DataFrame with columns matching NUMERIC_FEATURES + CATEGORICAL_FEATURES.
        y: Series of resale prices.
    """
    frames: list[pd.DataFrame] = []
    for filename in _RAW_FILES:
        path = data_dir / filename
        df = pd.read_csv(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Normalise string columns so categories are consistent across the five
    # source files, which were published at different times with different
    # capitalisation conventions.
    merged["flat_type"] = merged["flat_type"].str.strip().str.upper()
    merged["flat_model"] = merged["flat_model"].str.strip().str.title()
    merged["town"] = merged["town"].str.strip().str.upper()
    merged["storey_range"] = merged["storey_range"].str.strip().str.upper()

    merged["flat_info"] = merged["flat_type"] + " " + merged["flat_model"]

    month_dt = pd.to_datetime(merged["month"], format="%Y-%m")
    merged["float_time_series"] = month_dt.dt.year + (month_dt.dt.month - 1) / 12.0

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = merged[feature_cols].copy()
    y = merged[TARGET].copy()

    return X, y


def build_pipeline(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    max_features: float,
) -> Pipeline:
    """Build a sklearn Pipeline combining preprocessing and the GBR regressor.

    Bundling the ColumnTransformer inside the Pipeline means MLflow logs a
    self-contained model artifact: the serving layer calls pipeline.predict()
    on raw feature DataFrames with no separate preprocessing step.

    OneHotEncoder uses handle_unknown="ignore" so unseen categories at
    inference time (e.g. a new town) get all-zero encoding rather than an
    error.

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage factor applied to each tree's contribution.
        max_depth: Maximum depth of individual regression trees.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        max_features: Fraction of features to consider at each split.

    Returns:
        Unfitted sklearn Pipeline ready for .fit().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    loss="squared_error",
                    random_state=7,
                ),
            ),
        ]
    )


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary with keys "rmse", "mae", "r2".
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(args: argparse.Namespace, config: TrainingConfig) -> str:
    """Execute one training run and return the MLflow run ID.

    Logs params, train/test metrics, and the trained pipeline as a registered
    model artifact. Optionally promotes the resulting model version to an alias.

    Args:
        args: Parsed CLI arguments.
        config: Training configuration (tracking URI, experiment name, etc.).

    Returns:
        The MLflow run ID for the completed run.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    data_dir = Path(args.data_dir)
    logger.info("Loading data from %s", data_dir)
    X, y = load_and_prepare_data(data_dir)
    logger.info("Dataset loaded: %d rows", len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=7
    )
    logger.info("Train: %d rows  Test: %d rows", len(X_train), len(X_test))

    pipeline = build_pipeline(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    )

    run_name = f"gbr-n{args.n_estimators}-lr{args.learning_rate}-d{args.max_depth}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_features": args.max_features,
                "test_size": args.test_size,
                "random_state": 7,
            }
        )

        logger.info("Fitting pipeline (n_estimators=%d)...", args.n_estimators)
        pipeline.fit(X_train, y_train)

        train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
        test_metrics = compute_metrics(y_test, pipeline.predict(X_test))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        logger.info(
            "Test  RMSE=SGD %.0f  MAE=SGD %.0f  R²=%.4f",
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )

        # Infer the signature from the raw feature DataFrame so the serving
        # layer knows the expected input schema without inspecting the pipeline.
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.head(5)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=config.model_registry_name,
        )

        version = model_info.registered_model_version
        logger.info(
            "Registered %s version %s (run %s)",
            config.model_registry_name,
            version,
            run.info.run_id,
        )

        if args.promote:
            client = MlflowClient()
            client.set_registered_model_alias(config.model_registry_name, args.promote, version)
            logger.info(
                "Set @%s → version %s",
                args.promote,
                version,
            )

        return run.info.run_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train HDB resale price predictor and log to MLflow."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
        metavar="N",
        help="Number of boosting stages (default: 1000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        metavar="LR",
        help="GBR learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        metavar="D",
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=9,
        metavar="M",
        help="Minimum samples per leaf (default: 9)",
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=0.1,
        metavar="F",
        help="Fraction of features considered per split (default: 0.1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="T",
        help="Fraction of data held out for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--promote",
        choices=sorted(VALID_ALIASES),
        default=None,
        metavar="ALIAS",
        help=(
            "Alias to assign to the trained model version after registration. "
            f"One of: {', '.join(sorted(VALID_ALIASES))}. "
            "Omit to register without promotion."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    config = TrainingConfig()
    train(args, config)


if __name__ == "__main__":
    main()
