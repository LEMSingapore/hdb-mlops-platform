"""Background model loader with atomic swap for zero-downtime model updates.

The serving app calls load_initial() at startup to fetch the champion model
synchronously, then start_background_polling() to spawn a daemon thread that
checks the registry every RELOAD_INTERVAL_SECONDS. When the @champion alias
points to a new version, the daemon loads it and swaps the in-memory reference
under a threading.Lock. In-flight predictions hold a reference to the old model
object and complete safely; the old object is GC'd once no references remain.

The SHAP TreeExplainer is initialised alongside the model and stored in an
ExplainerBundle. If explainer initialisation fails during a background reload,
the existing model+explainer pair is retained rather than leaving the loader
in an inconsistent state.
"""

import logging
import threading
import time
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
import shap
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel

from serving.config import ServingConfig

logger = logging.getLogger(__name__)


@dataclass
class ExplainerBundle:
    """SHAP TreeExplainer paired with the preprocessing artifacts needed at request time.

    Stored together so that the preprocessor and explainer are always consistent
    with each other — both are derived from the same loaded model version.
    """

    explainer: shap.TreeExplainer
    preprocessor: object  # sklearn ColumnTransformer; typed as object to avoid hard sklearn import
    feature_names: list[str]


class ModelLoader:
    """Loads the champion model from MLflow registry and polls for version changes.

    Thread safety: _model, _current_version, and _explainer are only mutated
    inside self._lock. Model and explainer loading happen outside the lock —
    both are slow and blocking predictions during them would be unacceptable.
    The explainer and model are always swapped together; a failed explainer
    init prevents the model swap so the pair stays consistent.
    """

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._model: PyFuncModel | None = None
        self._current_version: int | None = None
        self._explainer: ExplainerBundle | None = None
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self._client = MlflowClient()

    def load_initial(self) -> None:
        """Synchronously load the champion model and explainer at application startup.

        Intentionally raises on failure so the app fails fast rather than
        starting in a state where /predict immediately returns 500.

        Raises:
            Exception: Any MLflow error from alias resolution, model download,
                       or SHAP TreeExplainer initialisation.
        """
        logger.info(
            "Loading %s@%s from %s",
            self._config.model_name,
            self._config.model_alias,
            self._config.mlflow_tracking_uri,
        )
        version = self._resolve_alias_version()
        model_uri = f"models:/{self._config.model_name}@{self._config.model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        bundle = self._build_explainer_bundle(model)
        with self._lock:
            self._model = model
            self._current_version = version
            self._explainer = bundle
        logger.info(
            "Loaded %s version %s with SHAP TreeExplainer (%d features)",
            self._config.model_name,
            version,
            len(bundle.feature_names),
        )

    def get_model(self) -> PyFuncModel:
        """Return the currently loaded model.

        Returns:
            The active PyFuncModel instance.

        Raises:
            RuntimeError: If called before load_initial() has succeeded.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not yet loaded; startup may have failed.")
            return self._model

    def get_version(self) -> int | None:
        """Return the version string of the currently loaded model."""
        with self._lock:
            return self._current_version

    def get_explainer(self) -> ExplainerBundle | None:
        """Return the SHAP ExplainerBundle for the currently loaded model version."""
        with self._lock:
            return self._explainer

    def get_run_id(self) -> str | None:
        """Return the MLflow run ID associated with the currently loaded version.

        Returns None if the version is unknown or the registry call fails.
        """
        version = self.get_version()
        if version is None:
            return None
        try:
            mv = self._client.get_model_version(self._config.model_name, str(version))
            return mv.run_id
        except Exception:
            logger.exception("Failed to fetch run ID for version %s", version)
            return None

    def start_background_polling(self) -> None:
        """Spawn a daemon thread that polls the registry on RELOAD_INTERVAL_SECONDS.

        The thread is a daemon so it does not prevent interpreter shutdown.
        """
        thread = threading.Thread(
            target=self._poll_loop,
            name="model-reload-poller",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Background polling started (interval=%ds)",
            self._config.reload_interval_seconds,
        )

    def _build_explainer_bundle(self, model: PyFuncModel) -> ExplainerBundle:
        """Initialise a TreeExplainer from the sklearn pipeline inside the pyfunc model.

        Extracts the fitted preprocessor and the GradientBoostingRegressor from
        the pipeline, then wraps them into an ExplainerBundle alongside the
        post-transformation feature names.

        Raises:
            Exception: If the model is not a sklearn pipeline with the expected
                       named steps, or if shap.TreeExplainer raises.
        """
        sklearn_pipeline = model._model_impl.sklearn_model  # type: ignore[attr-defined]
        preprocessor = sklearn_pipeline.named_steps["preprocessor"]
        regressor = sklearn_pipeline.named_steps["regressor"]
        feature_names = list(preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(regressor)
        return ExplainerBundle(
            explainer=explainer,
            preprocessor=preprocessor,
            feature_names=feature_names,
        )

    def _resolve_alias_version(self) -> int:
        """Return the model version currently pointed to by the configured alias."""
        mv = self._client.get_model_version_by_alias(
            self._config.model_name, self._config.model_alias
        )
        return int(mv.version)

    def _reload_if_changed(self) -> None:
        """Check the registry and swap the in-memory model+explainer if the alias moved.

        If explainer initialisation fails for a new version, the existing
        model+explainer pair is retained and the error is logged. This avoids
        leaving the loader in a state where model and explainer are mismatched.
        """
        try:
            latest_version = self._resolve_alias_version()
            with self._lock:
                unchanged = latest_version == self._current_version
            if unchanged:
                return

            logger.info(
                "Alias @%s moved to version %s; loading...",
                self._config.model_alias,
                latest_version,
            )
            model_uri = f"models:/{self._config.model_name}@{self._config.model_alias}"
            new_model = mlflow.pyfunc.load_model(model_uri)

            try:
                new_bundle = self._build_explainer_bundle(new_model)
            except Exception:
                logger.exception(
                    "Explainer initialisation failed for version %s; "
                    "retaining current model+explainer",
                    latest_version,
                )
                return

            with self._lock:
                self._model = new_model
                self._current_version = latest_version
                self._explainer = new_bundle
            logger.info(
                "Model and explainer swapped: %s now at version %s",
                self._config.model_name,
                latest_version,
            )
        except Exception:
            logger.exception(
                "Background reload failed; continuing with version %s",
                self._current_version,
            )

    def _poll_loop(self) -> None:
        """Blocking loop executed inside the background daemon thread."""
        while True:
            time.sleep(self._config.reload_interval_seconds)
            self._reload_if_changed()
