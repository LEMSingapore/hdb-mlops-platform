"""Background model loader with atomic swap for zero-downtime model updates.

The serving app calls load_initial() at startup to fetch the champion model
synchronously, then start_background_polling() to spawn a daemon thread that
checks the registry every RELOAD_INTERVAL_SECONDS. When the @champion alias
points to a new version, the daemon loads it and swaps the in-memory reference
under a threading.Lock. In-flight predictions hold a reference to the old model
object and complete safely; the old object is GC'd once no references remain.
"""

import logging
import threading
import time

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from serving.config import ServingConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads the champion model from MLflow registry and polls for version changes.

    Thread safety: _model and _current_version are only mutated inside
    self._lock. The model load itself happens outside the lock — loading is
    slow and blocking predictions during it would be unacceptable. Python's
    reference counting guarantees the old model stays alive until all
    in-flight predictions that hold a reference to it finish.
    """

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._model: mlflow.pyfunc.PyFuncModel | None = None
        self._current_version: str | None = None
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self._client = MlflowClient()

    def load_initial(self) -> None:
        """Synchronously load the champion model at application startup.

        Intentionally raises on failure so the app fails fast rather than
        starting in a state where /predict immediately returns 500.

        Raises:
            Exception: Any MLflow error from alias resolution or model download.
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
        with self._lock:
            self._model = model
            self._current_version = version
        logger.info(
            "Loaded %s version %s",
            self._config.model_name,
            version,
        )

    def get_model(self) -> mlflow.pyfunc.PyFuncModel:
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

    def get_version(self) -> str | None:
        """Return the version string of the currently loaded model."""
        with self._lock:
            return self._current_version

    def get_run_id(self) -> str | None:
        """Return the MLflow run ID associated with the currently loaded version.

        Returns None if the version is unknown or the registry call fails.
        """
        version = self.get_version()
        if version is None:
            return None
        try:
            mv = self._client.get_model_version(self._config.model_name, version)
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

    def _resolve_alias_version(self) -> str:
        """Return the model version currently pointed to by the configured alias."""
        mv = self._client.get_model_version_by_alias(
            self._config.model_name, self._config.model_alias
        )
        return mv.version

    def _reload_if_changed(self) -> None:
        """Check the registry and swap the in-memory model if the alias moved."""
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
            with self._lock:
                self._model = new_model
                self._current_version = latest_version
            logger.info(
                "Model swapped: %s now at version %s",
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
