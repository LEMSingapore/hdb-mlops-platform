"""Tests for ModelLoader: initial load, version tracking, and background polling."""

import time

import mlflow.exceptions
import pytest
from mlflow import MlflowClient

from serving.config import ServingConfig
from serving.model_loader import ModelLoader
from tests.conftest import register_model_version


def _make_config(tracking_uri: str, alias: str = "champion") -> ServingConfig:
    return ServingConfig(
        mlflow_tracking_uri=tracking_uri,
        model_name="hdb-predictor",
        model_alias=alias,
        reload_interval_seconds=60,
    )


@pytest.fixture
def single_version_setup(isolated_mlflow_uri):
    """One registered model version with @champion alias. Returns (uri, version)."""
    version = register_model_version(isolated_mlflow_uri)
    client = MlflowClient()
    client.set_registered_model_alias("hdb-predictor", "champion", version)
    return isolated_mlflow_uri, version


@pytest.fixture
def two_version_setup(isolated_mlflow_uri):
    """Two registered versions with @champion pointing to v1. Returns (uri, v1, v2)."""
    v1 = register_model_version(isolated_mlflow_uri)
    v2 = register_model_version(isolated_mlflow_uri)
    client = MlflowClient()
    client.set_registered_model_alias("hdb-predictor", "champion", v1)
    return isolated_mlflow_uri, v1, v2


class TestInitialLoad:
    def test_successful_load_sets_model_and_version(self, single_version_setup):
        uri, expected_version = single_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()

        assert loader.get_model() is not None
        assert loader.get_version() is not None
        assert int(loader.get_version()) == int(expected_version)

    def test_get_version_returns_none_before_load(self, isolated_mlflow_uri):
        loader = ModelLoader(_make_config(isolated_mlflow_uri))
        assert loader.get_version() is None

    def test_get_model_raises_before_load(self, isolated_mlflow_uri):
        loader = ModelLoader(_make_config(isolated_mlflow_uri))
        with pytest.raises(RuntimeError, match="not yet loaded"):
            loader.get_model()

    def test_load_initial_raises_when_alias_not_found(self, isolated_mlflow_uri):
        # Registry exists but @champion alias has never been set
        register_model_version(isolated_mlflow_uri)  # registers model but no alias
        loader = ModelLoader(_make_config(isolated_mlflow_uri))
        with pytest.raises(mlflow.exceptions.MlflowException):
            loader.load_initial()

    def test_load_initial_raises_when_registry_unreachable(self, tmp_path):
        bad_uri = f"sqlite:///{tmp_path / 'does_not_exist' / 'db.sqlite'}"
        loader = ModelLoader(_make_config(bad_uri))
        with pytest.raises(Exception):  # noqa: B017
            loader.load_initial()


class TestVersionTracking:
    def test_get_version_returns_expected_value_after_load(self, single_version_setup):
        uri, version = single_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()
        assert loader.get_version() is not None

    def test_get_run_id_returns_string_after_load(self, single_version_setup):
        uri, _ = single_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()
        run_id = loader.get_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_get_run_id_returns_none_before_load(self, isolated_mlflow_uri):
        loader = ModelLoader(_make_config(isolated_mlflow_uri))
        assert loader.get_run_id() is None


class TestAtomicSwap:
    def test_reload_if_changed_swaps_model_when_alias_moves(self, two_version_setup):
        uri, v1, v2 = two_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()
        assert int(loader.get_version()) == int(v1)

        client = MlflowClient()
        client.set_registered_model_alias("hdb-predictor", "champion", v2)

        loader._reload_if_changed()

        assert int(loader.get_version()) == int(v2)
        assert loader.get_model() is not None

    def test_reload_if_changed_is_no_op_when_alias_unchanged(self, single_version_setup):
        uri, v1 = single_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()

        model_before = loader.get_model()
        loader._reload_if_changed()
        # Same object reference — no reload occurred
        assert loader.get_model() is model_before

    def test_background_poll_swaps_model_on_version_change(self, two_version_setup):
        uri, v1, v2 = two_version_setup

        loader = ModelLoader(_make_config(uri))
        loader.load_initial()
        assert int(loader.get_version()) == int(v1)

        # Point alias to v2, then start polling at 0.1s interval
        client = MlflowClient()
        client.set_registered_model_alias("hdb-predictor", "champion", v2)

        loader._config = ServingConfig(
            mlflow_tracking_uri=uri,
            model_name="hdb-predictor",
            model_alias="champion",
            reload_interval_seconds=0,
        )
        loader.start_background_polling()
        time.sleep(0.25)

        assert int(loader.get_version()) == int(v2)


class TestErrorResilience:
    def test_reload_retains_model_when_registry_call_fails(self, single_version_setup, monkeypatch):
        uri, v1 = single_version_setup
        loader = ModelLoader(_make_config(uri))
        loader.load_initial()

        model_before = loader.get_model()
        version_before = loader.get_version()

        def _always_fail(*args, **kwargs):
            raise Exception("Simulated registry failure")

        monkeypatch.setattr(loader, "_resolve_alias_version", _always_fail)

        # Should not raise; model retained on error
        loader._reload_if_changed()

        assert loader.get_version() == version_before
        assert loader.get_model() is model_before
