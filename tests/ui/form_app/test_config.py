"""Tests for UIConfig environment variable loading."""

from ui.form_app.config import UIConfig


def test_defaults_without_env_vars():
    config = UIConfig()
    assert config.api_base_url == "http://127.0.0.1:8000"
    assert config.request_timeout_seconds == 10


def test_api_base_url_overridden_by_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://prod-api:9000")
    config = UIConfig()
    assert config.api_base_url == "http://prod-api:9000"


def test_request_timeout_overridden_by_env(monkeypatch):
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "30")
    config = UIConfig()
    assert config.request_timeout_seconds == 30
