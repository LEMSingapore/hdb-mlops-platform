"""Configuration for the MCP server, loaded from environment variables.

Defaults mirror the serving and data layers so the MCP server talks to the
same MLflow registry and SQLite database without extra configuration:

- ``mlflow_tracking_uri`` matches :class:`serving.config.ServingConfig`.
- ``db_path`` matches :class:`data.config.DataConfig` (env var ``HDB_DB_PATH``).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class MCPConfig(BaseSettings):
    """Settings for the HDB MCP server.

    All fields are overridable via environment variables. ``db_path`` reads from
    ``HDB_DB_PATH`` to stay consistent with the data layer's env contract.
    """

    model_config = SettingsConfigDict(protected_namespaces=())

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    db_path: Path = Field(
        default=_REPO_ROOT / "data" / "hdb.db",
        validation_alias="HDB_DB_PATH",
    )
    model_name: str = "hdb-predictor"
    model_alias: str = "champion"
