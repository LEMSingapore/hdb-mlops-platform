"""Data access configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).parent.parent.parent


class DataConfig(BaseSettings):
    """Configuration for the SQLite data layer.

    db_path defaults to data/hdb.db relative to the repo root and is
    overridable via the HDB_DB_PATH environment variable.
    """

    db_path: Path = _REPO_ROOT / "data" / "hdb.db"

    model_config = {"env_prefix": "HDB_"}
