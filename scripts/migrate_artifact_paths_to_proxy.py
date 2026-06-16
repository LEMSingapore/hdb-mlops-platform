"""Rewrite absolute MLflow artifact paths to ``mlflow-artifacts:`` proxy URIs.

Run once before bringing the docker-compose stack up against an ``mlflow.db``
created by local-process MLflow:

    python scripts/migrate_artifact_paths_to_proxy.py

MLflow 3 records artifact locations as absolute host filesystem paths in four
tables: ``experiments.artifact_location``, ``runs.artifact_uri``,
``model_versions.storage_location`` and ``logged_models.artifact_location``.
When the containerised tracking server hands one of those raw paths back to a
client, the client opens a *local* artifact repository against a path that does
not exist inside its container — the failure surfaces as
``MlflowException: No such artifact: ''``. ``--serve-artifacts`` only proxies
artifacts whose stored location already uses the ``mlflow-artifacts:`` scheme,
so it does not rescue absolute paths on its own.

This migration rewrites the prefix up to and including the ``mlruns`` directory
to ``mlflow-artifacts:``, leaving the path that follows intact. With the
tracking server started as ``--artifacts-destination /mlflow/artifacts`` and
``./mlruns`` mounted at ``/mlflow/artifacts``, a stored
``mlflow-artifacts:/1/models/m-abc/artifacts`` resolves to
``/mlflow/artifacts/1/models/m-abc/artifacts`` and is served over HTTP. The
client then uses the proxy artifact repository and never touches a host path.

The script is idempotent: values already using the ``mlflow-artifacts:`` scheme
are left untouched, so re-running it is a no-op. See
docs/adr/0006-mlflow-tracking-server-as-compose-service.md for the rationale.
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_DB_PATH = _REPO_ROOT / "mlflow.db"

# The artifact root directory whose contents the tracking server proxies. Every
# absolute path below this directory maps to a path under --artifacts-destination.
_ARTIFACT_ROOT_DIRNAME = "mlruns"
_PROXY_SCHEME = "mlflow-artifacts:"

# (table, column) pairs that MLflow 3 populates with artifact locations.
_TARGETS: list[tuple[str, str]] = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("model_versions", "storage_location"),
    ("logged_models", "artifact_location"),
]


def _to_proxy_uri(value: str) -> str | None:
    """Return the proxy-URI form of an absolute artifact path, or None to skip.

    Returns None when the value is already a proxy URI or does not contain the
    artifact root directory, so callers can distinguish "rewritten" from
    "left alone".
    """
    if value.startswith(_PROXY_SCHEME):
        return None
    marker = f"/{_ARTIFACT_ROOT_DIRNAME}"
    index = value.rfind(marker)
    if index == -1:
        return None
    suffix = value[index + len(marker) :]
    return f"{_PROXY_SCHEME}{suffix}"


def migrate(db_path: Path) -> int:
    """Rewrite absolute artifact paths in the database to proxy URIs.

    Args:
        db_path: Path to the MLflow SQLite backend store.

    Returns:
        The number of rows rewritten across all target tables.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"No MLflow database at {db_path}")

    rewritten = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for table, column in _TARGETS:
            values = {
                row["value"]
                for row in conn.execute(
                    f"SELECT DISTINCT {column} AS value FROM {table} WHERE {column} IS NOT NULL"
                ).fetchall()
            }
            for value in values:
                proxy = _to_proxy_uri(value)
                if proxy is None:
                    continue
                # Match on the stored value rather than rowid: some MLflow tables
                # are declared WITHOUT ROWID, so rowid is not addressable.
                cursor = conn.execute(
                    f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                    (proxy, value),
                )
                logger.info(
                    "%s.%s (%d row(s)): %s -> %s",
                    table,
                    column,
                    cursor.rowcount,
                    value,
                    proxy,
                )
                rewritten += cursor.rowcount
        conn.commit()

    logger.info("Rewrote %d artifact path(s) in %s", rewritten, db_path)
    return rewritten


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "db_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help=f"Path to the MLflow SQLite store (default: {_DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    try:
        migrate(args.db_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
