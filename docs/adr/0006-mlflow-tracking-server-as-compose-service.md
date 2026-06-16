# ADR 0006: MLflow tracking server as a docker-compose service

**Status:** Accepted
**Date:** 2026-06-16

## Context

Session A ([ADR 0005](0005-multi-stage-dockerfile-with-uv.md)) containerised the FastAPI prediction service as a single image that read the registry by bind-mounting the host's `mlflow.db` and `mlruns/` directly. Smoke testing exposed why that approach is a dead end. MLflow 3 records artifact locations as absolute host filesystem paths, and it does so in four tables, not one: `experiments.artifact_location`, `runs.artifact_uri`, `model_versions.storage_location`, and `logged_models.artifact_location`. When `mlflow.pyfunc.load_model("models:/hdb-predictor@champion")` runs inside a container, the client resolves the alias to a logged-model URI, reads `logged_models.artifact_location` — an absolute path like `/Users/cheeyoungchang/Projects/hdb-mlops-platform/mlruns/1/models/...` — and opens a *local* artifact repository against it. That path does not exist inside the container, so the load fails with `MlflowException: No such artifact: ''`. Bind-mounting `mlruns/` at the same absolute path the host uses papers over it, but only on my machine, and only until the path changes.

The fix is to stop treating the registry as a file the serving container reads and start treating it as a service the serving container calls. That is the canonical MLflow deployment topology, and Session B adopts it.

## Decision

I run MLflow as its own docker-compose service — `ghcr.io/mlflow/mlflow:v3.13.0`, pinned — exposing the tracking and registry API on port 5000. FastAPI reaches it over HTTP at `http://mlflow:5000` via `MLFLOW_TRACKING_URI`; it never reads `mlflow.db` or a host path again. The tracking server runs with `--serve-artifacts` and `--artifacts-destination /mlflow/artifacts`, so artifacts flow through the same HTTP API as metadata, served from the server's own mounted filesystem. FastAPI requests `models:/hdb-predictor@champion`, the server resolves it internally and streams the model back over HTTP.

SQLite remains the metadata backend, mounted at `sqlite:////mlflow/mlflow.db`; `mlruns/` is mounted at `/mlflow/artifacts`. `@champion` stays on version 1 — no retrain, no data migration.

## Consequences

FastAPI's data plane is now portable. The registry endpoint is configuration, not a baked-in path, so the same image runs unchanged against a local compose stack, a CI runner, or a future k3s deployment. This is what closes [#10](https://github.com/LEMSingapore/hdb-mlops-platform/issues/10): the tracking server is pinned to `v3.13.0` and FastAPI's image resolves `mlflow==3.13.0` through uv, so the library that serves a model and the library that trained it are byte-for-byte the same version on both sides of the wire.

Three things this session surfaced that `--serve-artifacts` alone does not solve, and that the design note did not anticipate:

**`--serve-artifacts` does not rewrite absolute paths already in the database.** The proxy only serves locations stored under the `mlflow-artifacts:` scheme. A pre-existing absolute path is handed to the client verbatim, which then falls back to the local artifact repository and fails exactly as before. The registry created by local-process MLflow therefore needs a one-time migration: `scripts/migrate_artifact_paths_to_proxy.py` rewrites the prefix up to and including `mlruns` to `mlflow-artifacts:` across all four tables, so `…/mlruns/1/models/m-abc/artifacts` becomes `mlflow-artifacts:/1/models/m-abc/artifacts`. With `mlruns/` mounted at `--artifacts-destination`, that URI resolves to the right file inside the server and streams over HTTP. The migration is idempotent and runs against the gitignored local `mlflow.db`; it is the documented prerequisite to bringing the stack up against a registry built before this ADR.

**MLflow 3 rejects cross-service requests by default.** The server ships DNS-rebinding protection that validates the `Host` header against an allowlist defaulting to localhost only. FastAPI connects as `http://mlflow:5000`, so the request carries `Host: mlflow:5000` and is refused with a 403. I keep the protection on and set `MLFLOW_SERVER_ALLOWED_HOSTS` to an explicit allowlist — `mlflow:5000` for the service-to-service call plus `localhost:5000` and `127.0.0.1:5000` for browser and host access — rather than disabling it with a wildcard. Setting the variable replaces the default list, which is why localhost has to be restated.

**Neither image ships `curl`.** The compose healthchecks therefore probe with the Python interpreter both images already carry, via `urllib.request`, instead of the `curl` the design sketch assumed.

The trade-off I am accepting: SQLite as the metadata backend is safe only because this stack runs a single FastAPI replica and a single tracking server, so there are no concurrent writers. Horizontal scaling would force the canonical Postgres swap — the design doc records that path and Phase 6 owns it. The tracking server is also now a hard dependency of the serving container's startup; FastAPI fails fast if it cannot reach `http://mlflow:5000`, which is the correct behaviour but does make the tracking server a single point of failure. Production hardening of that SPOF — replicas, readiness probes, a managed backend — is Phase 6 work, not this PR's.

## Alternatives considered

**Bind-mount `mlruns/` at the host's absolute path inside the container.** This is the Session A stopgap. It works on exactly one machine, breaks the moment the repository moves, and cannot survive CI or a remote deployment where the host path is meaningless. Rejected as fundamentally non-portable — it is the problem this ADR exists to remove.

**Rewrite stored paths to be relative and mount the artifact root identically in both containers, without `--serve-artifacts`.** This keeps clients reading artifacts off a shared filesystem instead of over HTTP. It would work for a co-located compose stack, but it re-couples the serving container to a filesystem layout and does not generalise to a deployment where FastAPI and MLflow do not share a volume. Serving artifacts over the API is the topology a real deployment uses, so I matched it now rather than carrying a shortcut into Phase 6.

**Disable DNS-rebinding protection with `MLFLOW_SERVER_ALLOWED_HOSTS=*`.** Simpler — one token, no need to restate localhost. Rejected because an explicit allowlist is the same amount of configuration to reason about and keeps a real security control switched on. A wildcard would also read, to anyone scanning the compose file, as not having understood why the 403 happened.

**Migrate the metadata backend to Postgres now.** The canonical production backend, and the obvious answer if I were scaling FastAPI past one replica. I am not, and the design doc explicitly defers it. Standing up a Postgres service for a single-writer registry is cost without benefit at this phase; it belongs to Phase 6 alongside the deployment that motivates it.
