# ADR 0005: Multi-stage Dockerfile with uv

**Status:** Accepted
**Date:** 2026-06-16

## Context

Phase 1 through 1.6c ran the FastAPI prediction service as a local process on my Mac. Phase 2 containerises it. Issue #10 asks me to pin the serving environment so the libraries that serve a model match the ones that trained it — a containerised image is how I close that gap.

Session A is the foundation: a single image that builds, runs, and serves a prediction. It deliberately stops short of docker-compose and a containerised MLflow tracking server (Session B) and CI (Session C). The container in this session talks to the host's existing `mlflow.db` and `mlruns/` through bind mounts, because standing up a real tracking service is Session B's job.

Two forces shape the image. First, it ends up in CI on every PR, so build time and layer caching matter. Second, sklearn, SHAP, and NumPy are heavy native dependencies, so the choice of base image and how dependencies are installed determines whether the final image is 600MB or 1.5GB.

## Decision

I build the serving image with a multi-stage Dockerfile on `python:3.12-slim`, installing runtime dependencies with `uv` in the builder stage and copying only the resulting virtualenv plus the application source into the runtime stage.

The runtime stage sets `MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db` as a default and runs `python -m uvicorn serving.app:app --app-dir /app/src`. The `--app-dir` flag puts the source on the import path without installing the package or editing `PYTHONPATH`.

## Consequences

The runtime image carries no compilers, no `uv`, and none of the dev or test dependencies — only the virtualenv and the source under `/app/src`. That keeps the final image in the 500-800MB range despite sklearn and SHAP, versus the 1.5GB+ a single-stage build with build tooling retained would produce.

Splitting dependency install from the source copy is a caching decision. Dependencies live in their own layer keyed on `pyproject.toml`; they only reinstall when that file changes. Editing `src/` invalidates only the cheap `COPY src/` layer, so the expensive sklearn/SHAP install is reused across the rebuilds that dominate day-to-day work and CI.

The trade-off I am accepting in Session A: MLflow stores artifact locations as absolute filesystem paths in `mlflow.db`. Bind-mounting the host's `mlruns/` only resolves if the directory is mounted at the same absolute path the database recorded. That is fragile and host-specific — it is exactly the friction Session B removes by running an MLflow tracking server, where FastAPI fetches artifacts over HTTP through the server's artifact proxy and never touches a host path. Session A proves the service containerises; Session B makes the data plane portable.

Session B has since landed and done exactly that — see [ADR 0006](0006-mlflow-tracking-server-as-compose-service.md). It also established that `--serve-artifacts` does not rewrite absolute paths already in the database, so the registry needed a one-time migration to `mlflow-artifacts:` proxy URIs on top of the tracking-server topology.

`MLFLOW_TRACKING_URI` now defaults to `http://mlflow:5000` so the image runs correctly under compose out of the box; standalone local runs override it via shell env or `.env`, with no image rebuild — the registry endpoint is configuration, not a baked-in constant.

## Alternatives considered

**Single-stage build.** Simpler Dockerfile, one `FROM`. Rejected because the build tooling and uv stay in the final image, and without a clean dependency/source split the layer cache is far less effective. The size and CI-time cost outweigh the simplicity.

**pip instead of uv.** pip is already in the base image, so it needs no bootstrap layer. I chose uv for consistency with the local development workflow established in Phase 1.6a — the same resolver and lockfile semantics in the container as on my machine — and for its faster cold installs, which matter on CI's uncached first build. The one extra `pip install uv` layer in the builder stage is a price paid in a stage that never ships.

**A slimmer base — `alpine` or `distroless`.** Alpine uses musl libc, which forces source builds of NumPy, sklearn, and SHAP instead of the prebuilt manylinux wheels; the build gets slower and more fragile. Distroless removes the shell I want for debugging in this phase. `python:3.12-slim` is the sweet spot — Debian glibc means binary wheels install for both ARM64 and x86_64, while slim already strips the bulk of a full Debian image.

**Installing the package into the runtime image** (`uv pip install .` with source present, then import normally). Rejected as unnecessary: the service is run directly from source, not imported as a library, so `--app-dir /app/src` is enough. Skipping the install avoids a build backend round-trip in the runtime stage and keeps the source copy as the only thing that changes between most rebuilds.
