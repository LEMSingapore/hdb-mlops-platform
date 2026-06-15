# Phase 2 Design — Docker + CI

**Author:** Chang Chee Young
**Status:** Planning complete; sessions pending
**Decisions locked:** 12 June 2026

---

## Context

Phase 1 through Phase 1.6c shipped a working MLOps platform: FastAPI serving with atomic model+explainer reload, MLflow alias-based promotion, MCP server exposing 5 tools, LangGraph orchestration in Streamlit. Everything runs as local processes on the developer's Mac.

Phase 2 containerises the platform and adds CI. It closes [#10](https://github.com/LEMSingapore/hdb-mlops-platform/issues/10) (pin serving environment to match training) and unlocks the foundation for any future deployment work ([#32](https://github.com/LEMSingapore/hdb-mlops-platform/issues/32)).

Phase 2 deliberately does *not* deploy anywhere, add authentication, add rate limiting, or migrate the MLflow backend to Postgres. Those are tracked in [#34](https://github.com/LEMSingapore/hdb-mlops-platform/issues/34) as deliberate scope omissions for any future Phase 6 deployment.

---

## Decisions

### 1. Service layout

- **FastAPI prediction service** — containerised
- **MLflow tracking server** — containerised, exposes port 5000 over HTTP
- **MCP server** — deferred (stays as Claude Desktop stdio subprocess; remote MCP is #32 Path A)
- **Streamlit chat app** — deferred (public deployment is #32 Path C)
- **Streamlit form app** — not containerised (diagnostic only)

Two services in docker-compose: `fastapi` and `mlflow`. FastAPI talks to MLflow over HTTP at `http://mlflow:5000`, not by reading the SQLite file directly. This means even though the metadata backend is SQLite, FastAPI accesses it through MLflow's API surface — consistent with how a real deployment would work.

### 2. MLflow backend

- **Metadata: SQLite via volume mount.** `sqlite:////mlflow/mlflow.db` inside the mlflow container, persisted in a docker volume.
- **Artifacts: filesystem via volume mount.** `/mlflow/artifacts` inside the mlflow container, persisted in the same or a sibling docker volume.

Trade-off documented: SQLite + concurrent writes is risky if scaling FastAPI past one replica. We're not, so it's fine. Postgres swap is the canonical upgrade if scaling.

No data migration: the existing `mlflow.db` and `mlruns/` get mounted into the container at startup. `@champion` v7 stays as `@champion`.

### 3. Image build

- **Multi-stage Dockerfile.** Builder stage installs deps into a venv; runtime stage copies the venv and source.
- **Base image: `python:3.12-slim`** for both stages.
- **uv for dependency install** in the builder stage. Matches local dev workflow.

Rough shape:

```dockerfile
FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

FROM python:3.12-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. CI scope

GitHub Actions on every PR, runs the following steps:

1. `pre-commit run --all-files` (~30s) — ruff, ruff-format, mypy
2. `python -m pytest` (~2min) — all 242 tests
3. `docker build` (~3-4min first run, ~30s cached) — verifies Dockerfile builds
4. `docker compose up -d` + healthcheck wait (~2-3min) — compose smoke
5. Integration smoke: `curl /predict` against the running container, assert response shape (~30s)

**Cache strategy:** `actions/setup-buildx-action` + `cache-from`/`cache-to` for Docker layers. Without caching, every CI run rebuilds the entire image; with caching, only changed layers rebuild.

**Not included:** image push to GHCR (deferred to Phase 6 when there's a deploy target consuming it).

### 5. Issue closure scope

| Issue | Phase 2 disposition |
|---|---|
| #10 — Pin serving environment | **Closes** (Dockerfile + locked deps pin Python + library versions identically for training and serving) |
| #11 — /predict 500 → 503 on loader failure | **Defer** (small serving-layer change, post-Phase-2 chore PR) |
| #29 — form_app env_file | **Defer** (5-minute fix, post-Phase-2 chore PR) |
| #30 — time-feature dominance | **Defer** (Phase 5 model work) |
| #31 — multi-turn awareness | **Defer** (future enhancement) |
| #32 — public access | **Enables but doesn't close** — Phase 2 ships containerised platform; deployment is Phase 6 |
| #34 — production readiness checklist | **Filed during Phase 2 planning** as the explicit record of deliberate scope omissions |

### 6. Session split

Three sessions, three PRs.

**Session A — Dockerfile + standalone local run.** ~3h.

Multi-stage Dockerfile only. `docker build && docker run` produces a running FastAPI container that talks to the *host's* `mlflow.db` and `mlruns/` via bind mounts. No compose yet. Sets up the foundation.

Files: `Dockerfile`, `.dockerignore`, README update. No compose, no CI.

**Session B — docker-compose with MLflow service.** ~3h.

`docker-compose.yml` with two services: `fastapi` and `mlflow`. FastAPI talks to MLflow over HTTP. Both services get healthchecks. Closes #10.

Files: `docker-compose.yml`, possibly a separate `Dockerfile.mlflow`, healthcheck scripts, README updates (a "Try it via Docker Compose" section).

**Session C — GitHub Actions CI.** ~3h.

All CI steps from Decision 4 + buildx caching. README gets a status badge.

Files: `.github/workflows/ci.yml`, README badge addition.

---

## Constraints (apply to all three sessions)

- Pre-commit must pass (ruff, ruff-format, mypy)
- All 242 existing tests must pass
- v7 stays as `@champion`; no retrain needed
- No new Python dependencies (Docker tooling, not Python)

---

## Success criteria

After Session C ships:

1. `docker compose up` brings the platform up locally with FastAPI + MLflow tracking on the same host
2. `curl localhost:8000/predict` returns a Tampines price from inside the FastAPI container
3. `localhost:5000` serves the MLflow UI from inside the MLflow container
4. GitHub Actions runs cleanly on every PR, including a green status badge on the README
5. Anyone with Docker installed can reproduce the deployment with two commands: `git clone` + `docker compose up`

Crucially: Phase 2 does *not* claim to ship a *deployment*. It ships the *artefacts that enable* deployment. Phase 6 is the deployment.

---

## What this design doc isn't

- **Not a hardening checklist.** Auth, rate limiting, TLS, secret management, monitoring, observability — all out of scope. See #34.
- **Not a deployment plan.** See #32 and Phase 6.
- **Not a model improvement.** See #30 and Phase 5.
