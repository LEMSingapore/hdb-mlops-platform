# Multi-stage build for the FastAPI prediction service.
#
# The builder stage installs runtime dependencies into an isolated virtualenv
# using uv. The runtime stage copies only that virtualenv plus the application
# source, so the final image carries no build tools, no compilers, and no
# dev/test dependencies. See docs/adr/0005-multi-stage-dockerfile-with-uv.md.

FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Only pyproject.toml is needed to resolve and install runtime dependencies.
# The application source is deliberately not copied here: dependencies change
# rarely while source changes often, so isolating this layer keeps the cache
# warm across most rebuilds. Installing "." with no source present builds a
# metadata-only wheel — the [project.dependencies] are installed, the package
# itself contributes nothing. The source arrives in the runtime stage instead.
COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY src/ /app/src/

# Defaults to the containerised MLflow tracking server reachable on the compose
# network. FastAPI fetches model artifacts over HTTP through the server's
# artifact proxy and never reads the SQLite file or a host path directly. Local
# standalone runs override this via shell env or .env. See
# docs/adr/0006-mlflow-tracking-server-as-compose-service.md.
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

# --app-dir puts /app/src on the import path so "serving.app" resolves without
# editing PYTHONPATH or installing the package into the runtime image.
CMD ["python", "-m", "uvicorn", "serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/src"]
