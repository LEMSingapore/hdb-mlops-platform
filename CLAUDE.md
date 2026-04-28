# CLAUDE.md

Context for Claude Code sessions on this repo. Read this before making changes.

---

## What this project is

HDB Resale Price Predictor upgraded from a Streamlit prototype into a production-grade MLOps platform. Solo build, portfolio piece for ML Engineering job applications in Singapore. Target audience for everything visible (README, ADRs, code) is a senior ML engineer reading the repo during a hiring loop.

The code quality bar is "an interviewer will read this." No hacks, no `# TODO: clean this up later`, no commented-out code. Naming and docstrings matter.

Full plan: `docs/build-plan.md`. Read it before working on any phase you haven't touched recently.

---

## Hard constraints

- **Open source only.** No proprietary services in the production stack. AKS is allowed for the recorded demo phase only — never as a runtime dependency.
- **Solo developer, ~13-15 hrs/week, 5-6 week budget.** Do not propose work that doesn't fit this envelope.
- **British English** in all prose (Singapore standard).
- **No emojis** in code, comments, commit messages, or docs.

---

## Stack (decided — do not re-litigate)

| Layer | Tool | Notes |
|---|---|---|
| Tracking + Registry | MLflow | Use **aliases**, never stages |
| Serving | FastAPI | Pydantic schemas, async where it matters |
| UI | Streamlit | Thin HTTP client over FastAPI |
| Data versioning | DVC | Remote = MinIO (same instance as MLflow artifacts) |
| Validation | Pandera | Schema in `src/ingestion/schema.py` |
| Drift detection | Evidently | Writes metrics to Prometheus |
| Metrics | Prometheus + Grafana | `prometheus-fastapi-instrumentator` for HTTP defaults |
| Object store | MinIO | One instance, two purposes (MLflow + DVC) |
| Backend store | Postgres | MLflow only |
| Container registry | Harbor | Self-hosted |
| CI | Forgejo Actions (preferred) or GitHub Actions (fallback) | Workflows transfer 1:1 |
| Orchestration | Kubernetes | k3s on VPS for live, AKS for recorded demo |
| Helm | Helm 3 | Two values files: `values-vps.yaml`, `values-aks.yaml` |
| IaC | OpenTofu | Not Terraform (BUSL since 2023) |
| Explainability | SHAP TreeExplainer | Wired to `/explain` endpoint |
| Lint/format | ruff | Both `ruff check` and `ruff format` |
| Types | mypy (loose) | Type hints expected on public APIs |
| Tests | pytest | Unit + integration; no E2E in CI |

---

## Architectural conventions

**MLflow promotion uses aliases.** Always.

```python
# Correct
client.set_registered_model_alias("hdb-predictor", "challenger", version=42)
model = mlflow.pyfunc.load_model("models:/hdb-predictor@champion")

# Wrong — stages are deprecated
client.transition_model_version_stage(...)  # Never write this
mlflow.pyfunc.load_model("models:/hdb-predictor/Production")  # Never write this
```

**Model reload is via background polling**, not restart, not admin endpoint. The FastAPI app spawns a daemon thread on startup that polls the registry every 60s and atomically swaps the in-memory model when `@champion` points to a new version. Implementation lives in `src/serving/model_loader.py`.

**Promotion thresholds are empirical.** The 2% RMSE gate is derived from 5-fold CV noise (≈1 std dev above the noise floor). Never hardcode a threshold without the CV calculation backing it. The derivation lives in `docs/adr/0003-promotion-threshold.md`.

**Synthetic drift is acknowledged, not hidden.** Drift demos use controlled perturbation (mean shift on `floor_area_sqm`, noise on `resale_price`). README and ADR `0004-synthetic-drift.md` frame this as simulating real Singapore HDB drift signals: Tengah new town entries, SBF completions, cooling measure adjustments, lease decay.

**Pandera ranges are empirical.** Bounds like `floor_area_sqm` between 28-200 must be backed by 1st/99th percentile of the actual training data, documented in `notebooks/01-data-profile.ipynb`.

**MLflow tracking server is a known SPOF.** Documented in `docs/adr/0002-mlflow-spof.md`. Do not silently work around it; any production-hardening discussion belongs in that ADR.

---

## Repository layout

```
.
├── src/
│   ├── training/       # train.py, evaluation, CV noise calc
│   ├── serving/        # FastAPI app, model_loader, schemas, SHAP
│   ├── ingestion/      # data.gov.sg client, Pandera schema, DVC hooks
│   └── monitoring/     # Evidently report runner, Prometheus emitter
├── infra/
│   ├── helm/           # chart, values-vps.yaml, values-aks.yaml
│   └── opentofu/       # VPS + AKS (separate state files)
├── docs/
│   ├── build-plan.md   # The full plan
│   ├── adr/            # Architecture Decision Records
│   └── runbooks/       # Operational procedures
├── notebooks/          # Profiling, CV analysis, SHAP exploration
├── tests/              # Mirrors src/ structure
├── docker-compose.yml  # Local dev stack
├── Dockerfile          # Multi-stage build for FastAPI
└── README.md           # Architecture diagram, quickstart, narrative
```

---

## Out of scope — do not propose adding

If a session starts suggesting any of these, push back:

- `/predict/batch` endpoint (deferred indefinitely; thin wrapper if ever needed)
- ArgoCD or any GitOps controller
- Sealed Secrets or External Secrets Operator (plain `kubectl create secret` with rotation runbook is fine)
- Feature store (Feast, Tecton, etc.)
- A/B testing infrastructure beyond `@shadow` alias
- Model cards (the README narrative covers this informally)
- ZenML or any pipeline orchestrator wrapping MLflow
- Switching off Postgres/MinIO to a "simpler" SQLite/local FS — they're already minimal
- Adding a frontend framework beyond Streamlit
- Multi-cloud anything

---

## Coding standards

- Python 3.12+.
- Type hints on all public functions and class methods. Inner helpers may skip.
- Pydantic v2 for all API schemas — never raw dicts at the API boundary.
- `ruff check` and `ruff format` clean before commit. Pre-commit hook enforces this.
- Docstrings on public APIs in NumPy or Google style — pick one and stay consistent within a module.
- No print statements in `src/`. Use `logging` configured in `src/serving/logging_config.py`.
- Constants in `src/<module>/config.py`, loaded from env via Pydantic Settings.
- Tests mirror source layout: `src/serving/model_loader.py` → `tests/serving/test_model_loader.py`.
- Test names describe behaviour: `test_model_swap_is_atomic_under_concurrent_predict`, not `test_swap`.

---

## Documentation style (for any markdown Claude writes)

The README, ADRs, and runbooks are interview-visible. Match the project author's voice:

- Direct and declarative. No hedging ("might be useful," "could potentially," "it's worth noting").
- Concrete specifics over vague claims. Name the tool, the version, the metric, the threshold.
- First person, active voice. "I chose MLflow because…" not "MLflow was chosen because…"
- Em dashes for asides — like this — rather than parentheses.
- Narrative flow in prose. Not everything needs to be a bullet list.
- British English. "Optimise," "behaviour," "centralised," "colour."
- Honest about limitations. Frame them with the production hardening path so they read as awareness, not gaps.
- Lead with the strongest proof point for whatever the section is about.

ADR template lives in `docs/adr/template.md`. New ADRs are numbered sequentially.

---

## Common commands

```bash
# Local dev stack
docker compose up -d
docker compose logs -f api

# Tests
pytest                         # all
pytest tests/serving/ -v       # one area
pytest -k "alias" -v           # by name pattern

# Lint + format
ruff check .
ruff format .
mypy src/

# MLflow
mlflow ui --backend-store-uri postgresql://...

# DVC
dvc pull                       # fetch data from MinIO
dvc push                       # publish updates
dvc repro                      # rerun pipeline if deps changed

# Helm (k3s live)
helm install hdb infra/helm/hdb -f infra/helm/values-vps.yaml
helm upgrade hdb infra/helm/hdb -f infra/helm/values-vps.yaml
helm uninstall hdb

# Helm (AKS demo)
helm install hdb infra/helm/hdb -f infra/helm/values-aks.yaml
# Teardown verified by infra/opentofu/aks/teardown.sh
```

---

## Session hygiene for Claude Code

- `/clear` between phases. Carrying Phase 2 context into Phase 5 wastes tokens and corrupts focus.
- Define architecture in chat (cheap) before implementing in Claude Code (expensive). Phase 5 especially benefits from a design pass first.
- Use subagents for isolated tasks — Dockerfile generation, schema writing, ADR drafting — to keep the main thread cheap.
- Read `docs/build-plan.md` and the relevant phase's acceptance criteria at the start of any session that touches a new phase.
- When in doubt about scope, the answer is "no, that's in the out-of-scope list" or "that goes in a future ADR, not this PR."

---

## Pre-flight checks before any commit

1. `ruff check . && ruff format --check .`
2. `mypy src/`
3. `pytest`
4. No `transition_model_version_stage` references in the diff.
5. No `print()` calls added to `src/`.
6. No new dependencies added without a one-line justification in the PR description.
7. If the change touches an architectural decision (model loading, promotion gate, drift thresholds), an ADR exists or has been updated.
