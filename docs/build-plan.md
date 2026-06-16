# HDB Resale Price Predictor — MLOps Platform Build Plan

**Author:** Chang Chee Young
**Status:** v2 — incorporates external review
**Last updated:** April 2026

---

## Revisions from v1

- MLflow stages replaced with aliases (`@champion`, `@challenger`) throughout. Stages still function in 2026 but are deprecated; new code targets the modern API.
- Phase resequenced: monitoring (Prometheus + Grafana) now precedes drift detection so Evidently has a metrics target.
- Promotion gate thresholds will be derived empirically from cross-validation noise floor, not picked.
- SHAP `/explain` endpoint added to Phase 1.
- AKS deliverable changed to recorded demo. Always-on live URL served from k3s on a small VPS to honour the OSS-only constraint.
- Sealed Secrets, ArgoCD stretch goal, and `/predict/batch` removed from scope.
- Model reload changed from restart-based to background polling against the registry.
- All proprietary tooling removed from the production stack (see Section 5).

---

## 1. Context & Objectives

I'm upgrading an existing sklearn-based HDB resale price predictor (Streamlit + pickled model) into a production-grade MLOps platform. This is a portfolio project supporting ML Engineering job applications in Singapore.

**Why HDB specifically.** Singapore's data.gov.sg publishes resale transactions monthly, giving me a genuine retraining and drift detection narrative most portfolio projects fake with synthetic data shifts.

**Primary objectives:**

1. Demonstrate the full MLOps loop: data ingest → validate → train → register → serve → monitor → detect drift → retrain → gated promotion.
2. Produce concrete interview talking points: champion/challenger eval, drift thresholds derived from CV noise floor, alias-based model promotion, K8s deployment.
3. Reuse prior AKS capstone work (Terraform IaC, HPA, LoadBalancer) as the recorded K8s demo.

**Constraints:**

- Solo build, ~13-15 hours/week (revised up from v1 — 10 hrs/week is not enough).
- Open source only. No proprietary services in the production stack.
- Driven primarily by Claude Code with subagent orchestration.
- Must ship within 5-6 weeks part-time so it's available for September job applications.

---

## 2. Current State

- sklearn pipeline (Gradient Boosting) trained on HDB resale data through 2024.
- Model serialised as pickle, loaded directly inside the Streamlit app.
- No tests, no CI, no data versioning, no monitoring, no retraining loop.
- Deployed on Streamlit Community Cloud.
- Repository: monolithic, no separation between training, serving, and UI.

---

## 3. Target Architecture

**Data plane:** data.gov.sg HDB resale dataset → scheduled ingestion job → Pandera validation → DVC-versioned snapshot → training pipeline.

**Training plane:** Training script logs runs to MLflow (params, metrics, artifacts). Trained models registered in MLflow Model Registry. Promotion managed via aliases: `@champion` (currently serving), `@challenger` (under evaluation), `@shadow` (optional, for parallel inference comparison). Promotion gate compares challenger against champion on a held-out evaluation set with thresholds derived from cross-validation noise.

**Serving plane:** FastAPI service loads `models:/hdb-predictor@champion` from the MLflow registry at startup. Background thread polls the registry every 60s and atomically swaps the model if the alias points to a new version. Endpoints: `/predict`, `/explain` (SHAP), `/health`, `/model-info`. Streamlit becomes a thin client.

**Monitoring plane:** Prometheus scrapes FastAPI metrics (latency, throughput, error rate, prediction distribution histograms). Grafana dashboards. Evidently runs scheduled drift reports against the training reference and writes drift metrics back to Prometheus, making drift a first-class observable signal.

**Orchestration plane:** Forgejo Actions (or GitHub Actions if convenience wins over OSS purity — see Section 5 note) for CI and scheduled jobs. Helm-deployed FastAPI + MLflow on k3s for the always-on live URL. AKS deployment exists as a recorded demo only.

---

## 4. Phase Breakdown

Each phase has explicit acceptance criteria — a reviewer should be able to verify "done" objectively.

### Phase 1 — FastAPI + MLflow with Aliases + SHAP (14-19 hours)

**Scope:** Decouple model from app. Stand up MLflow tracking server and registry using the modern alias-based API. Build FastAPI service that loads from registry with background reload. Add SHAP explainability.

**Deliverables:**

- Refactored `train.py` that logs runs to MLflow (params, metrics, model artifact, signature, input example).
- MLflow tracking server running locally with SQLite backend store and local filesystem artifact store (Postgres + MinIO deferred to Phase 2).
- Model versions promoted via `set_registered_model_alias()` — no `transition_model_version_stage()` calls anywhere.
- FastAPI app with Pydantic request/response schemas.
- Endpoints: `POST /predict`, `POST /explain`, `GET /health`, `GET /model-info`.
- Model loaded from registry by alias (`models:/hdb-predictor@champion`) at startup.
- Background reload thread that polls the registry every 60s and atomically swaps the in-memory model when `@champion` points to a new version.
- SHAP TreeExplainer wired to `/explain` — returns feature contributions for a single prediction.
- Streamlit refactored as a thin HTTP client with a SHAP waterfall visualisation panel.

**Acceptance criteria:**

- `mlflow ui` shows at least 3 training runs with comparable metrics.
- A model can be promoted to `@champion` via MLflow API and FastAPI picks it up within 60s **without restart**.
- No references to `MlflowClient().transition_model_version_stage()` exist in the codebase.
- `/explain` returns a structured SHAP response in under 200ms for a single prediction.
- `pytest` covers schema validation, end-to-end prediction, and explainability output shape.

**Risks:** MLflow's signature inference can be finicky with sklearn pipelines containing custom transformers. Mitigation: explicit signature definition.

---

### Phase 2 — Docker + Forgejo Actions CI (6-8 hours)

**Status: complete** (Sessions A #37, B #38, C #41). Three deliberate deltas from the plan below, each recorded in an ADR. The MLflow backend stayed on SQLite rather than moving to Postgres + MinIO — the stack runs a single tracking server and a single FastAPI replica, so there are no concurrent writers, and the Postgres swap is the documented upgrade if that ever changes ([ADR 0006](adr/0006-mlflow-tracking-server-as-compose-service.md)). CI runs on GitHub Actions, not Forgejo — the workflows transfer 1:1, so this is the reversible convenience choice the note below anticipated. And CI verifies the image builds and runs end-to-end but does not push it to a registry; the push has no consumer until there is a deploy target, so it is deferred to Phase 6 ([ADR 0007](adr/0007-ci-workflow-and-registry-strategy.md)). CI seeds a synthetic `@champion` for its compose smoke test because the real artifact is too large to commit and the training data is not on the runner.

**Scope:** Containerise everything. Wire up CI.

**Deliverables:**

- Multi-stage Dockerfile for FastAPI service (builder + runtime, non-root user, < 400MB final image).
- `docker-compose.yml` running FastAPI + MLflow + Postgres (MLflow backend store) + MinIO (artifact store, also doubles as DVC remote — see Phase 3).
- CI workflow: ruff lint → pytest → docker build → push to self-hosted Harbor registry on main.
- Pre-commit hooks: ruff, ruff-format, mypy (loose), trailing whitespace.

**Acceptance criteria:**

- `docker compose up` produces a working stack accessible at documented ports.
- A failed test blocks the registry push.
- Image is built and tagged with both `latest` and the commit SHA.

**Note on CI choice:** Forgejo Actions is the OSS-pure option. If self-hosting Forgejo proves to be a time sink, document the trade-off and use GitHub Actions on a public repo — the workflow files transfer 1:1, so this is reversible.

---

### Phase 3 — DVC + Pandera + Scheduled Ingestion (10-14 hours)

**Scope:** Version data. Validate it. Pull fresh data on a schedule.

**Deliverables:**

- DVC initialised with the project's MinIO instance as remote (consolidates infra — no separate B2/R2 dependency).
- `data/raw/` and `data/processed/` tracked by DVC, not Git.
- Pandera schema for HDB data: town categorical with allowed values, `floor_area_sqm` in plausible range (validated empirically — see acceptance criteria), `resale_price` positive, no nulls in required fields, `month` parseable as date.
- Ingestion script pulling latest month from data.gov.sg API, validating, then committing to DVC.
- Scheduled CI job (cron) running monthly that pulls new data and opens a PR if validation passes.

**Pre-work required:** Verify current data.gov.sg API endpoints. The resale dataset has been migrated in recent years; the v1 plan referenced an API surface that may have changed. Confirm the current endpoint, auth model, and rate limits before scoping the ingestion script.

**Acceptance criteria:**

- `dvc pull` reproduces the training dataset on a fresh clone.
- Validation rejects intentionally corrupted data with a clear error.
- The Pandera `floor_area_sqm` range is justified by the 1st and 99th percentile of the actual training distribution, documented in a notebook.
- The scheduled job successfully ran at least once end-to-end.

**Risks:** DVC remote auth in CI is fiddly. Mitigation: use repository secrets and document the setup in README. data.gov.sg API may have rate limits — handle with backoff.

---

### Phase 4 — Prometheus + Grafana Monitoring (6-9 hours) *(moved earlier from v1)*

**Scope:** Observability over the serving layer. Establishes the metrics substrate before Phase 5 needs it.

**Deliverables:**

- `prometheus-fastapi-instrumentator` integrated for default HTTP metrics.
- Custom metrics: prediction value histogram (bucketed by SGD 100k), model version gauge, model reload counter.
- Prometheus scrape config in `docker-compose.yml`.
- Grafana dashboard JSON checked into repo with panels for: request rate, latency p50/p95/p99, error rate, prediction distribution, current model version.
- Alert rules: error rate > 1% over 5min, p95 latency > 500ms over 5min.

**Acceptance criteria:**

- Dashboard loads with live data when the stack is running.
- Forcing a deliberate error (malformed request) shows up in the error rate panel within 30s.
- Alert rules evaluate without errors in Prometheus.
- Drift metric placeholders exist on the dashboard (populated in Phase 5).

**Risks:** Cardinality explosion if metrics are labelled by raw town/flat_type combinations. Mitigation: bucket and cap labels.

---

### Phase 5 — Evidently + Retrain/Promotion Loop with Empirical Thresholds (16-21 hours) *(moved later from v1, scope expanded)*

**Scope:** Detect drift. Trigger retraining. Gate promotion using empirically derived thresholds.

**Deliverables:**

- **CV noise floor analysis:** 5-fold cross-validation on the current Gradient Boosting model. Record mean and standard deviation of RMSE across folds. Set the promotion threshold at approximately 1 standard deviation above the CV noise floor. Document the calculation in an ADR.
- Reference dataset snapshot stored as a DVC artifact.
- Evidently report comparing current production data against reference: PSI on numerical features, Jensen-Shannon divergence on categoricals, target drift on `resale_price`.
- Drift thresholds in config (e.g., PSI > 0.2 on any numerical feature triggers alert, > 0.3 triggers retrain).
- Drift metrics written back to Prometheus via the Evidently → Prometheus integration.
- Retraining pipeline that, when triggered, trains a candidate model on the latest data window.
- **Promotion gate:** candidate (assigned `@challenger` alias) must beat champion on holdout RMSE by at least the CV-derived threshold AND not regress on per-town RMSE by more than 5% on any single town. (Latter prevents global wins masking local regressions in low-volume towns.)
- If gate passes, candidate's `@challenger` alias is upgraded to `@champion` automatically. The previous champion retains version history; rollback is one alias reassignment.
- Promotion gate failure reasons logged as structured metrics visible in Grafana.

**Acceptance criteria:**

- Synthetic drift injection (shift mean `floor_area_sqm` by 20%, add 15% noise to `resale_price`) reliably triggers the drift detector.
- A retraining run that fails the promotion gate does NOT get promoted, and the failure reason appears in the Grafana drift panel.
- A retraining run that passes the gate has its `@challenger` alias atomically reassigned to `@champion`.
- README contains the CV noise floor calculation and the derivation of the promotion threshold.
- Synthetic drift framing is documented honestly: reference covers 2020-2024 transactions, perturbation simulates macro repricing conditions (Tengah new town entries, SBF completions, cooling measure adjustments, lease decay effects).

**Risks:** HDB data drifts slowly in reality. Mitigation already documented above — synthetic perturbation is acknowledged as controlled testing, not hidden.

---

### Phase 6 — Deployment: k3s Live + AKS Recorded (8-12 hours)

**Scope:** Production-style deployment. Always-on URL via k3s on a small VPS. AKS as a recorded interview demo using prior Gladwell work.

**Deliverables:**

- Helm chart with `values-vps.yaml` and `values-aks.yaml`.
- Deployments for FastAPI, MLflow tracking server, Postgres, MinIO, Prometheus, Grafana.
- NGINX Ingress + cert-manager for TLS (Let's Encrypt).
- HPA on FastAPI keyed off CPU and request rate.
- ConfigMaps for non-secret config; plain Kubernetes Secrets created via `kubectl` with documented rotation procedure.
- **Live deployment:** k3s on a Hetzner CX22 (or equivalent OVH/Linode VPS), ~€5/month, always-on, accessible at a stable URL.
- **Recorded demo:** 3-5 minute screen recording on AKS showing `helm install` → Locust load test triggering HPA scale-up from 1 → ≥ 3 replicas → `helm uninstall` leaving no dangling Azure resources. Linked from README.

**Acceptance criteria:**

- `helm install -f values-vps.yaml` produces a working deployment on k3s reachable via Ingress with valid TLS.
- Same chart with `values-aks.yaml` produces a working deployment on AKS.
- Load test on AKS triggers HPA scale-up from 1 to ≥ 3 replicas (recorded).
- AKS teardown is scripted and verified to leave no dangling resources (recorded).
- README links to both the live k3s URL and the AKS demo recording.

**Risks:** Single-node k3s on a small VPS has no HA. Acceptable for portfolio. Document this in the architecture ADR alongside the MLflow SPOF discussion.

---

## 5. Technology Choices & Rationale

All open source. License notes included where relevant.

| Choice | License | Rationale |
|---|---|---|
| MLflow | Apache 2.0 | Registry built-in, widely recognised in JDs, alias-based workflow is modern |
| FastAPI | MIT | Async, Pydantic-native, fast, idiomatic Python |
| Streamlit | Apache 2.0 | Library is OSS; current Community Cloud hosting is proprietary — to be replaced by the k3s deployment |
| DVC | Apache 2.0 | Git-native, simple mental model, MinIO remote keeps infra consolidated |
| Pandera | MIT | Pythonic, integrates with pandas natively |
| Evidently | Apache 2.0 | Purpose-built for ML drift, Prometheus integration |
| Prometheus + Grafana | Apache 2.0 / AGPL | Industry standard, free, signal in JDs (AGPL relevant only if modifying and distributing Grafana) |
| MinIO | AGPL v3 | Doubles as MLflow artifact store and DVC remote |
| Postgres | PostgreSQL License | MLflow backend store |
| Harbor | Apache 2.0 | Self-hosted OCI registry |
| Forgejo + Forgejo Actions | GPL v3+ | OSS-pure GitHub alternative; reversible to GitHub Actions if time-constrained |
| k3s | Apache 2.0 | Lightweight Kubernetes, runs comfortably on a small VPS |
| AKS | Proprietary (managed wrapper over OSS Kubernetes) | Recorded demo only — not in the always-on stack |
| OpenTofu | MPL 2.0 | Replaces Terraform (now BUSL, no longer OSI-approved) |
| SHAP | MIT | Tree explainer is fast enough for online inference |

---

## 6. Cross-Cutting Concerns

**Repository structure:** Monorepo with clear separation — `src/training/`, `src/serving/`, `src/ingestion/`, `infra/helm/`, `infra/opentofu/`, `notebooks/`, `tests/`.

**Documentation:** README at root with architecture diagram, quickstart, and design decisions log. ADRs for significant choices: alias-based promotion (vs stages), promotion threshold derivation, MLflow SPOF acknowledgement, k3s vs AKS split.

**Testing strategy:** Unit tests on transformers, validators, and SHAP output shapes. Integration tests on the FastAPI app with a fixture model. No end-to-end test against k3s — too slow for CI.

**Secrets:** Repository secrets for CI. `kubectl create secret` for cluster credentials with rotation procedure documented. Nothing in plaintext, nothing in `.env` committed.

---

## 7. Known Limitations (for the README and interview prep)

These are documented honestly rather than hidden:

1. **MLflow tracking server is a SPOF.** Acceptable for portfolio. Production hardening path: HA Postgres cluster + S3-backed artifact store. Documented as ADR.
2. **k3s single-node deployment has no HA.** Acceptable for portfolio. AKS demo shows the multi-node story.
3. **Synthetic drift injection** for the demo. Real HDB drift is slow; injection is controlled perturbation testing, framed as such.
4. **Production promotion is automated within the gate.** Some teams prefer manual promotion gates for the final `@champion` reassignment. Trade-off documented; flag exists in config to switch behaviour.

---

## 8. Success Metrics

The project is "done" when:

- A reviewer can clone the repo and run `docker compose up` to get a working local stack within 5 minutes.
- The k3s live URL is reachable with a valid TLS cert.
- The AKS demo recording is linked from the README.
- README contains an architecture diagram, design rationale, ADRs, and a clear narrative I can walk an interviewer through in 10 minutes.
- I can describe — without notes — exactly what happens when new data arrives, from ingestion through optional promotion, including the CV noise floor justification for the promotion threshold.
- At least one interview yields an unprompted positive comment about the project.

---

## 9. Revised Timeline

| Phase | Hours | Cumulative |
|-------|-------|-----------|
| 1. FastAPI + MLflow + SHAP | 14-19 | 14-19 |
| 2. Docker + CI | 6-8 | 20-27 |
| 3. DVC + Pandera + Ingestion | 10-14 | 30-41 |
| 4. Prometheus + Grafana | 6-9 | 36-50 |
| 5. Evidently + Retrain Loop | 16-21 | 52-71 |
| 6. k3s Live + AKS Recorded | 8-12 | 60-83 |

At 13-15 hours/week, this lands in 5-6 weeks. Tight but achievable. If anything slips, the candidate cuts (in order) are: SHAP `/explain` reduced to a notebook rather than an endpoint; Forgejo deferred in favour of GitHub Actions; AKS recording deferred past initial launch.
