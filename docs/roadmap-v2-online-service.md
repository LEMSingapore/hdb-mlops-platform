# Roadmap v2 — HDB MLOps as a Live Online Service

**Author:** Chang Chee Young
**Status:** MVP scope locked; build pending
**Decisions locked:** 17 June 2026
**Supersedes:** the phase ordering in `docs/build-plan.md` (April 2026 snapshot, predates Phase 1.6 and the online-service goal)

---

## Why this document exists

The original build plan sequenced the work as: Phase 3 (DVC + Pandera + ingestion) → Phase 4 (Prometheus + Grafana) → Phase 5 (Evidently + retrain gate) → Phase 6 (deployment). That order was designed for learning each MLops layer in sequence, with deployment as the final step.

The actual goal has changed: a **live, publicly reachable HDB price service** that recruiters and interviewers can click and use. That reframes deployment from "final step" to "the spine everything else improves on." This document captures a deliberately minimal first release — a working public demo — and defers the ambitious pieces to a documented roadmap rather than building them under the pre-18-July time pressure.

The guiding principle: **a clickable URL that works, plus the story of how it is architected, beats a half-built full vision.** The deferred pieces become interview conversation ("here is my roadmap and why each piece matters") without the cost and risk of building them all now.

---

## MVP scope — what ships

A minimum live demo of the existing Phase 2 containerised stack, on a cheap always-on box, behind HTTPS, with cost protection on the LLM-calling surface.

### Public surfaces

- **FastAPI `/predict` and `/explain`** — the core ML serving story.
- **Streamlit chat UI** — the human-facing demo: natural language in, price + SHAP explanation out. The surface a recruiter actually plays with.
- **MCP server — NOT public.** Stays local, demonstrated live in interviews via Claude Desktop pointed at the local platform. Remote MCP is deferred (see roadmap).
- **MLflow tracking UI — NOT public.** Internal to the compose network only.

### Decisions (locked 17 June 2026)

1. **Public surface:** FastAPI `/predict` + `/explain` and the Streamlit chat. MCP local-only. MLflow internal.
2. **Hardening:** minimum viable — HTTPS via Caddy, rate-limiting, and a hard LLM budget cap (decision A). No user login/auth; the prediction demo is read-only and public by design.
3. **Auto-ingestion:** none in the MVP. Ships with the current trained champion model. The monthly data.gov.sg ingestion + retrain pipeline is the headline of the next phase, documented in the roadmap below.
4. **Deployment target:** cheapest Hetzner EU box (CX-class, ~SGD $5-8/month). Accepts ~150-250ms latency for Singapore users as an MVP trade-off; in-region hosting is a documented upgrade.
5. **(A) LLM cost exposure:** the Streamlit chat calls Claude Haiku per turn. Protected by a **separate Anthropic workspace** (not just a separate key — Anthropic spend limits are workspace-level) with a **hard monthly budget cap** (~SGD $20-30). The chat must **degrade gracefully** when the cap is hit: catch the quota error and show "demo limit reached this month, try the form-based predictor" rather than a stack trace. The form-based predictor (no LLM) remains available as the fallback.
6. **(B) Domain:** DuckDNS free subdomain (e.g. `hdb-mlops.duckdns.org`) for the MVP, with Caddy automatic TLS via the HTTP challenge. **Upgrade path:** a real domain (`changcheeyoung.com` or fallback `.dev`/`.app`) when registration resolves — a ~10-minute change (one DNS A-record + one Caddyfile hostname line + Caddy reload). Nothing else in the stack depends on the hostname.
7. **(C) Deploy mechanism:** manual SSH — `git pull` + `docker compose up -d` on the box. Automated push-to-deploy via GitHub Actions is deferred (Phase-6 territory).
8. **(D) Model on the box:** train once on the box after first deploy (`docker compose run` the training stage). This roots the MLflow registry's artifact paths to the box's own filesystem, sidestepping the absolute-path issue found in Phase 2 Session B. Requires the training data (SQLite DB ~162MB, or rebuilt from data.gov.sg CSVs) present on the box as a one-time step.

### Deployment posture

One Hetzner box running the existing Phase 2 `docker compose` stack (FastAPI + MLflow tracking), with **Caddy** as a reverse proxy in front providing automatic HTTPS and basic per-IP rate limiting. DuckDNS points a hostname at the box IP. Caddy provisions the Let's Encrypt cert via HTTP challenge.

```
Internet
   │  (HTTPS, DuckDNS hostname)
   ▼
 Caddy  ── reverse proxy + TLS + rate-limit
   │
   ├──► Streamlit chat  (calls Claude Haiku — budget-capped workspace)
   └──► FastAPI /predict, /explain  (no LLM — cheap to serve)
         │
         └──► MLflow tracking server  (internal only, not exposed)
```

---

## Minimum hardening checklist (resolves the MVP-critical subset of #34)

- **HTTPS** — Caddy automatic TLS. Required before sharing the URL.
- **LLM budget cap** — separate Anthropic workspace, hard monthly limit, graceful degradation in the chat. This is the bankruptcy-prevention control.
- **Basic rate-limit** — Caddy per-IP rate limiting on all routes. Protects `/predict`/`/explain` from CPU exhaustion (acceptable MVP risk: worst case the box gets slow, not expensive) and adds a layer in front of the chat.
- **No login** — deliberate. A public read-only prediction demo does not need authentication; it needs the budget cap and rate-limit above.

Explicitly NOT in MVP hardening (tracked in #34 for any future scaling): user auth, secret manager for keys, multi-replica, DDoS protection beyond basic rate-limit, WAF.

---

## Deferred roadmap (post-MVP, documented not built)

These are the ambitious pieces. Each is a legitimate interview talking point as roadmap. Each carries cost or risk that makes building it under the current deadline unwise.

### R1 — Automated monthly ingestion + retrain

The "auto-ingest monthly" goal. A scheduled job: pull latest data.gov.sg HDB CSVs → validate (Pandera) → version (DVC) → retrain a challenger → evaluate against champion on a held-out gate → **promote only if better**.

Critical design decision deferred to this phase: **auto-promotion vs human-in-the-loop.** Strong recommendation is human-in-the-loop (auto-train + auto-evaluate, but a notification/PR for manual promotion approval) rather than fully unattended champion swaps. Unattended promotion risks silently serving a worse model. Most production systems keep a human approving the swap.

Depends on: Pandera validation (R2) and DVC versioning (R3) as prerequisites — you cannot safely auto-retrain on data you have not validated and versioned.

### R2 — Data validation (Pandera)

Schema-on-read at ingestion: column types, value ranges, null rules, distribution-shift checks. Fail fast when data.gov.sg silently changes its export format. Prerequisite for trusting any automation.

### R3 — Data versioning (DVC)

Git-like versioning for the training data. Reproducibility: `git checkout` a commit + `dvc pull` recovers the exact data that trained that commit's model. Local-directory remote is sufficient for a portfolio piece; MinIO/S3 is the canonical upgrade.

### R4 — Remote MCP server

Expose the MCP server over Streamable HTTP transport with OAuth, so any MCP client can reach it remotely — not just local Claude Desktop. Highest-effort surface (transport + auth + HTTPS). For the MVP, MCP is demonstrated live in interviews from the local machine instead. (This is Issue #32 Path A.)

### R5 — Monitoring (Prometheus + Grafana)

HTTP metrics, prediction-distribution histograms, model-version gauge, alert rules. Matters more once the service is live and unattended than it did running by hand. Especially relevant once R1's monthly retrain is automated — you want to see if a retrain broke something.

### R6 — Model improvement: geospatial features

From the Tian Jie TDS article (2021) and general domain knowledge: enrich features with distance-to-nearest-MRT, distance-to-nearest-mall, travel-time-to-CBD. These were top predictors in that study. This is a **model-quality** improvement (Phase-5-ish), distinct from infrastructure. A new feature set → a new challenger → the promotion gate decides. The existing champion/challenger alias machinery already handles "the model changes"; this is just a better challenger. Note: the current model's heavy reliance on the time-trend feature (#30) is the related known limitation.

### R7 — In-region hosting + real domain

Upgrade the EU box to a Singapore-region box (lower latency, cleaner "hosted in Singapore" story) and swap DuckDNS for `changcheeyoung.com`. Both are easy swaps once justified by real traffic or interview need.

---

## Portfolio cleanup (related, non-blocking)

- **Retire the old HDB Streamlit demo** (`hdbresaleprice-...streamlit.app`) when the new platform goes live. It is the obsolete pickle-in-Streamlit version of this very project; two competing HDB demos confuse visitors. Point the portfolio link at the new service.
- **Migrate portfolio to a custom domain** once `changcheeyoung.com` registers — point GitHub Pages at the apex, serve the whole portfolio from `changcheeyoung.com` instead of `...github.io`. Easy professionalism win across all projects, not just HDB.

---

## Session 1 build plan (the MVP, first build session)

Goal: HDB platform live at a DuckDNS HTTPS URL, FastAPI + Streamlit chat public, model trained on the box, LLM budget-capped.

Rough sequence (the build session will refine):

1. **Provision the box.** Create cheapest Hetzner EU instance (Ubuntu), SSH in, install Docker + Docker Compose + git.
2. **Get the data + code on the box.** `git clone` the repo. scp the training SQLite DB (or rebuild from CSVs).
3. **Train once on the box.** `docker compose run` the training stage → champion registered with box-local artifact paths. Verify `@champion` loads.
4. **Bring up the stack.** `docker compose up -d` → FastAPI + MLflow tracking. Verify `/predict` returns the canonical Tampines prediction in-container.
5. **DuckDNS + Caddy.** Register the DuckDNS subdomain, point it at the box IP. Install Caddy (or add as a compose service), configure reverse proxy + automatic TLS + per-IP rate-limit for the Streamlit and FastAPI routes.
6. **Separate Anthropic workspace + budget cap.** Create the workspace, set the hard monthly limit, put its key in the box's `.env`. Add graceful-degradation handling to the Streamlit chat for the quota-exhausted case.
7. **Smoke test end to end** over HTTPS from a browser: form predict, chat predict, explanation rendering, and a deliberate cap-hit test to confirm graceful degradation.
8. **Docs + portfolio.** README "Live demo" section with the URL. Note the old-Streamlit retirement.

Estimated 8-15 hours, likely across 2-3 sessions. Comfortably shippable before 18 July with margin.

---

## What this roadmap is not

- **Not a full production deployment.** No auth, no multi-replica, no secret manager, no monitoring. MVP demo posture only. See #34.
- **Not the auto-retraining service yet.** Ships with a static trained model; automation is R1.
- **Not a re-architecture.** Deliberately reuses the Phase 2 compose stack as-is, on a box, behind Caddy. The scale-to-zero stateless-serving alternative (Path 1) was considered and declined in favour of minimal change.
