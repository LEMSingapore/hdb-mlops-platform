# ADR 0002: SQLite as a Derived Data Store for Training and Queries

**Status:** Accepted
**Date:** 2026-05-05

## Context

The training pipeline previously read five raw CSVs from `data/raw/` directly in `train.py`, concatenating them with pandas and selecting the five model features. This worked but had two problems as the project grew.

First, the Phase 1.6b MCP tool `find_similar_transactions` needs to search historical transactions by town, flat type, floor area, and lease year. Running this as a pandas operation on raw in-memory CSVs on every tool call is expensive and forces the tool to hold 975k rows in memory for a lookup that benefits from indexing.

Second, AIAP submissions expect data loading to go through a structured query interface rather than reading raw files inline in the training script. An SQLite-backed data layer satisfies that expectation without introducing a separate database service.

## Decision

I will maintain a SQLite database at `data/hdb.db` as a derived artefact built from the raw CSVs by `scripts/csv_to_sqlite.py`. The database is gitignored — the CSVs in `data/raw/` remain the source of truth. The `src/data/` module provides the access layer: `get_connection()` for connection management, `load_all_transactions()` for training, `count_transactions()` for sanity checks, and `find_similar()` for the MCP tool's nearest-neighbour lookup.

The transactions table schema retains all columns from the original CSVs — month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price — with four indexes targeting the expected query patterns.

**remaining_lease is dropped.** The 2017+ CSV includes `remaining_lease`, but the value is computable as `99 - (transaction_year - lease_commence_date)` and is not used anywhere in the current model or queries. Storing it would require handling it as a nullable column across the earlier CSVs, which adds complexity for no benefit. If a future query needs it, it can be computed on the fly.

## Consequences

**Positive:**

- `find_similar()` can use SQL indexes (town, flat_type, month, town+flat_type composite) rather than pandas filtering on an in-memory DataFrame. At 975k rows with exact town+flat_type filtering, the indexed query is substantially faster for the MCP use case.
- Training reads are cleaner — one `load_all_transactions()` call replaces a CSV loop with five hardcoded filenames.
- `data/hdb.db` is a single file that can be rebuilt deterministically from the committed CSVs. Rebuilding takes about 20 seconds and produces a 60-70 MB file.
- The data layer is isolated in `src/data/` with its own tests, making it easy to swap the backing store in a future phase if needed.

**Trade-offs:**

- `python scripts/csv_to_sqlite.py` is a new required setup step. Missing it gives a clear error message directing users to run it; the README setup block includes the step.
- SQLite is a single-file, single-writer database — fine for this portfolio project, where training is a single process and the MCP tool is read-only. It is not suitable as-is if multiple training jobs run concurrently or if writes are needed at query time.
- The database file is not tracked in version control. On a fresh clone the user must run `csv_to_sqlite.py` before training or testing against real data. Test suites use the `tiny_sqlite_db` fixture instead of the real database, so tests do not require the build step.

## Alternatives considered

**Keep reading CSVs directly in train.py.** Rejected because it does not give `find_similar_transactions` a queryable backing store. The MCP tool could build an in-memory index on startup, but that reloads 975k rows every time the server starts — wasteful and fragile. SQLite with indexes is the right tool.

**Postgres.** Rejected for Phase 1.6 — Postgres requires a running server, adding a Docker dependency before Phase 2 containerisation. SQLite is stdlib, zero-config, and already sufficient for the read patterns this project needs. Phase 2 may introduce Postgres as the MLflow backend store; if the data layer also needs to move then, the `src/data/connection.py` abstraction makes the change local.

**DVC-tracked SQLite file.** Rejected as premature. DVC is planned for Phase 3 when data versioning becomes necessary. Adding it now to version a file that rebuilds from committed CSVs in 20 seconds is overhead without benefit.

**Parquet with pandas.** Rejected — Parquet gives columnar compression but no indexing or SQL interface. `find_similar_transactions` benefits from indexed lookups; a Parquet file would require loading all rows and filtering in pandas on every call, which is worse than the current CSV approach and far worse than SQLite with indexes.
