# Legacy Chat App (Reference Only)

This is the original Streamlit chat application from the pre-MLOps version of the HDB resale price predictor. **It is preserved here as reference material.** It is not run, tested, or deployed by the current platform.

## What it was

A Streamlit chat UI where users described a flat in plain English (e.g. "4-room in Tampines, 90 sqm, postal 520329, lease started 1992") and Claude Haiku extracted the required fields via tool use, then called a local XGBoost model.

## Why it's here, not used

Three mismatches with the current platform:

1. **Different model.** Used XGBoost; the current pipeline uses GradientBoostingRegressor.
2. **Different feature schema.** Used `floor_area`, `lease_commence_date`, `postal_code`, `town`, `flat_type` (5 fields). The current `HDBFeatureInput` schema has 7 fields including `flat_model`, `storey_range`, `month`, and no `postal_code`.
3. **Architectural mismatch.** Loaded a pickle directly via `joblib.load`. The current platform requires UI to call FastAPI `/predict` and `/explain` over HTTP.

## What's missing

The `models/XBR_trained_hdb_resale_modelV4a.pkl` file is not committed — it's not needed as reference. The chat app would not run from this directory anyway.

## Future plan

Once the platform is past Phase 2 (Docker + CI), this chat agent will be retrofitted to call `/predict` and `/explain` over HTTP, with the tool-use schema updated to match `HDBFeatureInput`. The chat then becomes a sibling app to the form-based `src/ui/streamlit_app.py`, demonstrating two ergonomic input modes against the same MLOps backend.

Tracked in a follow-up issue.
