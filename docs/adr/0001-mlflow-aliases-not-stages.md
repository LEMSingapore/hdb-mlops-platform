# ADR 0001: Use MLflow Aliases Instead of Stages

**Status:** Accepted
**Date:** 2026-04-28

## Context

MLflow Model Registry historically used four stages to manage the model lifecycle: `None`, `Staging`, `Production`, and `Archived`. Models were promoted by transitioning between stages via `MlflowClient().transition_model_version_stage()`.

MLflow deprecated stage-based promotion starting in version 2.9, in favour of a more flexible alias and tag-based system. Stages still function in 2026 but emit deprecation warnings and will be removed in a future major release.

This project will be reviewed by senior ML engineers during job interviews. Code that uses deprecated APIs is a negative signal regardless of whether it still runs.

## Decision

I will use MLflow **aliases** for all model promotion and lifecycle management:

- `@champion` — the model currently serving production traffic
- `@challenger` — a candidate under evaluation that has passed the promotion gate
- `@shadow` — optional, for parallel inference comparison against the champion

Promotion is performed via `client.set_registered_model_alias()`. Models are loaded by alias using `models:/hdb-predictor@champion`.

No code in this repository will call `transition_model_version_stage()` or load models by stage path (`models:/<name>/Production`).

## Consequences

**Positive:**

- Future-proof against the removal of stages in upcoming MLflow versions.
- Aliases are more expressive — `@champion` and `@shadow` can point to different versions simultaneously, supporting champion/challenger and shadow-mode evaluation cleanly.
- Rollback becomes a single alias reassignment rather than a stage transition.
- Code reads more naturally to reviewers familiar with the modern MLflow API.

**Trade-offs:**

- Some MLflow tutorials and examples still use stages, so adapting them requires translation.
- The serving layer must explicitly resolve aliases to model versions on load and on background refresh, slightly more code than referencing a stage path directly.

## Alternatives considered

**Stay on stages.** Rejected — deprecated, will be removed, and signals outdated practice to interviewers. The fact that stages still function is not a sufficient reason to keep using them.

**Tags only, no aliases.** Rejected — tags are descriptive metadata, not promotion targets. They cannot be loaded directly via the model URI scheme. Using tags alone would require custom resolution logic where aliases provide it natively.

**Custom stage system via tags.** Rejected as scope creep. The MLflow team built aliases specifically to replace stages; reinventing this layer adds maintenance burden with no benefit.
