"""Human-readable labels for raw SHAP feature names.

The SHAP explainer operates on the post-``ColumnTransformer`` feature space, so
its feature names carry sklearn's transformer prefixes — ``num__floor_area_sqm``,
``cat__town_TAMPINES``, ``month__float_time_series``. Those are correct
identifiers, but they leak pipeline internals when an LLM client narrates them to
a human ("the month__float_time_series feature contributes..."). This module
translates a raw feature name into a label suitable for direct presentation,
interpolating the caller's input values where that adds context.

The translation happens once, at the MCP tool layer, so every consumer —
Claude Desktop today, the Phase 1.6c LangGraph orchestrator tomorrow — gets
readable labels without re-deriving what the pipeline did internally.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Exact-match labels for the numeric pass-throughs and the month transformer.
# Each maps a raw feature name to a function of the original five-field input
# dict, so the label can carry the concrete value the user supplied.
_EXACT_LABELS: dict[str, Callable[[dict[str, Any]], str]] = {
    "month__float_time_series": lambda v: f"Transaction date ({v['month']})",
    "num__floor_area_sqm": lambda v: f"Floor area ({v['floor_area_sqm']} sqm)",
    "num__lease_commence_date": lambda v: f"Lease start year ({v['lease_commence_date']})",
}


def _format_town(value: str) -> str:
    """Title-case a town name, preserving inter-word spacing.

    "TAMPINES" -> "Tampines", "BUKIT BATOK" -> "Bukit Batok".
    """
    return value.title()


def _format_flat_type(value: str) -> str:
    """Normalise a flat-type label to a single hyphenated form.

    Spaces and hyphens collapse to a single hyphen, then the whole token is
    capitalised: "4 ROOM" -> "4-room", "EXECUTIVE" -> "Executive",
    "MULTI GENERATION" / "MULTI-GENERATION" -> "Multi-generation".
    """
    return re.sub(r"[\s-]+", "-", value.strip()).capitalize()


# One-hot categorical columns: maps the raw column name to a friendly prefix
# and the value formatter to apply to the extracted category.
_CATEGORICAL: dict[str, tuple[str, Callable[[str], str]]] = {
    "town": ("Town", _format_town),
    "flat_type": ("Flat type", _format_flat_type),
}

# Matches ``cat__<column>_<value>`` where <column> is one of the known
# categorical columns. The column alternation is anchored before the value
# group so a column name containing an underscore (flat_type) is not split at
# the wrong boundary.
_CAT_PATTERN = re.compile(
    r"^cat__(" + "|".join(re.escape(column) for column in _CATEGORICAL) + r")_(.+)$"
)


def format_feature_label(raw_name: str, input_values: dict[str, Any]) -> str:
    """Translate a raw SHAP feature name into a human-readable label.

    Args:
        raw_name: A feature name as produced by the sklearn pipeline's
            ``get_feature_names_out`` — e.g. ``num__floor_area_sqm``,
            ``cat__town_TAMPINES``, ``month__float_time_series``.
        input_values: The original five-field prediction input. Used to
            interpolate concrete values into the numeric labels, so a label
            reads "Floor area (95 sqm)" rather than just "Floor area".

    Returns:
        A label suitable for direct presentation to a human. Unknown feature
        names are returned unchanged — leaking an unhandled raw name is
        preferable to raising on a feature the mapping does not cover.
    """
    exact = _EXACT_LABELS.get(raw_name)
    if exact is not None:
        return exact(input_values)

    match = _CAT_PATTERN.match(raw_name)
    if match is not None:
        column, value = match.group(1), match.group(2)
        friendly_label, formatter = _CATEGORICAL[column]
        return f"{friendly_label}: {formatter(value)}"

    return raw_name


def aggregate_one_hot_contributions(
    raw_contributions: dict[str, float],
    input_values: dict[str, Any],
) -> dict[str, float]:
    """Group one-hot SHAP contributions by their source categorical column.

    SHAP assigns a contribution to every one-hot dummy, including the dummies
    whose value is 0 for this row. Selecting the top-N by absolute SHAP value
    over the raw per-dummy contributions surfaces inactive categories — for a
    4-room flat the top contributors can include ``cat__flat_type_3 ROOM``,
    which a human reads as "being a 3-room added to the price" when it actually
    means "switching this flat to a 3-room would change the price by that much".
    That counterfactual reading is the wrong contract for a human-facing label.
    Aggregating the dummies of a column into a single signed sum, attributed to
    the row's active category, restores the intuitive "what this flat's town /
    flat type contributes" reading.

    For raw contributions like::

        {"cat__town_TAMPINES": 34666, "cat__town_BEDOK": -1030, ...,
         "num__floor_area_sqm": 15807, ...}

    and an input dict ``{"town": "TAMPINES", "flat_type": "4 ROOM", ...}``,
    each one-hot group collapses to a single entry keyed by its active dummy
    (the one matching the input value), carrying the signed sum of every dummy
    in the group. Numeric and month features pass through unchanged::

        {"cat__town_TAMPINES": <sum of all town dummies>,
         "cat__flat_type_4 ROOM": <sum of all flat_type dummies>,
         "num__floor_area_sqm": 15807,
         "num__lease_commence_date": -20353,
         "month__float_time_series": 246378}

    Args:
        raw_contributions: Per-feature SHAP contributions keyed by raw pipeline
            feature name, as produced by ``get_feature_names_out``.
        input_values: The prediction input. The categorical values must match
            the encoder's vocabulary (e.g. "TAMPINES", "4 ROOM") so the active
            dummy can be identified.

    Returns:
        A contribution dict with one-hot groups collapsed to their active dummy.
        If a group's active dummy is absent — an input value outside the
        training vocabulary — that group falls back to its per-dummy entries so
        no contribution is silently dropped.
    """
    grouped: dict[str, dict[str, float]] = {column: {} for column in _CATEGORICAL}
    aggregated: dict[str, float] = {}

    for name, value in raw_contributions.items():
        match = _CAT_PATTERN.match(name)
        if match is None:
            aggregated[name] = value
            continue
        grouped[match.group(1)][name] = value

    for column, dummies in grouped.items():
        if not dummies:
            continue
        active_name = f"cat__{column}_{input_values.get(column)}"
        if active_name in dummies:
            aggregated[active_name] = sum(dummies.values())
        else:
            logger.debug(
                "No active one-hot dummy %r for column %r in SHAP contributions; "
                "falling back to per-dummy entries.",
                active_name,
                column,
            )
            aggregated.update(dummies)

    return aggregated
