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

import re
from collections.abc import Callable
from typing import Any

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
