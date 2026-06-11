"""Tests for the GraphState schema.

Pure Pydantic behaviour — defaults, validation, and the model_copy(update=)
pattern the graph relies on to produce a merged state between nodes.
"""

import pytest
from pydantic import ValidationError

from ui.chat_app.graph.state import GraphState, Status


class TestDefaults:
    def test_optional_fields_default_to_none(self) -> None:
        state = GraphState(user_message="hello")
        assert state.town is None
        assert state.flat_type is None
        assert state.floor_area_sqm is None
        assert state.lease_commence_date is None
        assert state.month is None
        assert state.postal_code is None
        assert state.resolved_town is None
        assert state.predicted_price is None
        assert state.model_version is None
        assert state.base_value is None
        assert state.error is None

    def test_collection_and_status_defaults(self) -> None:
        state = GraphState(user_message="hello")
        assert state.missing_fields == []
        assert state.top_contributors == []
        assert state.response_text == ""
        assert state.status == "pending"

    def test_list_defaults_are_not_shared_between_instances(self) -> None:
        first = GraphState(user_message="a")
        second = GraphState(user_message="b")
        first.missing_fields.append("town")
        assert second.missing_fields == []


class TestValidation:
    def test_user_message_is_required(self) -> None:
        with pytest.raises(ValidationError):
            GraphState()  # type: ignore[call-arg]

    def test_status_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            GraphState(user_message="x", status="bogus")  # type: ignore[arg-type]

    def test_status_accepts_each_known_value(self) -> None:
        statuses: tuple[Status, ...] = (
            "pending",
            "ready_to_predict",
            "needs_follow_up",
            "out_of_scope",
            "error",
        )
        for status in statuses:
            assert GraphState(user_message="x", status=status).status == status

    def test_floor_area_int_coerced_to_float(self) -> None:
        state = GraphState(user_message="x", floor_area_sqm=95)
        assert state.floor_area_sqm == 95.0
        assert isinstance(state.floor_area_sqm, float)


class TestModelCopyUpdate:
    def test_update_returns_new_state_with_changed_field(self) -> None:
        original = GraphState(user_message="x")
        updated = original.model_copy(update={"town": "TAMPINES"})
        assert updated.town == "TAMPINES"

    def test_update_leaves_original_unchanged(self) -> None:
        original = GraphState(user_message="x")
        original.model_copy(update={"town": "TAMPINES"})
        assert original.town is None

    def test_update_merges_multiple_fields(self) -> None:
        original = GraphState(user_message="x")
        updated = original.model_copy(
            update={"status": "ready_to_predict", "predicted_price": 500_000.0}
        )
        assert updated.status == "ready_to_predict"
        assert updated.predicted_price == 500_000.0
