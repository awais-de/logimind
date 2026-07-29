"""Unit tests for agents/tools/dhl_tracking_mock.py."""

from agents.tools.dhl_tracking_mock import dhl_tracking_mock


def test_returns_expected_keys() -> None:
    result = dhl_tracking_mock("1234567890")

    assert set(result.keys()) == {
        "tracking_number",
        "status",
        "origin",
        "destination",
        "estimated_delivery",
        "events",
    }
    assert result["tracking_number"] == "1234567890"
    assert all(
        set(event.keys()) == {"timestamp", "location", "description"}
        for event in result["events"]
    )


def test_is_deterministic_for_same_tracking_number() -> None:
    first = dhl_tracking_mock("1234567890")
    second = dhl_tracking_mock("1234567890")

    assert first["status"] == second["status"]
    assert first["origin"] == second["origin"]
    assert first["destination"] == second["destination"]
    assert first["estimated_delivery"] == second["estimated_delivery"]
    assert [(e["location"], e["description"]) for e in first["events"]] == [
        (e["location"], e["description"]) for e in second["events"]
    ]


def test_differs_for_different_tracking_numbers() -> None:
    first = dhl_tracking_mock("1234567890")
    second = dhl_tracking_mock("ABCDEF9999")

    assert (first["origin"], first["destination"]) != (second["origin"], second["destination"])


def test_origin_and_destination_are_different_hubs() -> None:
    result = dhl_tracking_mock("some-tracking-number")

    assert result["origin"] != result["destination"]


def test_events_are_non_empty_and_end_with_current_status() -> None:
    result = dhl_tracking_mock("1234567890")

    assert len(result["events"]) >= 1
    assert result["events"][-1]["description"] == result["status"]
