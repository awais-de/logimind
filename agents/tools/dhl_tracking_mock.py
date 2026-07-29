"""Simulated DHL shipment tracking tool for RetrieverAgent."""

import hashlib
import logging
import random
from datetime import datetime, timedelta

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_STATUSES = ["Pending Pickup", "In Transit", "Customs Clearance", "Out for Delivery", "Delivered"]
_HUBS = ["Leipzig, DE", "Cincinnati, US", "Hong Kong, HK", "Singapore, SG", "Bonn, DE"]


class TrackingEvent(BaseModel):
    """A single event in a shipment's mock tracking history.

    Attributes:
        timestamp: When the event occurred, ISO 8601.
        location: Where the event occurred.
        description: Human-readable description of the event.
    """

    timestamp: str
    location: str
    description: str


class TrackingInfo(BaseModel):
    """Mock tracking status for a shipment.

    Attributes:
        tracking_number: The queried tracking number.
        status: Current shipment status.
        origin: Mock origin hub.
        destination: Mock destination hub.
        estimated_delivery: Mock estimated delivery date, ISO 8601.
        events: Tracking history, oldest first.
    """

    tracking_number: str
    status: str
    origin: str
    destination: str
    estimated_delivery: str
    events: list[TrackingEvent]


def dhl_tracking_mock(tracking_number: str) -> dict:
    """Return simulated tracking status for a shipment.

    This is a mock tool, not a real DHL tracking lookup: it deterministically
    derives plausible status data from the tracking number, so the same
    tracking number always produces the same status, origin, destination,
    and event history.

    Args:
        tracking_number: The shipment tracking number to look up.

    Returns:
        Mock tracking info as a dict, matching TrackingInfo's fields.
    """
    rng = random.Random(int(hashlib.sha256(tracking_number.encode()).hexdigest(), 16))

    origin, destination = rng.sample(_HUBS, 2)
    status_index = rng.randrange(len(_STATUSES))
    status = _STATUSES[status_index]

    now = datetime.now()
    events = []
    for i, past_status in enumerate(_STATUSES[: status_index + 1]):
        event_time = now - timedelta(days=(status_index - i), hours=rng.randrange(0, 12))
        location = origin if i == 0 else rng.choice(_HUBS)
        events.append(
            TrackingEvent(
                timestamp=event_time.isoformat(),
                location=location,
                description=past_status,
            )
        )

    estimated_delivery = (now + timedelta(days=rng.randrange(1, 5))).date().isoformat()

    info = TrackingInfo(
        tracking_number=tracking_number,
        status=status,
        origin=origin,
        destination=destination,
        estimated_delivery=estimated_delivery,
        events=events,
    )
    return info.model_dump()
