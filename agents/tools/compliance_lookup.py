"""Structured shipment-classification lookup tool for RetrieverAgent."""

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ComplianceRule(BaseModel):
    """One structured compliance rule for a shipment category and destination.

    Attributes:
        category: Item category or classification the rule applies to.
        destination: Destination country/region code the rule applies to.
        restrictions: Restrictions that apply to this category and
            destination.
        documentation_required: Documentation needed to ship this category
            to this destination.
    """

    category: str
    destination: str
    restrictions: list[str]
    documentation_required: list[str]


_COMPLIANCE_TABLE: list[ComplianceRule] = [
    ComplianceRule(
        category="lithium_batteries",
        destination="DE",
        restrictions=[
            "Must be shipped as Dangerous Goods if not installed in equipment",
            "Watt-hour rating must be marked on the outer package",
        ],
        documentation_required=["Dangerous Goods Declaration", "UN38.3 test summary"],
    ),
    ComplianceRule(
        category="lithium_batteries",
        destination="US",
        restrictions=[
            "Must follow IATA Section II packing instructions if installed in equipment",
            "State of charge must not exceed 30% for standalone cells",
        ],
        documentation_required=["UN38.3 test summary"],
    ),
    ComplianceRule(
        category="alcohol",
        destination="DE",
        restrictions=[
            "Import permitted only below the personal-use duty threshold",
            "Prohibited above 70% ABV",
        ],
        documentation_required=["Commercial invoice with ABV stated"],
    ),
    ComplianceRule(
        category="alcohol",
        destination="UK",
        restrictions=["Requires an alcohol duty declaration above the personal allowance"],
        documentation_required=["Commercial invoice", "Alcohol duty declaration"],
    ),
    ComplianceRule(
        category="perishable_food",
        destination="NL",
        restrictions=[
            "Subject to EU food safety import controls",
            "Must ship on a temperature-controlled service",
        ],
        documentation_required=["Health certificate", "Commercial invoice"],
    ),
]


def compliance_lookup(category: str, destination: str) -> dict | None:
    """Look up structured restrictions/documentation for a shipment.

    This is a small, hand-curated reference table covering a handful of
    item categories and destinations. It illustrates matching against
    structured rules rather than free-text document search -- it is not
    an authoritative or complete source of DHL's actual customs and
    compliance requirements.

    Args:
        category: Item category or classification, e.g.
            "lithium_batteries". Matched case-insensitively; spaces are
            treated as underscores.
        destination: Destination country/region code, e.g. "DE", or a
            "City, CC" string as returned by dhl_tracking_mock -- only the
            trailing country code is matched.

    Returns:
        The matched rule as a dict, or None if no rule matches this
        category/destination combination.
    """
    category_key = category.strip().lower().replace(" ", "_")
    destination_key = destination.split(",")[-1].strip().upper()

    for rule in _COMPLIANCE_TABLE:
        if rule.category == category_key and rule.destination == destination_key:
            return rule.model_dump()

    logger.info("No compliance rule for category=%s destination=%s", category_key, destination_key)
    return None
