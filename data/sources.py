"""DHL public document sources for the Layer 1 ingestion pipeline.

URLs below are placeholders pending manual verification.
"""

from datetime import date

from pydantic import BaseModel, HttpUrl


class DocumentSource(BaseModel):
    """A single DHL public document to be ingested.

    Attributes:
        doc_id: Stable slug used as the document's identifier throughout
            the pipeline (chunking, embedding, metadata index).
        name: Human-readable document title.
        url: Direct link to the PDF.
        category: Document type, used for metadata tagging and agent
            routing (e.g. "rate_guide", "annual_report", "customs").
        region: Country/region the document is specific to, if any.
        publication_date: Document publication date, if known.
    """

    doc_id: str
    name: str
    url: HttpUrl
    category: str
    region: str | None = None
    publication_date: date | None = None


SOURCES: list[DocumentSource] = [
    DocumentSource(
        doc_id="dhl-rate-guide-us",
        name="DHL Express Service & Rate Guide (US)",
        url="https://example.com/TODO/dhl-rate-guide-us.pdf",
        category="rate_guide",
        region="US",
    ),
    DocumentSource(
        doc_id="dhl-rate-guide-uk",
        name="DHL Express Service & Rate Guide (UK)",
        url="https://example.com/TODO/dhl-rate-guide-uk.pdf",
        category="rate_guide",
        region="UK",
    ),
    DocumentSource(
        doc_id="dhl-rate-guide-de",
        name="DHL Express Service & Rate Guide (DE)",
        url="https://example.com/TODO/dhl-rate-guide-de.pdf",
        category="rate_guide",
        region="DE",
    ),
    DocumentSource(
        doc_id="dhl-customs-guidelines",
        name="DHL Express Global Customs Customer Guidelines",
        url="https://example.com/TODO/dhl-customs-guidelines.pdf",
        category="customs",
    ),
    DocumentSource(
        doc_id="dhl-annual-report-2025",
        name="DHL Group Annual Report 2025",
        url="https://example.com/TODO/dhl-annual-report-2025.pdf",
        category="annual_report",
    ),
    DocumentSource(
        doc_id="dhl-annual-report-2024",
        name="DHL Group Annual Report 2024",
        url="https://example.com/TODO/dhl-annual-report-2024.pdf",
        category="annual_report",
    ),
    DocumentSource(
        doc_id="dhl-packing-guide",
        name="DHL Essential Packing Guide",
        url="https://example.com/TODO/dhl-packing-guide.pdf",
        category="packing_guide",
    ),
    DocumentSource(
        doc_id="dhl-prohibited-restricted-items",
        name="DHL Prohibited and Restricted Items Guide",
        url="https://example.com/TODO/dhl-prohibited-restricted-items.pdf",
        category="restricted_items",
    ),
    DocumentSource(
        doc_id="dhl-incoterms-2020",
        name="DHL Incoterms 2020 Guide",
        url="https://example.com/TODO/dhl-incoterms-2020.pdf",
        category="incoterms",
    ),
    DocumentSource(
        doc_id="dhl-us-customs-import-guide",
        name="DHL US Customs Import Industry Guide",
        url="https://example.com/TODO/dhl-us-customs-import-guide.pdf",
        category="customs",
        region="US",
    ),
    DocumentSource(
        doc_id="dhl-strategy-2030",
        name="DHL Group Strategy 2030",
        url="https://example.com/TODO/dhl-strategy-2030.pdf",
        category="strategy",
    ),
    DocumentSource(
        doc_id="dhl-sustainability-report-2024",
        name="DHL Sustainability Report 2024",
        url="https://example.com/TODO/dhl-sustainability-report-2024.pdf",
        category="sustainability",
    ),
]
