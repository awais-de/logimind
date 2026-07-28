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
        url="https://www.dhl.com/content/dam/dhl/global/dhl-express/documents/pdf/service-and-rate-guide-us-en.pdf",
        category="rate_guide",
        region="US",
    ),
    DocumentSource(
        doc_id="dhl-rate-guide-uk",
        name="DHL Express Service & Rate Guide (UK)",
        url="https://mydhl.express.dhl/content/dam/downloads/gb/en/rate-guide/service_and_rate_guide_gb_en.pdf.coredownload.pdf",
        category="rate_guide",
        region="UK",
    ),
    DocumentSource(
        doc_id="dhl-rate-guide-de",
        name="DHL Express Service & Rate Guide (DE)",
        url="https://www.dhl.de/dam/jcr:66008de1-5933-4497-a194-ab7f4626d671/dhl-express-produkte-services-und-preise-en.pdf",
        category="rate_guide",
        region="DE",
    ),
    DocumentSource(
        doc_id="dhl-customs-guidelines",
        name="DHL Express Global Customs Customer Guidelines",
        url="https://mydhl.express.dhl/content/dam/downloads/global/en/customs-guide/express_global_customs_customer_guidelines.pdf.coredownload.pdf",
        category="customs",
        publication_date=date(2026, 1, 1),
    ),
    DocumentSource(
        doc_id="dhl-annual-report-2025",
        name="DHL Group Annual Report 2025",
        url="https://group.dhl.com/content/dam/deutschepostdhl/en/media-center/investors/documents/annual-reports/DHL-Group-2025-Annual-Report.pdf",
        category="annual_report",
    ),
    DocumentSource(
        doc_id="dhl-annual-report-2024",
        name="DHL Group Annual Report 2024",
        url="https://group.dhl.com/content/dam/deutschepostdhl/en/media-center/investors/documents/annual-reports/DHL-Group-2024-Annual-Report.pdf",
        category="annual_report",
    ),
    DocumentSource(
        doc_id="dhl-packing-guide",
        name="DHL Essential Packing Guide",
        url="https://www.dhl.de/dam/jcr:ca1fbf3f-97d5-418a-9a95-2653c57eb3d6/dhl-express-packing-guide-en.pdf",
        category="packing_guide",
    ),
    DocumentSource(
        doc_id="dhl-prohibited-restricted-items",
        name="DHL Prohibited and Restricted Items Guide",
        url="https://www.dhlexpress.nl/sites/default/files/Verboden%20%20beperkt%20toegelaten%20items%20EN-V2-2024.pdf",
        category="restricted_items",
        region="NL",
    ),
    DocumentSource(
        doc_id="dhl-incoterms-2020",
        name="DHL Incoterms 2020 Guide",
        url="https://www.dhl.com/content/dam/dhl/global/dhl-global-forwarding/documents/pdf/glo-dgf-incoterms-2020-brochure.pdf",
        category="incoterms",
    ),
    DocumentSource(
        doc_id="dhl-us-customs-import-guide",
        name="DHL US Customs Import Industry Guide",
        url="https://www.dhl.com/discover/content/dam/global-master/6-shipping-with-dhl/importing-with-dhl/required-import-documents/Customs-Industry-Guide-October-2020.pdf",
        category="customs",
        region="US",
        publication_date=date(2020, 10, 1),
    ),
    DocumentSource(
        doc_id="dhl-strategy-2030",
        name="DHL Group Strategy 2030",
        url="https://group.dhl.com/content/dam/deutschepostdhl/en/media-relations/press-releases/2024/pr-dhl-group-strategy-2030-20240923.pdf",
        category="strategy",
        publication_date=date(2024, 9, 23),
    ),
    DocumentSource(
        doc_id="dhl-sustainability-report-2024",
        name="DHL Sustainability Report 2024",
        url="https://reporting-hub.group.dhl.com/ecomaXL/files/DHL-Group_2024-Progress-Report-on-Sustainability.pdf",
        category="sustainability",
    ),
    DocumentSource(
        doc_id="dhl-packing-guide-palletised-am",
        name="DHL Express Large Palletised Packing Guide (Americas, Imperial)",
        url="https://www.dhl.com/content/dam/dhl/local/us/dhl-global-forwarding/documents/pdf/dhl_express_large_palletised_packing_guide_am_en_imperial.pdf",
        category="packing_guide",
        region="AM",
    ),
    DocumentSource(
        doc_id="dhl-parcel-intl-conditions-uk",
        name="DHL Parcel UK International Parcel Conditions",
        url="https://www.dhl.com/content/dam/dhl/local/gb/dhl-parcel/documents/pdf/gb-parcel-international-parcel-conditions-september-2023.pdf",
        category="terms_conditions",
        region="UK",
        publication_date=date(2023, 9, 1),
    ),
]
