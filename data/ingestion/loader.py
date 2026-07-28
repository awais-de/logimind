"""Load and clean DHL PDFs from data/raw/ into page-level text."""

import logging
import re
from collections import Counter
from pathlib import Path

import fitz
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MIN_CHARS_PER_PAGE = 20
BOILERPLATE_LINE_THRESHOLD = 0.5


class PageText(BaseModel):
    """Cleaned text for a single page of a document.

    Attributes:
        page_number: 1-indexed page number within the source PDF.
        text: Cleaned page text.
    """

    page_number: int
    text: str


class LoadedDocument(BaseModel):
    """A PDF document loaded and cleaned into per-page text.

    Attributes:
        doc_id: Identifier matching the source's doc_id in data/sources.py.
        pages: Cleaned text for each usable page, in page order.
    """

    doc_id: str
    pages: list[PageText]


def _extract_raw_pages(pdf_path: Path) -> list[str]:
    """Extract raw per-page text from a PDF."""
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _find_boilerplate_lines(raw_pages: list[str]) -> set[str]:
    """Find lines repeated across most pages, e.g. running headers/footers."""
    if len(raw_pages) < 4:
        return set()

    line_counts: Counter[str] = Counter()
    for page_text in raw_pages:
        lines = {line.strip() for line in page_text.splitlines() if line.strip()}
        line_counts.update(lines)

    threshold = len(raw_pages) * BOILERPLATE_LINE_THRESHOLD
    return {line for line, count in line_counts.items() if count >= threshold}


def _clean_page(page_text: str, boilerplate_lines: set[str]) -> str:
    """Strip boilerplate lines and normalize whitespace for one page."""
    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip() and line.strip() not in boilerplate_lines
    ]
    text = "\n".join(lines)
    return re.sub(r"[ \t]+", " ", text).strip()


def load_document(doc_id: str, pdf_path: Path) -> LoadedDocument:
    """Load a single PDF into cleaned, page-level text.

    Args:
        doc_id: Identifier for the document, matching data/sources.py.
        pdf_path: Path to the source PDF.

    Returns:
        The loaded document with repeated headers/footers stripped and
        pages with no usable text dropped.
    """
    raw_pages = _extract_raw_pages(pdf_path)
    boilerplate_lines = _find_boilerplate_lines(raw_pages)

    pages: list[PageText] = []
    for page_number, raw_text in enumerate(raw_pages, start=1):
        cleaned = _clean_page(raw_text, boilerplate_lines)
        if len(cleaned) < MIN_CHARS_PER_PAGE:
            logger.warning("Dropping page %d of %s: no usable text", page_number, doc_id)
            continue
        pages.append(PageText(page_number=page_number, text=cleaned))

    return LoadedDocument(doc_id=doc_id, pages=pages)


def load_all(raw_dir: Path = RAW_DIR) -> list[LoadedDocument]:
    """Load every PDF in raw_dir into cleaned LoadedDocuments.

    Args:
        raw_dir: Directory containing downloaded PDFs, named <doc_id>.pdf.

    Returns:
        A LoadedDocument for each PDF found in raw_dir, sorted by doc_id.
    """
    return [
        load_document(pdf_path.stem, pdf_path)
        for pdf_path in sorted(raw_dir.glob("*.pdf"))
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for document in load_all():
        logger.info("Loaded %s: %d usable pages", document.doc_id, len(document.pages))
